//! safetensors-zig — Zig 0.16 reader for the HuggingFace safetensors
//! tensor serialisation format.
//!
//! Format (per the upstream spec):
//!   bytes [0, 8)        : little-endian u64 = N (header length in bytes)
//!   bytes [8, 8+N)      : UTF-8 JSON header
//!                          - top-level object keyed by tensor name
//!                          - each value = { dtype: string, shape: [int...],
//!                            data_offsets: [start, end] }
//!                          - optional "__metadata__" : { string -> string }
//!   bytes [8+N, EOF)    : tensor data, contiguous, offsets relative to
//!                         the data segment start (not the file start)
//!
//! Two entry points:
//!   - `openFromBytes(allocator, bytes)` — parse from an in-memory buffer.
//!     Tensor data is borrowed from `bytes`; no file handle held.
//!   - `open(allocator, io, path)` — open a file via the Zig 0.16 Io
//!     abstraction. Tensor data is read lazily via `readTensorBytes`.
//!
//! This implementation validates:
//!   - header length fits the file
//!   - JSON parses to an object whose keys are tensor objects
//!   - dtype is a recognised safetensors dtype literal
//!   - shape entries are non-negative integers
//!   - data_offsets form a non-overlapping cover of the data segment
//!   - tensor byteLen == numel * dtype.byteSize()
//!   - tensor bytes lie within the data segment

const std = @import("std");

pub const Error = error{
    InvalidHeader,
    HeaderTooLarge,
    UnknownDtype,
    InvalidShape,
    OffsetOutOfBounds,
    OffsetOverlap,
    Truncated,
    InvalidJSON,
};

/// Header guard. Real-world safetensors headers are O(MB) at most.
/// 100 MB is well above any production header and prevents an attacker
/// from forcing the loader to allocate gigabytes from a single u64 field.
pub const max_header_bytes: u64 = 100 * 1024 * 1024;

/// safetensors dtype enum. Names match the upstream string literals
/// (BOOL, U8, I8, F8_E5M2, F8_E4M3, U16, I16, F16, BF16, U32, I32, F32,
/// U64, I64, F64).
pub const DType = enum {
    @"bool",
    u8,
    i8,
    f8_e5m2,
    f8_e4m3,
    u16,
    i16,
    f16,
    bf16,
    u32,
    i32,
    f32,
    u64,
    i64,
    f64,

    pub fn fromString(s: []const u8) ?DType {
        const Pair = struct { name: []const u8, dtype: DType };
        const table = [_]Pair{
            .{ .name = "BOOL", .dtype = .@"bool" },
            .{ .name = "U8", .dtype = .u8 },
            .{ .name = "I8", .dtype = .i8 },
            .{ .name = "F8_E5M2", .dtype = .f8_e5m2 },
            .{ .name = "F8_E4M3", .dtype = .f8_e4m3 },
            .{ .name = "U16", .dtype = .u16 },
            .{ .name = "I16", .dtype = .i16 },
            .{ .name = "F16", .dtype = .f16 },
            .{ .name = "BF16", .dtype = .bf16 },
            .{ .name = "U32", .dtype = .u32 },
            .{ .name = "I32", .dtype = .i32 },
            .{ .name = "F32", .dtype = .f32 },
            .{ .name = "U64", .dtype = .u64 },
            .{ .name = "I64", .dtype = .i64 },
            .{ .name = "F64", .dtype = .f64 },
        };
        for (table) |p| {
            if (std.mem.eql(u8, s, p.name)) return p.dtype;
        }
        return null;
    }

    pub fn toString(self: DType) []const u8 {
        return switch (self) {
            .@"bool" => "BOOL",
            .u8 => "U8",
            .i8 => "I8",
            .f8_e5m2 => "F8_E5M2",
            .f8_e4m3 => "F8_E4M3",
            .u16 => "U16",
            .i16 => "I16",
            .f16 => "F16",
            .bf16 => "BF16",
            .u32 => "U32",
            .i32 => "I32",
            .f32 => "F32",
            .u64 => "U64",
            .i64 => "I64",
            .f64 => "F64",
        };
    }

    pub fn byteSize(self: DType) usize {
        return switch (self) {
            .@"bool", .u8, .i8, .f8_e5m2, .f8_e4m3 => 1,
            .u16, .i16, .f16, .bf16 => 2,
            .u32, .i32, .f32 => 4,
            .u64, .i64, .f64 => 8,
        };
    }
};

pub const Tensor = struct {
    name: []const u8,
    dtype: DType,
    shape: []const u64,
    /// Offset within the data segment (NOT within the file).
    data_offset_start: u64,
    data_offset_end: u64,

    pub fn numel(self: Tensor) u64 {
        var n: u64 = 1;
        for (self.shape) |d| n *= d;
        return n;
    }

    pub fn byteLen(self: Tensor) u64 {
        return self.data_offset_end - self.data_offset_start;
    }
};

pub const MetadataEntry = struct {
    key: []const u8,
    value: []const u8,
};

/// Internal header-parse result, shared by openFromBytes and open.
const ParsedHeader = struct {
    arena_state: *std.heap.ArenaAllocator,
    header_size: u64,
    data_start: u64,
    data_len: u64,
    tensors: []const Tensor,
    metadata: []const MetadataEntry,

    fn deinit(self: *ParsedHeader, alloc: std.mem.Allocator) void {
        self.arena_state.deinit();
        alloc.destroy(self.arena_state);
    }
};

/// Result of parsing a safetensors file from an in-memory byte slice.
/// `tensor_bytes(t)` returns a borrowed slice into the original buffer.
pub const Parsed = struct {
    allocator: std.mem.Allocator,
    arena_state: *std.heap.ArenaAllocator,
    bytes: []const u8,
    header_size: u64,
    data_start: u64,
    tensors: []const Tensor,
    metadata: []const MetadataEntry,

    pub fn deinit(self: *Parsed) void {
        self.arena_state.deinit();
        self.allocator.destroy(self.arena_state);
    }

    pub fn get(self: *const Parsed, name: []const u8) ?*const Tensor {
        for (self.tensors) |*t| {
            if (std.mem.eql(u8, t.name, name)) return t;
        }
        return null;
    }

    /// Returns a slice into the underlying buffer for the tensor's raw bytes.
    /// Lifetime is tied to the buffer the caller passed to openFromBytes.
    pub fn tensorBytes(self: *const Parsed, tensor: *const Tensor) []const u8 {
        const start = self.data_start + tensor.data_offset_start;
        const end = self.data_start + tensor.data_offset_end;
        return self.bytes[start..end];
    }
};

/// Result of opening a safetensors file by path. Tensor data is read
/// lazily via readTensorBytes / readTensorBytesAlloc.
pub const SafeTensors = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    arena_state: *std.heap.ArenaAllocator,
    file: std.Io.File,
    file_size: u64,
    header_size: u64,
    /// Absolute file offset where tensor data starts (= 8 + header_size).
    data_start: u64,
    tensors: []const Tensor,
    metadata: []const MetadataEntry,

    pub fn open(
        allocator: std.mem.Allocator,
        io: std.Io,
        path: []const u8,
    ) !SafeTensors {
        const file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only });
        errdefer file.close(io);
        const file_size = try file.length(io);
        if (file_size < 8) return Error.InvalidHeader;

        var hdr_size_buf: [8]u8 = undefined;
        const n1 = try file.readPositionalAll(io, &hdr_size_buf, 0);
        if (n1 != 8) return Error.Truncated;
        const header_size = std.mem.readInt(u64, &hdr_size_buf, .little);
        if (header_size > file_size - 8) return Error.HeaderTooLarge;
        if (header_size > max_header_bytes) return Error.HeaderTooLarge;

        const arena_state = try allocator.create(std.heap.ArenaAllocator);
        errdefer allocator.destroy(arena_state);
        arena_state.* = std.heap.ArenaAllocator.init(allocator);
        errdefer arena_state.deinit();
        const aa = arena_state.allocator();

        const json_buf = try aa.alloc(u8, @intCast(header_size));
        const n2 = try file.readPositionalAll(io, json_buf, 8);
        if (n2 != header_size) return Error.Truncated;

        const data_start: u64 = 8 + header_size;
        const data_len = file_size - data_start;

        const tensors, const metadata = try parseHeaderJson(aa, json_buf, data_len);

        return .{
            .allocator = allocator,
            .io = io,
            .arena_state = arena_state,
            .file = file,
            .file_size = file_size,
            .header_size = header_size,
            .data_start = data_start,
            .tensors = tensors,
            .metadata = metadata,
        };
    }

    pub fn deinit(self: *SafeTensors) void {
        self.file.close(self.io);
        self.arena_state.deinit();
        self.allocator.destroy(self.arena_state);
    }

    pub fn get(self: *const SafeTensors, name: []const u8) ?*const Tensor {
        for (self.tensors) |*t| {
            if (std.mem.eql(u8, t.name, name)) return t;
        }
        return null;
    }

    /// Read tensor bytes into the caller-provided buffer. Buffer length
    /// must equal tensor.byteLen() exactly.
    pub fn readTensorBytes(
        self: *SafeTensors,
        tensor: *const Tensor,
        buf: []u8,
    ) !void {
        if (buf.len != tensor.byteLen()) return Error.Truncated;
        const offset = self.data_start + tensor.data_offset_start;
        const n = try self.file.readPositionalAll(self.io, buf, offset);
        if (n != buf.len) return Error.Truncated;
    }

    /// Like readTensorBytes but allocates the buffer for the caller.
    pub fn readTensorBytesAlloc(
        self: *SafeTensors,
        alloc: std.mem.Allocator,
        tensor: *const Tensor,
    ) ![]u8 {
        const buf = try alloc.alloc(u8, @intCast(tensor.byteLen()));
        errdefer alloc.free(buf);
        try self.readTensorBytes(tensor, buf);
        return buf;
    }
};

/// Parse a safetensors file from an in-memory byte slice. The returned
/// `Parsed` borrows from `bytes`; do not free `bytes` until calling
/// `Parsed.deinit()`.
pub fn openFromBytes(
    allocator: std.mem.Allocator,
    bytes: []const u8,
) !Parsed {
    if (bytes.len < 8) return Error.InvalidHeader;
    const header_size = std.mem.readInt(u64, bytes[0..8], .little);
    if (header_size > bytes.len - 8) return Error.HeaderTooLarge;
    if (header_size > max_header_bytes) return Error.HeaderTooLarge;

    const arena_state = try allocator.create(std.heap.ArenaAllocator);
    errdefer allocator.destroy(arena_state);
    arena_state.* = std.heap.ArenaAllocator.init(allocator);
    errdefer arena_state.deinit();
    const aa = arena_state.allocator();

    const data_start: u64 = 8 + header_size;
    const data_len: u64 = bytes.len - data_start;
    const json_buf = bytes[8..][0..@intCast(header_size)];

    const tensors, const metadata = try parseHeaderJson(aa, json_buf, data_len);

    return .{
        .allocator = allocator,
        .arena_state = arena_state,
        .bytes = bytes,
        .header_size = header_size,
        .data_start = data_start,
        .tensors = tensors,
        .metadata = metadata,
    };
}

/// Parse the JSON header buffer via the hand-tuned safetensors-specific
/// parser in src/parser.zig. Replaces the v0.1 std.json path (which
/// was 97% of total parse time on the Llama-shape fixture).
fn parseHeaderJson(
    aa: std.mem.Allocator,
    json_buf: []const u8,
    data_len: u64,
) !struct { []const Tensor, []const MetadataEntry } {
    const parser = @import("parser.zig");
    const r = try parser.parseHeader(aa, json_buf, data_len);
    return .{ r.tensors, r.metadata };
}

fn lessByStart(_: void, x: Tensor, y: Tensor) bool {
    return x.data_offset_start < y.data_offset_start;
}

/// Helper for tests + benches: build a safetensors byte stream from
/// (header JSON, data) pair. Caller owns the returned slice.
pub fn buildBytes(
    alloc: std.mem.Allocator,
    header_json: []const u8,
    data: []const u8,
) ![]u8 {
    const total = 8 + header_json.len + data.len;
    const buf = try alloc.alloc(u8, total);
    std.mem.writeInt(u64, buf[0..8], header_json.len, .little);
    @memcpy(buf[8..][0..header_json.len], header_json);
    @memcpy(buf[8 + header_json.len ..][0..data.len], data);
    return buf;
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "DType.fromString round-trip" {
    try testing.expectEqual(@as(?DType, .f32), DType.fromString("F32"));
    try testing.expectEqual(@as(?DType, .bf16), DType.fromString("BF16"));
    try testing.expectEqual(@as(?DType, .f8_e4m3), DType.fromString("F8_E4M3"));
    try testing.expectEqual(@as(?DType, null), DType.fromString("NOPE"));
}

test "DType.byteSize covers every variant" {
    try testing.expectEqual(@as(usize, 1), DType.@"bool".byteSize());
    try testing.expectEqual(@as(usize, 1), DType.u8.byteSize());
    try testing.expectEqual(@as(usize, 1), DType.f8_e5m2.byteSize());
    try testing.expectEqual(@as(usize, 2), DType.f16.byteSize());
    try testing.expectEqual(@as(usize, 2), DType.bf16.byteSize());
    try testing.expectEqual(@as(usize, 4), DType.f32.byteSize());
    try testing.expectEqual(@as(usize, 8), DType.f64.byteSize());
}

test "DType.toString round-trips with fromString" {
    inline for (@typeInfo(DType).@"enum".fields) |f| {
        const dt: DType = @enumFromInt(f.value);
        const s = dt.toString();
        try testing.expectEqual(@as(?DType, dt), DType.fromString(s));
    }
}

test "parse minimal one-tensor file (F32 1x1)" {
    const alloc = testing.allocator;
    const data: [4]u8 = .{ 0x00, 0x00, 0x80, 0x3f }; // 1.0 little-endian
    const header =
        \\{"w":{"dtype":"F32","shape":[1,1],"data_offsets":[0,4]}}
    ;
    const bytes = try buildBytes(alloc, header, &data);
    defer alloc.free(bytes);

    var st = try openFromBytes(alloc, bytes);
    defer st.deinit();

    try testing.expectEqual(@as(usize, 1), st.tensors.len);
    const t = st.get("w") orelse return error.TestExpectedTensor;
    try testing.expectEqual(DType.f32, t.dtype);
    try testing.expectEqualSlices(u64, &[_]u64{ 1, 1 }, t.shape);
    try testing.expectEqual(@as(u64, 4), t.byteLen());
    try testing.expectEqualSlices(u8, &data, st.tensorBytes(t));
}

test "parse two-tensor file with metadata" {
    const alloc = testing.allocator;
    var data_buf: [12]u8 = undefined;
    @memset(&data_buf, 0);
    const header =
        \\{"__metadata__":{"author":"stax"},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F64","shape":[1],"data_offsets":[4,12]}}
    ;
    const bytes = try buildBytes(alloc, header, &data_buf);
    defer alloc.free(bytes);

    var st = try openFromBytes(alloc, bytes);
    defer st.deinit();

    try testing.expectEqual(@as(usize, 2), st.tensors.len);
    try testing.expect(st.get("a") != null);
    try testing.expect(st.get("b") != null);
    try testing.expectEqual(@as(usize, 1), st.metadata.len);
    try testing.expectEqualStrings("author", st.metadata[0].key);
    try testing.expectEqualStrings("stax", st.metadata[0].value);
}

test "rejects truncated header-size byte" {
    const alloc = testing.allocator;
    const bytes = [_]u8{ 0x00, 0x00, 0x00 }; // <8 bytes
    try testing.expectError(Error.InvalidHeader, openFromBytes(alloc, &bytes));
}

test "rejects header-larger-than-file" {
    const alloc = testing.allocator;
    var bytes: [16]u8 = undefined;
    std.mem.writeInt(u64, bytes[0..8], 9999, .little);
    @memset(bytes[8..], 0);
    try testing.expectError(Error.HeaderTooLarge, openFromBytes(alloc, &bytes));
}

test "rejects unknown dtype" {
    const alloc = testing.allocator;
    const data: [4]u8 = .{ 0, 0, 0, 0 };
    const header =
        \\{"w":{"dtype":"F999","shape":[1],"data_offsets":[0,4]}}
    ;
    const bytes = try buildBytes(alloc, header, &data);
    defer alloc.free(bytes);
    try testing.expectError(Error.UnknownDtype, openFromBytes(alloc, bytes));
}

test "rejects byteLen mismatch with shape*dtype" {
    const alloc = testing.allocator;
    var data_buf: [8]u8 = undefined;
    @memset(&data_buf, 0);
    const header =
        \\{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}
    ;
    const bytes = try buildBytes(alloc, header, &data_buf);
    defer alloc.free(bytes);
    try testing.expectError(Error.InvalidHeader, openFromBytes(alloc, bytes));
}

test "rejects overlapping data offsets" {
    const alloc = testing.allocator;
    var data_buf: [8]u8 = undefined;
    @memset(&data_buf, 0);
    const header =
        \\{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[2,6]}}
    ;
    const bytes = try buildBytes(alloc, header, &data_buf);
    defer alloc.free(bytes);
    try testing.expectError(Error.OffsetOverlap, openFromBytes(alloc, bytes));
}

test "rejects offset extending past data segment" {
    const alloc = testing.allocator;
    var data_buf: [4]u8 = undefined;
    @memset(&data_buf, 0);
    const header =
        \\{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}
    ;
    const bytes = try buildBytes(alloc, header, &data_buf);
    defer alloc.free(bytes);
    try testing.expectError(Error.OffsetOutOfBounds, openFromBytes(alloc, bytes));
}

test "rejects invalid JSON header" {
    const alloc = testing.allocator;
    const data: [4]u8 = .{ 0, 0, 0, 0 };
    const header = "{not json at all";
    const bytes = try buildBytes(alloc, header, &data);
    defer alloc.free(bytes);
    try testing.expectError(Error.InvalidJSON, openFromBytes(alloc, bytes));
}

test "empty file (no tensors, just __metadata__) parses" {
    const alloc = testing.allocator;
    const header =
        \\{"__metadata__":{"version":"1"}}
    ;
    const bytes = try buildBytes(alloc, header, &[_]u8{});
    defer alloc.free(bytes);
    var st = try openFromBytes(alloc, bytes);
    defer st.deinit();
    try testing.expectEqual(@as(usize, 0), st.tensors.len);
    try testing.expectEqual(@as(usize, 1), st.metadata.len);
}
