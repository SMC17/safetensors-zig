//! Hand-tuned single-pass JSON parser specialised for the safetensors
//! header format. Replaces std.json (84μs of 87μs total in v0.1) with
//! a parser that knows the exact safetensors object shape:
//!
//!   { "tensor_name": { "dtype": "F32", "shape": [...], "data_offsets": [a, b] },
//!     ...
//!     "__metadata__": { "key": "value", ... } }
//!
//! The parser is single-pass, byte-by-byte, with no AST construction
//! and no per-token allocation beyond the arena-backed output slices.
//! It accepts the de-facto-canonical safetensors format emitted by
//! `huggingface/safetensors` and rejects anything it doesn't
//! recognise with InvalidJSON.
//!
//! Strings: only `"` and `\\` escapes are recognised; other backslash
//! escapes (`\n`, `\u..`, etc.) return InvalidJSON. In practice
//! safetensors tensor names are ASCII identifiers and metadata values
//! are short strings — this constraint is well within the format's
//! observed usage.

const std = @import("std");
const root = @import("root.zig");

const DType = root.DType;
const Tensor = root.Tensor;
const MetadataEntry = root.MetadataEntry;
const Error = root.Error;
const ParserError = Error || std.mem.Allocator.Error;

pub const ParseResult = struct {
    tensors: []const Tensor,
    metadata: []const MetadataEntry,
};

const Cursor = struct {
    src: []const u8,
    pos: usize,

    inline fn peek(self: *const Cursor) ?u8 {
        if (self.pos >= self.src.len) return null;
        return self.src[self.pos];
    }

    inline fn next(self: *Cursor) ?u8 {
        if (self.pos >= self.src.len) return null;
        const c = self.src[self.pos];
        self.pos += 1;
        return c;
    }

    inline fn skipWhitespace(self: *Cursor) void {
        while (self.pos < self.src.len) {
            const c = self.src[self.pos];
            if (c == ' ' or c == '\t' or c == '\n' or c == '\r') {
                self.pos += 1;
            } else {
                return;
            }
        }
    }

    fn expect(self: *Cursor, c: u8) ParserError!void {
        self.skipWhitespace();
        const got = self.next() orelse return Error.InvalidJSON;
        if (got != c) return Error.InvalidJSON;
    }

    /// Parse a JSON string. Returns a slice that may either point
    /// directly into self.src (if no escape sequences) or be
    /// allocated in `aa` (if escapes were present).
    fn parseString(self: *Cursor, aa: std.mem.Allocator) ParserError![]const u8 {
        self.skipWhitespace();
        if ((self.next() orelse return Error.InvalidJSON) != '"') return Error.InvalidJSON;
        const start = self.pos;

        // Fast path: SIMD-accelerated scan for either `"` or `\\`.
        // 99%+ of safetensors tensor-name strings have no escape; the
        // happy path finds the closing `"` directly via indexOfAny
        // which lowers to a vectorised byte search on x86_64 / aarch64.
        const rest = self.src[self.pos..];
        const first_special = std.mem.indexOfAny(u8, rest, "\"\\") orelse
            return Error.InvalidJSON;
        const abs_first = self.pos + first_special;
        if (rest[first_special] == '"') {
            @branchHint(.likely);
            self.pos = abs_first + 1;
            return self.src[start..abs_first];
        }

        // Slow path: has at least one escape, walk byte by byte.
        self.pos = abs_first;
        while (self.pos < self.src.len) : (self.pos += 1) {
            const c = self.src[self.pos];
            if (c == '"') {
                const end = self.pos;
                self.pos += 1;
                // Allocate a copy with escapes resolved
                var buf = try aa.alloc(u8, end - start);
                var w: usize = 0;
                var r: usize = start;
                while (r < end) {
                    const b = self.src[r];
                    if (b == '\\') {
                        r += 1;
                        if (r >= end) return Error.InvalidJSON;
                        const esc = self.src[r];
                        buf[w] = switch (esc) {
                            '"' => '"',
                            '\\' => '\\',
                            '/' => '/',
                            else => return Error.InvalidJSON,
                        };
                        w += 1;
                        r += 1;
                    } else {
                        buf[w] = b;
                        w += 1;
                        r += 1;
                    }
                }
                return buf[0..w];
            } else if (c == '\\') {
                self.pos += 1;
                if (self.pos >= self.src.len) return Error.InvalidJSON;
            }
        }
        return Error.InvalidJSON;
    }

    /// Parse a non-negative integer literal.
    fn parseUnsigned(self: *Cursor) ParserError!u64 {
        self.skipWhitespace();
        var v: u64 = 0;
        var saw_digit = false;
        while (self.pos < self.src.len) {
            const c = self.src[self.pos];
            if (c < '0' or c > '9') break;
            saw_digit = true;
            v = v *% 10 + (c - '0');
            self.pos += 1;
        }
        if (!saw_digit) return Error.InvalidJSON;
        return v;
    }

    /// Skip a JSON value (object | array | string | number | true | false | null).
    /// Used when the parser doesn't care about the value, only that it's well-formed.
    fn skipValue(self: *Cursor) ParserError!void {
        self.skipWhitespace();
        const c = self.peek() orelse return Error.InvalidJSON;
        switch (c) {
            '{' => try self.skipObject(),
            '[' => try self.skipArray(),
            '"' => {
                // skip string
                _ = self.next();
                while (self.pos < self.src.len) {
                    const b = self.src[self.pos];
                    self.pos += 1;
                    if (b == '\\') {
                        if (self.pos >= self.src.len) return Error.InvalidJSON;
                        self.pos += 1;
                    } else if (b == '"') return;
                }
                return Error.InvalidJSON;
            },
            '0'...'9', '-' => {
                _ = self.next();
                while (self.pos < self.src.len) {
                    const b = self.src[self.pos];
                    if ((b >= '0' and b <= '9') or b == '.' or b == 'e' or b == 'E' or b == '+' or b == '-') {
                        self.pos += 1;
                    } else break;
                }
            },
            't' => {
                if (!std.mem.startsWith(u8, self.src[self.pos..], "true")) return Error.InvalidJSON;
                self.pos += 4;
            },
            'f' => {
                if (!std.mem.startsWith(u8, self.src[self.pos..], "false")) return Error.InvalidJSON;
                self.pos += 5;
            },
            'n' => {
                if (!std.mem.startsWith(u8, self.src[self.pos..], "null")) return Error.InvalidJSON;
                self.pos += 4;
            },
            else => return Error.InvalidJSON,
        }
    }

    fn skipObject(self: *Cursor) ParserError!void {
        try self.expect('{');
        self.skipWhitespace();
        if (self.peek() == @as(u8, '}')) {
            _ = self.next();
            return;
        }
        while (true) {
            // key
            self.skipWhitespace();
            if ((self.next() orelse return Error.InvalidJSON) != '"') return Error.InvalidJSON;
            while (self.pos < self.src.len) {
                const b = self.src[self.pos];
                self.pos += 1;
                if (b == '\\') {
                    if (self.pos >= self.src.len) return Error.InvalidJSON;
                    self.pos += 1;
                } else if (b == '"') break;
            }
            try self.expect(':');
            try self.skipValue();
            self.skipWhitespace();
            const next_c = self.next() orelse return Error.InvalidJSON;
            if (next_c == ',') continue;
            if (next_c == '}') return;
            return Error.InvalidJSON;
        }
    }

    fn skipArray(self: *Cursor) ParserError!void {
        try self.expect('[');
        self.skipWhitespace();
        if (self.peek() == @as(u8, ']')) {
            _ = self.next();
            return;
        }
        while (true) {
            try self.skipValue();
            self.skipWhitespace();
            const next_c = self.next() orelse return Error.InvalidJSON;
            if (next_c == ',') continue;
            if (next_c == ']') return;
            return Error.InvalidJSON;
        }
    }
};

/// Parse a safetensors JSON header into Tensor + MetadataEntry slices
/// allocated in `aa`. Validates against the data segment length.
pub fn parseHeader(
    aa: std.mem.Allocator,
    json: []const u8,
    data_len: u64,
) ParserError!ParseResult {
    var cur: Cursor = .{ .src = json, .pos = 0 };

    try cur.expect('{');

    var tensor_list: std.ArrayList(Tensor) = .empty;
    var meta_list: std.ArrayList(MetadataEntry) = .empty;

    // Pre-allocate based on a rough estimate of tensors-per-byte.
    // Real safetensors entries are ~80-120 bytes each; estimate 1
    // tensor per 100 bytes of header. Saves repeated grow+memcpy
    // during the parse loop.
    const estimated_tensors = @max(8, json.len / 100);
    try tensor_list.ensureTotalCapacity(aa, estimated_tensors);

    cur.skipWhitespace();
    if (cur.peek() == @as(u8, '}')) {
        _ = cur.next();
        return .{ .tensors = &.{}, .metadata = &.{} };
    }

    while (true) {
        const key = try cur.parseString(aa);
        try cur.expect(':');

        if (std.mem.eql(u8, key, "__metadata__")) {
            try parseMetadataObject(&cur, aa, &meta_list);
        } else {
            const t = try parseTensorObject(&cur, aa, key);
            try tensor_list.append(aa, t);
        }

        cur.skipWhitespace();
        const sep = cur.next() orelse return Error.InvalidJSON;
        if (sep == ',') {
            @branchHint(.likely);
            continue;
        }
        if (sep == '}') break;
        return Error.InvalidJSON;
    }

    // Sort + validate. Fast-path: HF emits tensors in offset-increasing
    // order, so we check first and skip the sort if already sorted.
    const tensors_owned = try tensor_list.toOwnedSlice(aa);
    var already_sorted = true;
    {
        var i: usize = 1;
        while (i < tensors_owned.len) : (i += 1) {
            if (tensors_owned[i - 1].data_offset_start > tensors_owned[i].data_offset_start) {
                already_sorted = false;
                break;
            }
        }
    }
    if (!already_sorted) {
        std.sort.pdq(Tensor, tensors_owned, {}, lessByStart);
    }

    var cursor: u64 = 0;
    for (tensors_owned) |t| {
        if (t.data_offset_start < cursor) return Error.OffsetOverlap;
        if (t.data_offset_end > data_len) return Error.OffsetOutOfBounds;
        const expected_bytes = t.numel() * @as(u64, @intCast(t.dtype.byteSize()));
        if (t.byteLen() != expected_bytes) return Error.InvalidHeader;
        cursor = t.data_offset_end;
    }

    const meta_owned = try meta_list.toOwnedSlice(aa);

    return .{ .tensors = tensors_owned, .metadata = meta_owned };
}

fn parseTensorObject(cur: *Cursor, aa: std.mem.Allocator, name: []const u8) ParserError!Tensor {
    try cur.expect('{');

    // safetensors tensor objects have exactly three keys: dtype, shape, data_offsets.
    // Order is canonical in HF output but we don't depend on it.
    var dtype: ?DType = null;
    var shape: ?[]u64 = null;
    var off_start: ?u64 = null;
    var off_end: ?u64 = null;

    cur.skipWhitespace();
    if (cur.peek() == @as(u8, '}')) {
        _ = cur.next();
        return Error.InvalidHeader; // empty tensor object — invalid
    }

    while (true) {
        const key = try cur.parseString(aa);
        try cur.expect(':');

        // First-letter dispatch: avoid full std.mem.eql for the common
        // {dtype, shape, data_offsets} keys.
        var handled = false;
        if (key.len > 0) {
            const c0 = key[0];
            if (c0 == 'd') {
                if (key.len == 5 and key[1] == 't') {
                    const s = try cur.parseString(aa);
                    dtype = DType.fromString(s) orelse return Error.UnknownDtype;
                    handled = true;
                } else if (key.len == 12 and key[1] == 'a') {
                    const pair = try parseOffsetPair(cur);
                    off_start = pair[0];
                    off_end = pair[1];
                    handled = true;
                }
            } else if (c0 == 's') {
                if (key.len == 5 and key[1] == 'h') {
                    shape = try parseShape(cur, aa);
                    handled = true;
                }
            }
        }
        if (!handled) {
            // Unknown key — skip its value (forward-compat).
            try cur.skipValue();
        }

        cur.skipWhitespace();
        const sep = cur.next() orelse return Error.InvalidJSON;
        if (sep == ',') continue;
        if (sep == '}') break;
        return Error.InvalidJSON;
    }

    if (dtype == null or shape == null or off_start == null or off_end == null) return Error.InvalidHeader;
    if (off_end.? < off_start.?) return Error.OffsetOutOfBounds;

    return .{
        .name = name,
        .dtype = dtype.?,
        .shape = shape.?,
        .data_offset_start = off_start.?,
        .data_offset_end = off_end.?,
    };
}

fn parseShape(cur: *Cursor, aa: std.mem.Allocator) ParserError![]u64 {
    try cur.expect('[');
    cur.skipWhitespace();
    if (cur.peek() == @as(u8, ']')) {
        _ = cur.next();
        return try aa.alloc(u64, 0);
    }
    var dims: std.ArrayList(u64) = .empty;
    while (true) {
        const d = try cur.parseUnsigned();
        try dims.append(aa, d);
        cur.skipWhitespace();
        const sep = cur.next() orelse return Error.InvalidJSON;
        if (sep == ',') continue;
        if (sep == ']') break;
        return Error.InvalidJSON;
    }
    return try dims.toOwnedSlice(aa);
}

fn parseOffsetPair(cur: *Cursor) ParserError![2]u64 {
    try cur.expect('[');
    const a = try cur.parseUnsigned();
    try cur.expect(',');
    const b = try cur.parseUnsigned();
    try cur.expect(']');
    return .{ a, b };
}

fn parseMetadataObject(
    cur: *Cursor,
    aa: std.mem.Allocator,
    list: *std.ArrayList(MetadataEntry),
) ParserError!void {
    try cur.expect('{');
    cur.skipWhitespace();
    if (cur.peek() == @as(u8, '}')) {
        _ = cur.next();
        return;
    }
    while (true) {
        const key = try cur.parseString(aa);
        try cur.expect(':');
        // Value must be a string in safetensors metadata.
        const val = try cur.parseString(aa);
        try list.append(aa, .{ .key = key, .value = val });
        cur.skipWhitespace();
        const sep = cur.next() orelse return Error.InvalidJSON;
        if (sep == ',') continue;
        if (sep == '}') return;
        return Error.InvalidJSON;
    }
}

fn lessByStart(_: void, x: Tensor, y: Tensor) bool {
    return x.data_offset_start < y.data_offset_start;
}

// ============================================================================
// Tests — these complement the test suite in root.zig by exercising the
// fast-path parser against the same fixtures.
// ============================================================================

const testing = std.testing;

test "fast parser — minimal one tensor" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const json =
        \\{"w":{"dtype":"F32","shape":[1,1],"data_offsets":[0,4]}}
    ;
    const r = try parseHeader(aa, json, 4);
    try testing.expectEqual(@as(usize, 1), r.tensors.len);
    try testing.expectEqualStrings("w", r.tensors[0].name);
    try testing.expectEqual(DType.f32, r.tensors[0].dtype);
    try testing.expectEqualSlices(u64, &[_]u64{ 1, 1 }, r.tensors[0].shape);
    try testing.expectEqual(@as(u64, 0), r.tensors[0].data_offset_start);
    try testing.expectEqual(@as(u64, 4), r.tensors[0].data_offset_end);
}

test "fast parser — metadata" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const json =
        \\{"__metadata__":{"author":"stax","framework":"pt"},"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}
    ;
    const r = try parseHeader(aa, json, 4);
    try testing.expectEqual(@as(usize, 1), r.tensors.len);
    try testing.expectEqual(@as(usize, 2), r.metadata.len);
}

test "fast parser — rejects bad dtype" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const json =
        \\{"w":{"dtype":"F999","shape":[1],"data_offsets":[0,4]}}
    ;
    try testing.expectError(Error.UnknownDtype, parseHeader(aa, json, 4));
}

test "fast parser — handles whitespace" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const json =
        \\{
        \\  "w": {
        \\    "dtype": "F32",
        \\    "shape": [2, 2],
        \\    "data_offsets": [0, 16]
        \\  }
        \\}
    ;
    const r = try parseHeader(aa, json, 16);
    try testing.expectEqual(@as(usize, 1), r.tensors.len);
    try testing.expectEqualSlices(u64, &[_]u64{ 2, 2 }, r.tensors[0].shape);
}

test "fast parser — escaped quote in tensor name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    // Name contains an escaped quote: foo\"bar -> foo"bar
    const json =
        \\{"foo\"bar":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}
    ;
    const r = try parseHeader(aa, json, 4);
    try testing.expectEqual(@as(usize, 1), r.tensors.len);
    try testing.expectEqualStrings("foo\"bar", r.tensors[0].name);
}

test "fast parser — unknown key in tensor object skipped" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const json =
        \\{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"extra":"ignored"}}
    ;
    const r = try parseHeader(aa, json, 4);
    try testing.expectEqual(@as(usize, 1), r.tensors.len);
}
