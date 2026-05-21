//! safetensors-inspect — print the schema of a .safetensors file.
//!
//! Usage: safetensors-inspect <path>

const std = @import("std");
const safetensors = @import("safetensors");

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, alloc);
    defer iter.deinit();
    _ = iter.next(); // skip exe name
    const path = iter.next() orelse {
        std.debug.print("usage: safetensors-inspect <path>\n", .{});
        std.process.exit(2);
    };

    const bytes = try std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        alloc,
        std.Io.Limit.limited(8 * 1024 * 1024 * 1024),
    );
    defer alloc.free(bytes);

    var st = try safetensors.openFromBytes(alloc, bytes);
    defer st.deinit();

    std.debug.print("FILE: {s}\n", .{path});
    std.debug.print("HEADER: {d} bytes\n", .{st.header_size});
    std.debug.print("DATA:   {d} bytes\n", .{bytes.len - st.data_start});
    std.debug.print("TENSORS: {d}\n", .{st.tensors.len});
    std.debug.print("METADATA: {d}\n", .{st.metadata.len});
    for (st.metadata) |m| {
        std.debug.print("  {s}: {s}\n", .{ m.key, m.value });
    }
    std.debug.print("----\n", .{});

    for (st.tensors) |t| {
        std.debug.print("{s}  {s}  [", .{ t.name, t.dtype.toString() });
        for (t.shape, 0..) |d, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{d}", .{d});
        }
        std.debug.print("]  {d}..{d}  numel={d}\n", .{
            t.data_offset_start,
            t.data_offset_end,
            t.numel(),
        });
    }
}
