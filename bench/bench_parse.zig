//! Header-parse throughput bench.
//!
//! Builds a synthetic safetensors byte stream with K tensors entirely
//! in-memory, then parses+deinits in a loop. Reports tensors-per-second
//! and ns-per-parse for the header-parse-and-validate path. This is a
//! self-bench, NOT a comparison against the upstream Rust crate.

const std = @import("std");
const safetensors = @import("safetensors");

const ITERS = 1000;
const TENSOR_COUNT = 256;

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    // Build a synthetic header with TENSOR_COUNT F32 [4,4] entries.
    var header_buf: std.ArrayList(u8) = .empty;
    defer header_buf.deinit(alloc);
    try header_buf.appendSlice(alloc, "{");
    var offset: u64 = 0;
    var i: usize = 0;
    while (i < TENSOR_COUNT) : (i += 1) {
        if (i > 0) try header_buf.appendSlice(alloc, ",");
        var name_buf: [128]u8 = undefined;
        const chunk = try std.fmt.bufPrint(
            &name_buf,
            "\"t{d}\":{{\"dtype\":\"F32\",\"shape\":[4,4],\"data_offsets\":[{d},{d}]}}",
            .{ i, offset, offset + 64 },
        );
        try header_buf.appendSlice(alloc, chunk);
        offset += 64;
    }
    try header_buf.appendSlice(alloc, "}");

    const data_len: u64 = TENSOR_COUNT * 64;
    const data_buf = try alloc.alloc(u8, data_len);
    defer alloc.free(data_buf);
    @memset(data_buf, 0);

    const bytes = try safetensors.buildBytes(alloc, header_buf.items, data_buf);
    defer alloc.free(bytes);

    // Warmup
    {
        var st = try safetensors.openFromBytes(alloc, bytes);
        st.deinit();
    }

    const start_ts = std.Io.Clock.awake.now(io);
    var iter: usize = 0;
    while (iter < ITERS) : (iter += 1) {
        var st = try safetensors.openFromBytes(alloc, bytes);
        st.deinit();
    }
    const end_ts = std.Io.Clock.awake.now(io);

    const elapsed_ns: u64 = @intCast(start_ts.durationTo(end_ts).nanoseconds);
    const ns_per_iter = elapsed_ns / ITERS;
    const tensors_total = ITERS * TENSOR_COUNT;
    const tensors_per_sec = @as(u64, tensors_total) * std.time.ns_per_s / elapsed_ns;

    std.debug.print("iters             {d}\n", .{ITERS});
    std.debug.print("tensors_per_file  {d}\n", .{TENSOR_COUNT});
    std.debug.print("header_bytes      {d}\n", .{header_buf.items.len});
    std.debug.print("ns_per_parse      {d}\n", .{ns_per_iter});
    std.debug.print("us_per_parse      {d}\n", .{ns_per_iter / 1000});
    std.debug.print("tensors_per_sec   {d}\n", .{tensors_per_sec});
}
