//! Real-model parse bench on the full TinyLlama-1.1B 2.2GB fixture.
//!
//! Companion to bench_realmodel.zig (which uses the 33 MB synthetic
//! Llama-shape fixture). This one reads the actual checkpoint that
//! ships with the repo at tests/fixtures/tinyllama-real/model.safetensors
//! and measures parse throughput on the full 201-tensor header.
//!
//! HF safetensors reference measured 2026-05-28 on the same fixture:
//!   cold open+keys: 1553 ms
//!   warm open+keys: 1.0 - 1.1 ms (3-trial median)
//!
//! Read once into a 2.2 GB buffer outside the timing loop so we measure
//! parse cost (JSON header scan + offset extraction) not disk I/O.

const std = @import("std");
const safetensors = @import("safetensors");

const ITERS = 1000;
const FIXTURE = "tests/fixtures/tinyllama-real/model.safetensors";
// 3 GB upper bound so the 2.2 GB file fits.
const READ_LIMIT_BYTES: u64 = 3 * 1024 * 1024 * 1024;

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    const bytes = std.Io.Dir.cwd().readFileAlloc(
        io,
        FIXTURE,
        alloc,
        std.Io.Limit.limited(READ_LIMIT_BYTES),
    ) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print(
                "fixture {s} not present. The 2.2 GB checkpoint is in tests/fixtures/tinyllama-real/.\nIf missing, fetch via the FIXTURES.md instructions.\n",
                .{FIXTURE},
            );
            std.process.exit(2);
        },
        else => return err,
    };
    defer alloc.free(bytes);

    // Warmup — once to populate caches.
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
    const parses_per_sec = @as(u64, ITERS) * std.time.ns_per_s / elapsed_ns;

    std.debug.print("fixture          {s}\n", .{FIXTURE});
    std.debug.print("file_bytes       {d}\n", .{bytes.len});
    std.debug.print("iters            {d}\n", .{ITERS});
    std.debug.print("ns_per_parse     {d}\n", .{ns_per_iter});
    std.debug.print("us_per_parse     {d}\n", .{ns_per_iter / 1000});
    std.debug.print("parses_per_sec   {d}\n", .{parses_per_sec});
    std.debug.print("\n", .{});
    std.debug.print("Reference (HF safetensors Python, warm cache 3-trial median): ~1.0 ms\n", .{});
    std.debug.print("Reference (HF safetensors Python, cold cache):                ~1553 ms\n", .{});
}
