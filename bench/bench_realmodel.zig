//! Real-model fixture parse bench (Zig side of the Zig-vs-Rust
//! comparison).
//!
//! Reads tests/fixtures/tinyllama-shape.safetensors entirely into
//! memory once, then runs N parse iterations from that buffer to
//! isolate parse cost from disk I/O. Reports ns/parse and
//! parses/sec.
//!
//! The companion Rust bench at bench/bench_rust/ uses the official
//! `safetensors` crate on the same fixture for a fair head-to-head.

const std = @import("std");
const safetensors = @import("safetensors");

const ITERS = 1000;
const FIXTURE = "tests/fixtures/tinyllama-shape.safetensors";

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    const bytes = std.Io.Dir.cwd().readFileAlloc(
        io,
        FIXTURE,
        alloc,
        std.Io.Limit.limited(1024 * 1024 * 1024),
    ) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print(
                "fixture {s} not present. Build it with:\n  python3 tests/gen_fixture.py --zero\n",
                .{FIXTURE},
            );
            std.process.exit(2);
        },
        else => return err,
    };
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
    const parses_per_sec = @as(u64, ITERS) * std.time.ns_per_s / elapsed_ns;

    std.debug.print("fixture        {s}\n", .{FIXTURE});
    std.debug.print("file_bytes     {d}\n", .{bytes.len});
    std.debug.print("iters          {d}\n", .{ITERS});
    std.debug.print("ns_per_parse   {d}\n", .{ns_per_iter});
    std.debug.print("us_per_parse   {d}\n", .{ns_per_iter / 1000});
    std.debug.print("parses_per_sec {d}\n", .{parses_per_sec});
}
