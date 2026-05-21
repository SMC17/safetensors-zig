//! Profile-style breakdown of where parse time goes on the real-model
//! fixture. Times each phase separately so we can identify the
//! bottleneck before optimization.
//!
//! Phases timed (in nanoseconds, mean over ITERS):
//!   - file read (already-cached after warmup)
//!   - openFromBytes whole-path
//!   - (decomposed) header u64 read
//!   - (decomposed) JSON Value parse only
//!   - (decomposed) tensor extraction loop only
//!   - (decomposed) sort + validation only
//!
//! The decomposed numbers may not sum to whole-path because of
//! allocation interaction; they're indicative, not exhaustive.

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
            std.debug.print("fixture missing: build with python3 tests/gen_fixture.py --zero\n", .{});
            std.process.exit(2);
        },
        else => return err,
    };
    defer alloc.free(bytes);

    // Pre-extract the header bytes for the decomposed timings.
    const header_size = std.mem.readInt(u64, bytes[0..8], .little);
    const data_len: u64 = bytes.len - 8 - header_size;
    const json_buf = bytes[8..][0..@intCast(header_size)];

    // Warmup
    {
        var st = try safetensors.openFromBytes(alloc, bytes);
        st.deinit();
    }

    // === Phase 1: whole-path openFromBytes ===
    {
        const t0 = std.Io.Clock.awake.now(io);
        var i: usize = 0;
        while (i < ITERS) : (i += 1) {
            var st = try safetensors.openFromBytes(alloc, bytes);
            st.deinit();
        }
        const t1 = std.Io.Clock.awake.now(io);
        const ns: u64 = @intCast(t0.durationTo(t1).nanoseconds);
        std.debug.print("phase whole_openFromBytes  ns_per_iter={d:>8}\n", .{ns / ITERS});
    }

    // === Phase 2: std.json.parseFromSliceLeaky only ===
    {
        const t0 = std.Io.Clock.awake.now(io);
        var i: usize = 0;
        while (i < ITERS) : (i += 1) {
            var arena = std.heap.ArenaAllocator.init(alloc);
            defer arena.deinit();
            const aa = arena.allocator();
            const v = try std.json.parseFromSliceLeaky(std.json.Value, aa, json_buf, .{});
            std.mem.doNotOptimizeAway(v);
        }
        const t1 = std.Io.Clock.awake.now(io);
        const ns: u64 = @intCast(t0.durationTo(t1).nanoseconds);
        std.debug.print("phase std_json_parse_only  ns_per_iter={d:>8}\n", .{ns / ITERS});
    }

    // === Phase 3: u64 read only ===
    {
        const t0 = std.Io.Clock.awake.now(io);
        var sink: u64 = 0;
        var i: usize = 0;
        while (i < ITERS) : (i += 1) {
            sink ^= std.mem.readInt(u64, bytes[0..8], .little);
        }
        const t1 = std.Io.Clock.awake.now(io);
        const ns: u64 = @intCast(t0.durationTo(t1).nanoseconds);
        std.debug.print("phase u64_header_read      ns_per_iter={d:>8} sink={d}\n", .{ ns / ITERS, sink });
    }

    // === Phase 4: arena alloc + free only ===
    {
        const t0 = std.Io.Clock.awake.now(io);
        var i: usize = 0;
        while (i < ITERS) : (i += 1) {
            var arena = std.heap.ArenaAllocator.init(alloc);
            arena.deinit();
        }
        const t1 = std.Io.Clock.awake.now(io);
        const ns: u64 = @intCast(t0.durationTo(t1).nanoseconds);
        std.debug.print("phase arena_init_deinit    ns_per_iter={d:>8}\n", .{ns / ITERS});
    }

    std.debug.print("fixture_bytes              {d}\n", .{bytes.len});
    std.debug.print("header_bytes               {d}\n", .{header_size});
    std.debug.print("data_bytes                 {d}\n", .{data_len});
}
