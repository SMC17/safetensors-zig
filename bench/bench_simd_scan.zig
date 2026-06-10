//! Standalone structural-scan throughput bench.
//!
//! safetensors-zig v0.3 parser uses std.mem.indexOfAny to find the
//! next `"` or `\\` byte in the JSON string body — SWAR-vectorized
//! by the std lib. The v0.4 simdjson-style path would build a full
//! structural-character bitmask in one @Vector(32, u8) sweep, then
//! drive parsing from the bitmask.
//!
//! This bench compares:
//!   - indexOfAny over the full TinyLlama-shape header (~5 KB JSON)
//!   - @Vector(32, u8) explicit structural-scan producing bitmask
//!
//! Hypothesis: explicit @Vector(32, u8) compare + reduce wins on
//! Apple Silicon NEON because the bitmask is one shot per chunk vs
//! indexOfAny's incremental scan with branches.

const std = @import("std");

const FIXTURE_HEADER_BYTES: comptime_int = 5000;
const ITERS: usize = 100_000;
const SEED: u64 = 42;

// Structural chars: {, }, [, ], ",", :, ", \
const STRUCT_CHARS = "{}[],:\"\\";

fn scanIndexOfAny(src: []const u8) usize {
    var count: usize = 0;
    var pos: usize = 0;
    while (pos < src.len) {
        const rest = src[pos..];
        const idx = std.mem.indexOfAny(u8, rest, STRUCT_CHARS) orelse break;
        count += 1;
        pos += idx + 1;
    }
    return count;
}

const Vec32 = @Vector(32, u8);

fn scanVector32(src: []const u8) usize {
    var count: usize = 0;
    var i: usize = 0;
    while (i + 32 <= src.len) : (i += 32) {
        const chunk: Vec32 = src[i..][0..32].*;
        // Test against each structural char: produce 1 mask per char, OR them
        var any_mask: @Vector(32, bool) = @splat(false);
        inline for ("{}[],:\"\\") |c| {
            const c_v: Vec32 = @splat(c);
            const eq = chunk == c_v;
            any_mask = @select(bool, any_mask, @as(@Vector(32, bool), @splat(true)), eq);
        }
        // Count true lanes
        inline for (0..32) |lane| {
            if (any_mask[lane]) count += 1;
        }
    }
    // Scalar tail
    while (i < src.len) : (i += 1) {
        const c = src[i];
        inline for ("{}[],:\"\\") |sc| {
            if (c == sc) {
                count += 1;
                break;
            }
        }
    }
    return count;
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const alloc = arena.allocator();

    // Build a synthetic JSON-ish header with structural chars
    // distributed at realistic density (~10% of bytes are structural).
    var prng: std.Random.DefaultPrng = .init(SEED);
    const rng = prng.random();
    const src = try alloc.alloc(u8, FIXTURE_HEADER_BYTES);
    for (src) |*b| {
        const r = rng.intRangeAtMost(u8, 0, 99);
        b.* = if (r < 10) STRUCT_CHARS[r % STRUCT_CHARS.len] else 'a';
    }

    // Warmup
    _ = scanIndexOfAny(src);
    _ = scanVector32(src);

    // Time indexOfAny
    const t0 = std.Io.Clock.awake.now(io);
    var checksum_io: usize = 0;
    for (0..ITERS) |_| checksum_io +%= scanIndexOfAny(src);
    const ns_io: u64 = @intCast(t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

    // Time @Vector(32, u8)
    const t1 = std.Io.Clock.awake.now(io);
    var checksum_v: usize = 0;
    for (0..ITERS) |_| checksum_v +%= scanVector32(src);
    const ns_v: u64 = @intCast(t1.durationTo(std.Io.Clock.awake.now(io)).nanoseconds);

    std.debug.print("=== structural-scan throughput: indexOfAny vs @Vector(32, u8) ===\n", .{});
    std.debug.print("src bytes  = {d}\n", .{src.len});
    std.debug.print("iters      = {d}\n", .{ITERS});
    std.debug.print("indexOfAny\n", .{});
    std.debug.print("  ns_per_iter = {d}\n", .{ns_io / ITERS});
    std.debug.print("  ns_per_byte = {d}\n", .{ns_io / ITERS / src.len});
    std.debug.print("  matches     = {d}\n", .{checksum_io / ITERS});
    std.debug.print("vector32\n", .{});
    std.debug.print("  ns_per_iter = {d}\n", .{ns_v / ITERS});
    std.debug.print("  ns_per_byte = {d}\n", .{ns_v / ITERS / src.len});
    std.debug.print("  matches     = {d}\n", .{checksum_v / ITERS});
    std.debug.print("speedup_vector32_over_indexofany = {d:.3}x\n", .{@as(f64, @floatFromInt(ns_io)) / @as(f64, @floatFromInt(ns_v))});
    std.debug.print("matches_equal = {}\n", .{checksum_io == checksum_v});
}
