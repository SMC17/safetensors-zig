const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Library module.
    const mod = b.addModule("safetensors", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Unit tests against src/root.zig.
    const tests = b.addTest(.{
        .root_module = mod,
    });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);

    // Real-model integration test — runs against
    // tests/fixtures/tinyllama-shape.safetensors if present (built by
    // tests/gen_fixture.py). Skipped if absent. Wired as a separate
    // module so it imports the public library as a consumer would.
    const realmodel_mod = b.createModule(.{
        .root_source_file = b.path("tests/test_realmodel.zig"),
        .target = target,
        .optimize = optimize,
    });
    realmodel_mod.addImport("safetensors", mod);
    const realmodel_tests = b.addTest(.{
        .root_module = realmodel_mod,
    });
    const run_realmodel_tests = b.addRunArtifact(realmodel_tests);
    test_step.dependOn(&run_realmodel_tests.step);
    const realmodel_step = b.step("test-realmodel", "Run real-model integration test only");
    realmodel_step.dependOn(&run_realmodel_tests.step);

    // Real-world TinyLlama-1.1B integration test (2.2GB fixture).
    const real_tl_mod = b.createModule(.{
        .root_source_file = b.path("tests/test_realtinyllama.zig"),
        .target = target,
        .optimize = optimize,
    });
    real_tl_mod.addImport("safetensors", mod);
    const real_tl_tests = b.addTest(.{
        .root_module = real_tl_mod,
    });
    const run_real_tl_tests = b.addRunArtifact(real_tl_tests);
    test_step.dependOn(&run_real_tl_tests.step);
    const real_tl_step = b.step("test-real-tinyllama", "Real TinyLlama-1.1B 2.2GB parse test");
    real_tl_step.dependOn(&run_real_tl_tests.step);

    // Inspect CLI: `safetensors-inspect <file.safetensors>`.
    const inspect_mod = b.createModule(.{
        .root_source_file = b.path("examples/inspect.zig"),
        .target = target,
        .optimize = optimize,
    });
    inspect_mod.addImport("safetensors", mod);
    const inspect_exe = b.addExecutable(.{
        .name = "safetensors-inspect",
        .root_module = inspect_mod,
    });
    
    if (optimize != .Debug) {
        inspect_exe.root_module.strip = true;
    }

    b.installArtifact(inspect_exe);
    const run_inspect = b.addRunArtifact(inspect_exe);
    if (b.args) |args| run_inspect.addArgs(args);
    const inspect_step = b.step("inspect", "Inspect a .safetensors file");
    inspect_step.dependOn(&run_inspect.step);

    // Benchmark — synthetic header parse throughput. Always ReleaseFast.
    const bench_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_parse.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_mod.addImport("safetensors", mod);
    const bench_exe = b.addExecutable(.{
        .name = "bench-parse",
        .root_module = bench_mod,
    });
    const run_bench = b.addRunArtifact(bench_exe);
    run_bench.has_side_effects = true;
    const bench_step = b.step("bench", "Run synthetic parse throughput benchmark");
    bench_step.dependOn(&run_bench.step);

    // Real-model fixture bench (Zig side of Zig-vs-Rust comparison).
    const bench_rm_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_realmodel.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_rm_mod.addImport("safetensors", mod);
    const bench_rm_exe = b.addExecutable(.{
        .name = "bench-realmodel",
        .root_module = bench_rm_mod,
    });
    const run_bench_rm = b.addRunArtifact(bench_rm_exe);
    run_bench_rm.has_side_effects = true;
    const bench_rm_step = b.step("bench-realmodel", "Bench parse against real-model fixture");
    bench_rm_step.dependOn(&run_bench_rm.step);

    // Profile-style breakdown of parse phases.
    const bench_bd_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_breakdown.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_bd_mod.addImport("safetensors", mod);
    const bench_bd_exe = b.addExecutable(.{
        .name = "bench-breakdown",
        .root_module = bench_bd_mod,
    });
    const run_bench_bd = b.addRunArtifact(bench_bd_exe);
    run_bench_bd.has_side_effects = true;
    const bench_bd_step = b.step("bench-breakdown", "Profile-style breakdown of parse phases");
    bench_bd_step.dependOn(&run_bench_bd.step);

    // Real TinyLlama-1.1B 2.2GB parse bench — head-to-head with HF
    // safetensors warm-cache 1.0 ms reference measured 2026-05-28.
    const bench_real_tl_mod = b.createModule(.{
        .root_source_file = b.path("bench/bench_real_tinyllama.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_real_tl_mod.addImport("safetensors", mod);
    const bench_real_tl_exe = b.addExecutable(.{
        .name = "bench-real-tinyllama",
        .root_module = bench_real_tl_mod,
    });
    const run_bench_real_tl = b.addRunArtifact(bench_real_tl_exe);
    run_bench_real_tl.has_side_effects = true;
    const bench_real_tl_step = b.step("bench-real-tinyllama", "Bench parse against real TinyLlama-1.1B 2.2GB fixture");
    bench_real_tl_step.dependOn(&run_bench_real_tl.step);
}
