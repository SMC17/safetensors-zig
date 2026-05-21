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
    b.installArtifact(inspect_exe);
    const run_inspect = b.addRunArtifact(inspect_exe);
    if (b.args) |args| run_inspect.addArgs(args);
    const inspect_step = b.step("inspect", "Inspect a .safetensors file");
    inspect_step.dependOn(&run_inspect.step);

    // Benchmark — header parse throughput. Always ReleaseFast.
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
    const bench_step = b.step("bench", "Run parse throughput benchmark");
    bench_step.dependOn(&run_bench.step);
}
