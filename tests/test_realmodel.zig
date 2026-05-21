//! Real-model-shape integration test.
//!
//! Loads tests/fixtures/tinyllama-shape.safetensors (built by
//! tests/gen_fixture.py) and validates the parser against a 39-tensor
//! Llama-shaped fixture (~33 MB random-bytes data segment).
//!
//! If the fixture is missing, the test is skipped with a clear hint
//! about how to build it. We intentionally do NOT check the binary
//! into git — it is reproducibly generated.
//!
//! This is the v0.1 closing-test against the v0.0.1 "no real-model
//! coverage" non-claim.

const std = @import("std");
const safetensors = @import("safetensors");

const fixture_path = "tests/fixtures/tinyllama-shape.safetensors";

const expected = .{
    .tensor_count = 39,
    .metadata_count = 7,
    .layers = 4,
};

const expected_tensor_names = [_][]const u8{
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.post_attention_layernorm.weight",
    "model.norm.weight",
    "lm_head.weight",
};

test "realmodel fixture — Llama-shape 39-tensor parse" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;

    // If fixture is missing, skip (don't fail). Surface a hint.
    var f = std.Io.Dir.cwd().openFile(io, fixture_path, .{ .mode = .read_only }) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print(
                "\n[skip] {s} not present. Build it with:\n  python3 tests/gen_fixture.py --zero\n",
                .{fixture_path},
            );
            return error.SkipZigTest;
        },
        else => return err,
    };
    f.close(io);

    var st = try safetensors.SafeTensors.open(alloc, io, fixture_path);
    defer st.deinit();

    try std.testing.expectEqual(@as(usize, expected.tensor_count), st.tensors.len);
    try std.testing.expectEqual(@as(usize, expected.metadata_count), st.metadata.len);

    // Verify each expected tensor name is present.
    inline for (expected_tensor_names) |name| {
        const t = st.get(name) orelse {
            std.debug.print("\n[fail] expected tensor missing: {s}\n", .{name});
            return error.TestExpectedTensor;
        };
        try std.testing.expectEqual(safetensors.DType.f32, t.dtype);
    }

    // Verify metadata keys we know are present.
    var have_framework = false;
    var have_model_type = false;
    for (st.metadata) |m| {
        if (std.mem.eql(u8, m.key, "framework")) have_framework = true;
        if (std.mem.eql(u8, m.key, "model_type")) have_model_type = true;
    }
    try std.testing.expect(have_framework);
    try std.testing.expect(have_model_type);

    // Verify embed_tokens shape (load-bearing for any LLM loader)
    const embed = st.get("model.embed_tokens.weight").?;
    try std.testing.expectEqual(@as(usize, 2), embed.shape.len);
    try std.testing.expectEqual(@as(u64, 32000), embed.shape[0]);
    try std.testing.expectEqual(@as(u64, 128), embed.shape[1]);
    try std.testing.expectEqual(@as(u64, 32000 * 128 * 4), embed.byteLen());

    // Read first 16 bytes of embed_tokens — exercises the data-read path
    // on a real-size tensor (16 MB block).
    var head_buf: [16]u8 = undefined;
    const start_offset = st.data_start + embed.data_offset_start;
    const n = try st.file.readPositionalAll(st.io, &head_buf, start_offset);
    try std.testing.expectEqual(@as(usize, 16), n);
    // We don't assert specific bytes (random data); just that the read succeeded.
}
