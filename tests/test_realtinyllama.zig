//! Real-world TinyLlama-1.1B integration test.
//!
//! Loads tests/fixtures/tinyllama-real/model.safetensors (2.2 GB BF16
//! TinyLlama-1.1B-Chat-v1.0 downloaded from HuggingFace) and validates
//! the parser against the full real-world model:
//!   - 201 tensors
//!   - 22 transformer layers
//!   - BF16 dtype throughout
//!   - Grouped-query attention layout (Q=2048, KV=256 — 32 Q heads / 4 KV heads)
//!
//! Skipped if the fixture is absent. Build with:
//!   pip install huggingface_hub
//!   python3 -c "from huggingface_hub import hf_hub_download; \
//!     hf_hub_download(repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0', \
//!     filename='model.safetensors', \
//!     local_dir='tests/fixtures/tinyllama-real')"

const std = @import("std");
const safetensors = @import("safetensors");

const fixture_path = "tests/fixtures/tinyllama-real/model.safetensors";

test "real TinyLlama-1.1B — full 201-tensor BF16 parse" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;

    var f = std.Io.Dir.cwd().openFile(io, fixture_path, .{ .mode = .read_only }) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print(
                "\n[skip] {s} not present. Download with hf_hub_download (see file header).\n",
                .{fixture_path},
            );
            return error.SkipZigTest;
        },
        else => return err,
    };
    f.close(io);

    var st = try safetensors.SafeTensors.open(alloc, io, fixture_path);
    defer st.deinit();

    // TinyLlama-1.1B-Chat-v1.0 reproducibly has 201 tensors:
    //   2  for embed + lm_head
    //   22 × 9 = 198 per-layer tensors (q/k/v/o + gate/up/down + 2 norms)
    //   1  final norm
    try std.testing.expectEqual(@as(usize, 201), st.tensors.len);

    // All tensors should be BF16
    for (st.tensors) |t| {
        try std.testing.expectEqual(safetensors.DType.bf16, t.dtype);
    }

    // Specific tensors we expect by name and shape
    const embed = st.get("model.embed_tokens.weight") orelse return error.TestExpectedTensor;
    try std.testing.expectEqual(@as(u64, 32000), embed.shape[0]);
    try std.testing.expectEqual(@as(u64, 2048), embed.shape[1]);

    const lm_head = st.get("lm_head.weight") orelse return error.TestExpectedTensor;
    try std.testing.expectEqual(@as(u64, 32000), lm_head.shape[0]);
    try std.testing.expectEqual(@as(u64, 2048), lm_head.shape[1]);

    const final_norm = st.get("model.norm.weight") orelse return error.TestExpectedTensor;
    try std.testing.expectEqual(@as(usize, 1), final_norm.shape.len);
    try std.testing.expectEqual(@as(u64, 2048), final_norm.shape[0]);

    // Layer 0 attention shapes — TinyLlama uses GQA with 32 Q heads,
    // 4 KV heads, head_dim 64. Q is [2048, 2048]; K/V are [256, 2048].
    const q0 = st.get("model.layers.0.self_attn.q_proj.weight") orelse return error.TestExpectedTensor;
    try std.testing.expectEqual(@as(u64, 2048), q0.shape[0]);
    try std.testing.expectEqual(@as(u64, 2048), q0.shape[1]);
    const k0 = st.get("model.layers.0.self_attn.k_proj.weight") orelse return error.TestExpectedTensor;
    try std.testing.expectEqual(@as(u64, 256), k0.shape[0]);
    try std.testing.expectEqual(@as(u64, 2048), k0.shape[1]);

    // Verify all 22 layers are present
    var layer_count: usize = 0;
    var i: usize = 0;
    while (i < 22) : (i += 1) {
        var buf: [128]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, "model.layers.{d}.input_layernorm.weight", .{i});
        if (st.get(name) != null) layer_count += 1;
    }
    try std.testing.expectEqual(@as(usize, 22), layer_count);

    // Exercise the data-read path on a small portion of embed
    var head: [256]u8 = undefined;
    const start_offset = st.data_start + embed.data_offset_start;
    const n = try st.file.readPositionalAll(st.io, &head, start_offset);
    try std.testing.expectEqual(@as(usize, 256), n);
    // We don't assert specific values (BF16 random-from-training);
    // just that the read returned the requested length.
}
