#!/usr/bin/env python3
"""
Generate a realistic Llama-shaped safetensors fixture for v0.1
real-model integration testing.

Writes the file directly via the safetensors format spec:
  bytes [0,8): little-endian u64 = header length
  bytes [8, 8+N): UTF-8 JSON header
  bytes [8+N, EOF): contiguous tensor data

No external dependencies (no numpy, no torch, no safetensors package).
Faithful to the format spec.

Output: tests/fixtures/tinyllama-shape.safetensors (~40MB)
"""

import argparse
import json
import os
import secrets
import struct
import sys


# Llama-style architectural skeleton, downscaled for fixture size.
# Real Llama-3.2-1B: vocab=128256, hidden=2048, layers=16, heads=32, head_dim=64
# Fixture:           vocab=32000,  hidden=128,  layers=4,  heads=4,  head_dim=32
VOCAB = 32000
HIDDEN = 128
LAYERS = 4
HEADS = 4
HEAD_DIM = 32
INTERMEDIATE = 256
HEAD_DIM_TOTAL = HEADS * HEAD_DIM  # 128


def f32_bytes(n: int) -> bytes:
    """Generate n*4 random F32-shaped bytes. Uses os.urandom for
    determinism-less speed (this is a fixture, not crypto)."""
    return secrets.token_bytes(n * 4)


def add_tensor(tensors, offset, name, shape):
    """Append a tensor entry. Returns updated offset."""
    numel = 1
    for d in shape:
        numel *= d
    nbytes = numel * 4  # F32
    tensors[name] = {
        "dtype": "F32",
        "shape": list(shape),
        "data_offsets": [offset, offset + nbytes],
    }
    return offset + nbytes


def build_header():
    """Build the JSON header for a Llama-shaped model. Returns
    (header_dict, total_data_bytes)."""
    tensors = {}
    offset = 0

    # Embedding
    offset = add_tensor(tensors, offset, "model.embed_tokens.weight", (VOCAB, HIDDEN))

    # Per-layer tensors
    for layer in range(LAYERS):
        p = f"model.layers.{layer}"
        # Attention
        offset = add_tensor(tensors, offset, f"{p}.self_attn.q_proj.weight", (HEAD_DIM_TOTAL, HIDDEN))
        offset = add_tensor(tensors, offset, f"{p}.self_attn.k_proj.weight", (HEAD_DIM_TOTAL, HIDDEN))
        offset = add_tensor(tensors, offset, f"{p}.self_attn.v_proj.weight", (HEAD_DIM_TOTAL, HIDDEN))
        offset = add_tensor(tensors, offset, f"{p}.self_attn.o_proj.weight", (HIDDEN, HEAD_DIM_TOTAL))

        # MLP (Llama-style gate / up / down)
        offset = add_tensor(tensors, offset, f"{p}.mlp.gate_proj.weight", (INTERMEDIATE, HIDDEN))
        offset = add_tensor(tensors, offset, f"{p}.mlp.up_proj.weight", (INTERMEDIATE, HIDDEN))
        offset = add_tensor(tensors, offset, f"{p}.mlp.down_proj.weight", (HIDDEN, INTERMEDIATE))

        # Norms
        offset = add_tensor(tensors, offset, f"{p}.input_layernorm.weight", (HIDDEN,))
        offset = add_tensor(tensors, offset, f"{p}.post_attention_layernorm.weight", (HIDDEN,))

    # Final norm + LM head
    offset = add_tensor(tensors, offset, "model.norm.weight", (HIDDEN,))
    offset = add_tensor(tensors, offset, "lm_head.weight", (VOCAB, HIDDEN))

    # Metadata
    tensors["__metadata__"] = {
        "framework": "fixture",
        "format": "pt",
        "model_type": "llama-shape",
        "vocab_size": str(VOCAB),
        "hidden_size": str(HIDDEN),
        "num_hidden_layers": str(LAYERS),
        "num_attention_heads": str(HEADS),
    }

    return tensors, offset


def write_fixture(path: str, fill_zero: bool = False):
    header, total_data = build_header()
    header_json = json.dumps(header).encode("utf-8")
    n_tensors = sum(1 for k in header if k != "__metadata__")
    print(f"tensors:     {n_tensors}", file=sys.stderr)
    print(f"header:      {len(header_json):>10d} bytes", file=sys.stderr)
    print(f"data:        {total_data:>10d} bytes ({total_data / 1024 / 1024:.1f} MB)", file=sys.stderr)
    print(f"total:       {8 + len(header_json) + total_data:>10d} bytes", file=sys.stderr)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        if fill_zero:
            # Faster fixture build; tensor data is meaningless either way.
            f.write(b"\x00" * total_data)
        else:
            # Real random bytes — exercises the parser's robustness to non-zero data.
            # Write in chunks to avoid OOM at large sizes.
            CHUNK = 1024 * 1024
            written = 0
            while written < total_data:
                n = min(CHUNK, total_data - written)
                f.write(secrets.token_bytes(n))
                written += n
    print(f"wrote:       {path}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="tests/fixtures/tinyllama-shape.safetensors")
    p.add_argument("--zero", action="store_true",
                   help="Fill data segment with zeros (faster); default is random bytes")
    args = p.parse_args()
    write_fixture(args.out, fill_zero=args.zero)


if __name__ == "__main__":
    main()
