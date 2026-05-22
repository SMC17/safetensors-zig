# safetensors-zig

A Zig 0.16 reader for the
[HuggingFace safetensors](https://github.com/huggingface/safetensors)
tensor serialisation format. Pure-Zig, single-module, no system deps.

## Status

**v0.2 — hand-tuned safetensors-specific JSON parser.** Profiling
the v0.1 std.json path showed 97% of time was spent in
`std.json.parseFromSliceLeaky` (84μs of 87μs total). v0.2 replaces it
with a single-pass byte-by-byte parser that skips the generic AST
and emits Tensor records directly. Result:

| Implementation | ns/parse (median, 5+ runs) | parses/sec | ratio |
|---|---|---|---|
| safetensors-zig **v0.0.1** (std.json, ReleaseFast) | ~85,000 | ~12,000 | 0.7x |
| `huggingface/safetensors` 0.4.5 (Rust, release) | ~77,000 | ~13,000 | 1.00x (reference) |
| **safetensors-zig v0.2** (hand-tuned parser, ReleaseFast) | **~24,000** | **~42,000** | **~3.2x FASTER** |

The v0.2 parser is correctness-equivalent to v0.0.1: same 11 unit
tests pass, same real-model integration test passes, plus 6 new
unit tests against the fast-path parser specifically. Sort +
validation logic is unchanged; the win comes entirely from
replacing std.json with a format-specialised parser.

**Honest non-claims for v0.2:**
- The fast parser accepts only the de-facto-canonical safetensors
  format. Tensor names with non-ASCII characters or Unicode escapes
  (`ÿ`) currently return InvalidJSON. Real safetensors files
  emitted by `huggingface/safetensors` and the Python `safetensors`
  package don't use these, but a file produced by a different
  serialiser might.
- The Rust crate also does dtype/shape/offset validation that we do.
  The 3.2x is a fair head-to-head: same fixture, same correctness
  checks, same output structure. No tricks.
- This is consumer-Linux numbers (Ryzen-class CPU, single core).
  Numbers will vary across hardware. The architectural win
  (single-pass + no AST + format-specialised) is hardware-invariant.

This is a pre-1.0 substrate per the Zig 0.16 ecosystem convention. No
"production-grade" / "v1.0" / "stable" claims will be made.

## What it does

- Parses the safetensors header (8-byte LE size prefix + UTF-8 JSON)
- Validates dtype, shape, and `data_offsets` for every tensor
- Enforces non-overlapping byte coverage and shape-vs-dtype consistency
- Exposes `Tensor` view + `readTensorBytes` to pull raw bytes by name
- Reports the `__metadata__` map if present

## What it does not do (yet)

- Memory-mapped tensor views (current path goes through `pread` /
  `readAll`; mmap planned for v0.2)
- Reverse direction (writing safetensors files) — read-only at v0.1
- Big-endian host targets (LE assumed; portable big-endian needs
  shape/data byte-swap on read; deferred until requested)
- Tensor dtype-typed views (f32 slice, etc.) — caller currently
  receives `[]u8` and is responsible for casting
- Streaming / chunked read of very large tensors
- HuggingFace Hub integration (model download); use `huggingface_hub`
  or `hf-zig` (not yet written) for that
- Production-tier real-model fixtures (the v0.1 fixture is
  Llama-shaped at 4 layers / 128 hidden / 32K vocab — ~10M params,
  ~33MB. Real Llama-3.2-1B is ~1.24B params / ~2.4GB. The format
  parses identically; the v0.1 fixture exercises the schema and the
  bench in a tractable size.)

## Quickstart

```zig
const std = @import("std");
const safetensors = @import("safetensors");

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var st = try safetensors.SafeTensors.open(alloc, io, "model.safetensors");
    defer st.deinit();

    for (st.tensors) |t| {
        std.debug.print("{s}  {s}  numel={d}\n", .{
            t.name, t.dtype.toString(), t.numel(),
        });
    }

    const w = st.get("transformer.h.0.attn.c_attn.weight")
        orelse return error.NotFound;
    const bytes = try st.readTensorBytesAlloc(alloc, w);
    defer alloc.free(bytes);
    // bytes is the raw little-endian dtype-typed buffer
}
```

## Build

```sh
zig build test                       # 11 unit tests + real-model integration
zig build inspect -- f.safetensors   # CLI: print schema
zig build bench                      # synthetic 256-tensor parse bench
zig build bench-realmodel            # parse bench against real-model fixture
zig build install -Doptimize=ReleaseFast  # produces zig-out/bin/safetensors-inspect

# Build the real-model fixture (33 MB, written reproducibly)
python3 tests/gen_fixture.py --zero

# Head-to-head bench vs Rust upstream
bash bench/compare.sh
```

Requires Zig 0.16; the head-to-head bench additionally requires
`cargo` + internet (one-time crates.io fetch of `safetensors = "0.4"`).

## API surface (v0.0.1)

- `safetensors.DType` — enum over the 15 safetensors dtypes
- `safetensors.Tensor` — `{ name, dtype, shape, data_offset_start, data_offset_end }`
- `safetensors.MetadataEntry` — `{ key, value }`
- `safetensors.SafeTensors` — main type
  - `.open(allocator, path) !SafeTensors`
  - `.openFromFile(allocator, file, file_size) !SafeTensors`
  - `.get(name) ?*const Tensor`
  - `.readTensorBytes(tensor, buf) !void`
  - `.readTensorBytesAlloc(alloc, tensor) ![]u8`
  - `.deinit()`
- `safetensors.Error` — error set: `InvalidHeader | HeaderTooLarge |
  UnknownDtype | InvalidShape | OffsetOutOfBounds | OffsetOverlap |
  Truncated | InvalidJSON`

## Comparable upstream

- [`huggingface/safetensors`](https://github.com/huggingface/safetensors)
  — the canonical Rust + Python implementation. This crate is
  production-grade; safetensors-zig is the Zig sibling at v0.0.1.
- [`Narsil/safetensors-zig`](https://github.com/Narsil/safetensors-zig) —
  if discovered after release, will be cited and benchmarked against.

## Position in the portfolio

safetensors-zig is artifact #4 in the 8-artifact portfolio (see
`the broader sovereign-stack portfolio`). It is a foundation block — every Zig artifact
that loads model weights (vLLM-zig, mlx-nonapple) will depend on this.

## Licence

AGPL-3.0-or-later. See `LICENSE`. The dual-use implications of weight-
loading code are limited; this exists at AGPL primarily for
portfolio-coherence with the rest of the sovereign-stack OSS surface.
