# safetensors-zig

A Zig 0.16 reader for the
[HuggingFace safetensors](https://github.com/huggingface/safetensors)
tensor serialisation format. Pure-Zig, single-module, no system deps,
~600 lines including tests.

## Status

**v0.0.1 — works on synthetic test fixtures.** Real-world model
fixtures (Llama, Qwen, BERT, etc.) are not yet in the test matrix.
Perf parity claims against
[`huggingface/safetensors`](https://github.com/huggingface/safetensors)
(Rust) are not made until cross-implementation bench numbers land.

This is a pre-1.0 substrate. Per the Zig 0.16 ecosystem convention,
no "production-grade" / "v1.0" / "stable" claims will be made until
the implementation is exercised against the canonical model zoo and
benchmarked against the upstream Rust crate with audited numbers.

## What it does

- Parses the safetensors header (8-byte LE size prefix + UTF-8 JSON)
- Validates dtype, shape, and `data_offsets` for every tensor
- Enforces non-overlapping byte coverage and shape-vs-dtype consistency
- Exposes `Tensor` view + `readTensorBytes` to pull raw bytes by name
- Reports the `__metadata__` map if present

## What it does not do (yet)

- Memory-mapped tensor views (current path goes through `pread` /
  `readAll`; mmap planned for v0.1)
- Reverse direction (writing safetensors files) — read-only at v0.0.1
- Cross-implementation byte-exact validation against the Rust upstream
- Big-endian host targets (LE assumed; portable big-endian needs
  shape/data byte-swap on read; deferred until requested)
- Tensor dtype-typed views (f32 slice, etc.) — caller currently
  receives `[]u8` and is responsible for casting
- Streaming / chunked read of very large tensors
- HuggingFace Hub integration (model download); use `huggingface_hub`
  or `hf-zig` (not yet written) for that

## Quickstart

```zig
const std = @import("std");
const safetensors = @import("safetensors");

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    var st = try safetensors.SafeTensors.open(alloc, "model.safetensors");
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
zig build test              # unit tests
zig build inspect -- f.safetensors  # CLI: print schema
zig build bench             # header-parse throughput
zig build install -Doptimize=ReleaseFast  # produces zig-out/bin/safetensors-inspect
```

Requires Zig 0.16.

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
`~/portfolio/README.md`). It is a foundation block — every Zig artifact
that loads model weights (vLLM-zig, mlx-nonapple) will depend on this.

## Licence

AGPL-3.0-or-later. See `LICENSE`. The dual-use implications of weight-
loading code are limited; this exists at AGPL primarily for
portfolio-coherence with the rest of the sovereign-stack OSS surface.
