# safetensors-zig

**`tier-1 reproducible substrate`** · `flake.nix` · `scripts/checkpoint.sh` · `.githooks/post-commit`

[![CI](https://github.com/SMC17/safetensors-zig/actions/workflows/ci.yml/badge.svg)](https://github.com/SMC17/safetensors-zig/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/SMC17/safetensors-zig?sort=semver)](https://github.com/SMC17/safetensors-zig/releases)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](LICENSE)
[![Zig](https://img.shields.io/badge/Zig-0.16-orange.svg)](https://ziglang.org/)

A Zig 0.16 reader for the
[HuggingFace safetensors](https://github.com/huggingface/safetensors)
tensor serialisation format. Pure-Zig, single-module, no system deps.

## Status

**Reader prototype.** The current small task and historical benchmark record
are described separately below. Historical speed ratios have not been rerun
on the current host and are not current performance claims.

## One original-input task

```sh
zig build test --summary all
zig build -Doptimize=ReleaseSafe
python3 scripts/verify_inspect_task.py
```

The task inspects a 128-byte original fixture with one two-element F32 tensor
and metadata, then requires six malformed/missing-input cases to fail cleanly.
No real model weights are downloaded. Current local unit baseline: 19 pass,
two optional model tests skipped. This does not revalidate those real models.

Create a private working source/executable/log pack with
`python3 tools/selfserve.py selfserve.json --repo . --out /tmp/safetensors-task`.
This is acceptance evidence, not a release cut. Green CI and operator intent
remain required before release.

## Historical benchmark record (not remeasured)

| Implementation | ns/parse (median, 5+ runs) | parses/sec | ratio |
|---|---|---|---|
| safetensors-zig v0.0.1 (std.json, ReleaseFast) | ~85,000 | ~12,000 | 0.7x (slower) |
| `huggingface/safetensors` 0.4.5 (Rust, release) | ~50,000 | ~20,000 | 1.00x (reference) |
| safetensors-zig v0.2 (hand-tuned parser) | ~24,000 | ~42,000 | ~3.2x faster |
| **safetensors-zig v0.3** (hand-tuned + inline + SIMD scan) | **~10,000** | **~100,000** | **~5.0x FASTER** |

The v0.3 parser is the v0.2 single-pass hand-tuned parser plus:
- `std.ArrayList.ensureTotalCapacity` from header-size estimate
- First-letter dispatch for `{dtype, shape, data_offsets}` keys
  (eliminates `std.mem.eql` in the inner loop)
- `@branchHint(.likely)` on the comma-separator branch
- SIMD string scan via `std.mem.indexOfAny` for the `"` and `\` chars
- Sort-skip fast-path (HF emits offset-increasing; sort is typically
  skipped)

The old correctness/parity claims require their own recorded inputs and rerun.
The current tiny task checks schema inspection and refusal behavior only.

**Current evidence boundaries:**
- The historical ratio column needs reconciliation with the displayed timings
  and a matched baseline/hardware rerun. No 5x claim is validated by this task.
- Unicode/escape and full upstream compatibility require their own parity
  fixtures. The original ASCII fixture does not establish those properties.
- Large-model behavior, throughput and architecture-wide speedups are separate
  measurements. No hardware-invariant speed claim is made here.

This is a pre-1.0 substrate per the Zig 0.16 ecosystem convention. Claims stay
below release-readiness language until the evidence changes.

## What it does

- Parses the safetensors header (8-byte LE size prefix + UTF-8 JSON)
- Validates dtype, shape, and `data_offsets` for every tensor
- Enforces non-overlapping byte coverage and shape-vs-dtype consistency
- Exposes `Tensor` view + `readTensorBytes` to pull raw bytes by name
- Reports the `__metadata__` map if present

## What it does not do (yet)

- Memory-mapped tensor views (current path goes through `pread` /
  `readAll`)
- Reverse direction (writing safetensors files) — read-only
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

## API surface (v0.3)

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
  — the canonical Rust + Python implementation, widely deployed upstream.
  The current Zig task checks a tiny original header and refusal behavior.
  A current speed or full-compatibility comparison requires a matched rerun.

## Licence

AGPL-3.0-or-later. See `LICENSE`.

## Reproducible setup

```
nix develop        # Zig 0.16.0 + jq + curl + gawk + python3(websockets)
zig build test     # 17/17 expected
```

The repo's `flake.nix` pins exact dep versions (see `STATUS.md`).
`scripts/checkpoint.sh` + `.githooks/post-commit` auto-regenerate
`NEXT.md` after every commit (opt-in: `git config core.hooksPath .githooks`).
