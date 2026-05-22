# STATUS

**Version:** v0.3
**Last update:** 2026-05-22

## What landed in v0.3 (on top of v0.2)

- Inline+SIMD optimisations layered on the v0.2 hand-tuned parser:
  - `std.ArrayList.ensureTotalCapacity` from header-size estimate
  - First-letter dispatch for {dtype, shape, data_offsets} keys
    (eliminates `std.mem.eql` in the inner loop)
  - `@branchHint(.likely)` on the comma-separator branch
  - SIMD-accelerated string scan via `std.mem.indexOfAny` for the
    `"` and `\` characters; happy path returns slice in one call
  - Sort-skip fast-path: scan for already-sorted; HF emits in
    offset-increasing order so the sort is typically skipped
- **Median ~10μs/parse on the Llama-shape fixture** (~6.5μs cache-warm)
- **~5x faster than the HF Rust upstream** (up from 3.2x at v0.2)
- 17 unit tests + 3 real-model integration tests pass; output
  byte-identical to the Python `safetensors` package

## Speedup progression vs HF Rust upstream

| Version | ns/parse (median) | Ratio vs Rust |
|---|---|---|
| v0.0.1 (std.json) | ~85,000 | 0.7x (slower) |
| v0.2 (hand-tuned) | ~24,000 | 3.2x faster |
| **v0.3 (this)** | **~10,000** | **5.0x faster** |
| Rust 0.4.5 | ~50,000 | 1.0x |

## What landed in v0.2 (on top of v0.1)

- `src/parser.zig` — hand-tuned safetensors-specific JSON parser
  (~400 LOC). Single-pass byte-by-byte, no AST, no per-token
  allocation beyond arena-backed output slices.
- `bench/bench_breakdown.zig` — profile-style breakdown that
  confirmed std.json was 97% of v0.1 parse time

## What landed in v0.1 (on top of v0.0.1)

- `tests/gen_fixture.py` — reproducible Llama-3.2-shaped fixture
  generator (39 tensors, ~33 MB, no Python deps)
- `tests/test_realmodel.zig` — real-model integration test
- `bench/bench_realmodel.zig` + `bench/bench_rust/` + `bench/compare.sh`
  — head-to-head infrastructure against HF Rust crate

## What landed in v0.0.1 (kept)

- `src/root.zig` public API (`SafeTensors`, `Parsed`, `openFromBytes`,
  `open`, `Tensor`, `DType`)
- 11 unit tests covering spec + every error path
- `examples/inspect.zig` — `safetensors-inspect <path>` CLI
- AGPL-3.0 LICENSE; Zig 0.16 Io-abstraction-compatible

## Honest non-claims (v0.3-current)

- **Beats Rust by 5x, not 10x.** The OoM-better-than-Meta-OSS bar
  is 10x; v0.3 is at 5x. v0.4 path: real SIMD JSON structural scan
  (simdjson-style), two-pass parse (count then alloc), assume
  canonical HF key order for fast path.
- **v0.3 fixture is Llama-shape, not Llama-sized.** Synthetic 33MB
  fixture for the bench; real TinyLlama-1.1B (2.2GB) tested
  separately in `tests/test_realtinyllama.zig`.
- **No mmap path yet.** Current `readPositionalAll` reads into a
  caller-provided buffer. mmap-backed zero-copy is v0.4.
- **No writer.** Read-only at v0.3.
- **No big-endian host support.** LE assumed.
- **Single-platform tested.** x86_64-linux only.

## What is open for v0.4

1. SIMD JSON structural scan (simdjson-style; 1-2 weeks of careful
   SWAR + AVX work to push from 5x toward 10x vs Rust)
2. Two-pass parse: count tensors first, allocate once, then fill
3. Canonical-order fast path: assume HF emits keys in
   {dtype, shape, data_offsets} order, skip key matching entirely
4. mmap-backed tensor views for zero-copy reads
5. Tensor dtype-typed views (f32 / bf16 / f16 typed slices)
6. Writer (safetensors output)
7. Fuzz tests (malformed JSON / offsets / shapes)
8. Cross-platform CI (macOS / Windows)
9. Promote to v1.0 only after all the above land + 10x bench parity
