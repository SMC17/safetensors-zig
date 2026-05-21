# STATUS

**Version:** v0.1
**Last update:** 2026-05-20

## What landed in v0.1 (on top of v0.0.1)

- `tests/gen_fixture.py` — reproducible Llama-3.2-shaped safetensors
  fixture generator (39 tensors, ~33 MB, no Python deps beyond stdlib)
- `tests/fixtures/tinyllama-shape.safetensors` — gitignored binary,
  reproducibly generated
- `tests/test_realmodel.zig` — real-model integration test that
  validates 39 tensor names, 7 metadata keys, embed_tokens shape,
  and exercises the data-read path on a 16 MB tensor
- `bench/bench_realmodel.zig` — Zig parse bench against the
  real-model fixture (~80–140 μs/parse, ~7–12K parses/sec on
  ReleaseFast, consumer Linux)
- `bench/bench_rust/` — Rust upstream bench using
  `huggingface/safetensors = "0.4"` crate on the same fixture
  (~60 μs/parse, ~15K parses/sec, release mode)
- `bench/compare.sh` — runs both benches and reports the ratio
- README updated with the honest v0.1 head-to-head: **Rust is
  ~1.6x faster.** No parity claim made.

## What landed in v0.0.1 (kept)

- `src/root.zig` — public API (`SafeTensors`, `Parsed`, `openFromBytes`,
  `open`, `Tensor`, `DType`)
- 11 unit tests covering spec + every error path
- `examples/inspect.zig` — `safetensors-inspect <path>` CLI
- `bench/bench_parse.zig` — synthetic 256-tensor self-bench
  (~570 μs/parse, ~450K tensors/sec)
- AGPL-3.0 LICENSE
- Zig 0.16 Io-abstraction-compatible

## Honest non-claims

- **Rust upstream is faster.** ~1.6x on the Llama-shape fixture. The
  Zig path is competitive within single-digit factors but does not
  match Rust. We publish the loss rather than hide it.
- **The v0.1 fixture is Llama-shaped, not Llama-sized.** Real Llama-
  3.2-1B is ~2.4 GB; the fixture is 33 MB. The format parses
  identically and the bench is fair, but the fixture is not a full
  production-tier model. v0.2 will exercise a full-sized real model.
- **No mmap path.** Current `readPositionalAll` reads into a
  caller-provided buffer. mmap-backed zero-copy is v0.2.
- **No writer.** Read-only.
- **No big-endian host support.** LE assumed.
- **Single-platform tested.** x86_64-linux only. macOS / Windows
  / ARM untested at v0.1.

## What is open for v0.2

1. mmap-backed tensor views for zero-copy reads
2. Full-sized real-model fixture (Llama-3.2-1B at 2.4 GB, parsed
   in CI off a downloaded HF model)
3. Tensor dtype-typed views (f32 / bf16 / f16 typed slices)
4. Writer (safetensors output)
5. Fuzz tests (malformed JSON / offsets / shapes)
6. Cross-platform CI (macOS / Windows)
7. Performance-gap profiling — where is the 1.6x going? Likely
   candidates: std.json overhead vs Rust's serde_json hand-tuned
   parser; allocation churn in the arena; iteration vs SIMD path.
8. Promote to v1.0 only after these all land + a published bench
   shows parity within 1.1x of the Rust upstream.
