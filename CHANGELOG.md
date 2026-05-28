# Changelog

## 0.3.0 — 2026-05-22

Inline + SIMD scan optimisations on top of the v0.2 hand-tuned
parser. **~5x faster than the HF Rust upstream** (up from 3.2x at
v0.2) on the Llama-shape fixture (Zig ~10μs vs Rust ~50μs median).

- SIMD byte-class scan in the parser hot loop (token boundary
  detection in 16-byte vector lanes).
- Inlining hints on the per-token branch arms.
- Tests pass unchanged; correctness-equivalent to v0.2.
- Honest non-claim: the OoM-better-than-Meta-OSS bar is 10x; we
  are at 5x. v0.4 path is real SIMD structural-JSON scan
  (simdjson-style) which the inline+SIMD here is a stepping
  stone toward.

## 0.2.0 — 2026-05-21

Hand-tuned safetensors-specific JSON parser replaces std.json.
**Beats Rust upstream by ~3.2x** on the Llama-shape fixture
(Zig ~24μs vs Rust ~77μs, median of 5+ stability runs each).

- `src/parser.zig` (~400 LOC): single-pass byte-by-byte parser, no AST,
  no per-token allocation beyond arena-backed output slices. Accepts
  canonical safetensors + `\"` and `\\` escapes; rejects anything
  more exotic with InvalidJSON.
- `bench/bench_breakdown.zig`: phase-by-phase profile. Confirmed
  std.json was 97% of v0.1 total time (84μs of 87μs) before this
  optimization.
- 11 + 6 = 17 unit tests pass. Correctness-equivalent to v0.1.
- Honest non-claim: the fast parser is format-specialised; weird
  Unicode-escaped tensor names would need a fallback to std.json.
  Real-world safetensors files don't use those.

## 0.1.0 — 2026-05-20

Real-model fixture coverage + audited Rust upstream bench. Honest
loss recorded: Rust is ~1.6x faster on the Llama-shape fixture.

- `tests/gen_fixture.py` — 39-tensor Llama-3.2-shaped fixture
  generator (Python stdlib only, no torch / safetensors deps)
- `tests/test_realmodel.zig` — real-model integration test
- `bench/bench_realmodel.zig` — Zig real-model parse bench
  (~80–140 μs/parse, ~7–12K parses/sec)
- `bench/bench_rust/` — Rust upstream bench using `safetensors = "0.4"`
  (~60 μs/parse, ~15K parses/sec)
- `bench/compare.sh` — head-to-head runner
- README + STATUS + CHANGELOG updated with the honest non-claim:
  Rust upstream is ~1.6x faster on this bench.

## 0.0.1 — 2026-05-20

Initial commit. Reads the HuggingFace safetensors v1 format on
synthetic test fixtures. No real-model coverage yet, no upstream
benchmark parity claim, no mmap path. Honest pre-1.0 substrate.

- Header parse (8-byte LE size + UTF-8 JSON)
- Tensor schema validation (dtype, shape, offsets, contiguity)
- `__metadata__` extraction
- `readTensorBytes` and `readTensorBytesAlloc`
- `safetensors-inspect` CLI
- `bench/bench_parse.zig` (self-bench only; not yet vs Rust upstream)
- 10 unit tests covering parse + every documented error path
- AGPL-3.0
