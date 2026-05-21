# Changelog

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
