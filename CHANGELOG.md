# Changelog

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
