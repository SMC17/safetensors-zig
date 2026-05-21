# STATUS

**Version:** v0.0.1
**Last update:** 2026-05-20

## What landed

- `build.zig` + `build.zig.zon` (Zig 0.16, no deps)
- `src/root.zig` — public API
- 10 unit tests in `src/root.zig`:
  - DType.fromString round-trip
  - DType.byteSize coverage
  - DType.toString-fromString invariant (every variant)
  - parse minimal one-tensor F32 file
  - parse two-tensor file with __metadata__
  - reject truncated header-size byte
  - reject header-larger-than-file
  - reject unknown dtype
  - reject byteLen mismatch with shape*dtype
  - reject overlapping data offsets
  - reject offset extending past data segment
- `examples/inspect.zig` — `safetensors-inspect <path>` CLI
- `bench/bench_parse.zig` — header-parse throughput on synthetic
  256-tensor fixture (no upstream-comparison numbers yet)
- AGPL-3.0 LICENSE
- README with honest non-claims

## What is open (deferred to v0.1)

- mmap-backed tensor views (currently `readAll` into caller buffer)
- Tensor dtype-typed views (f32 slice etc.) — currently `[]u8` only
- Big-endian host support
- Writing safetensors files (read-only at v0.0.1)
- Real-model-zoo test matrix (Llama-3.2 / Qwen / BERT fixtures)
- Cross-implementation bench vs `huggingface/safetensors` Rust crate
- Streaming / chunked read for huge tensors

## Honest non-claims

- **No perf parity claim** against the Rust upstream is made until the
  bench harness can compare against the same fixture loaded by
  `safetensors` (Rust crate). The bench currently measures self-speed
  on a 256-tensor synthetic header, which is not a comparable number.
- **No correctness claim** against the canonical model zoo is made
  until at least one real model file (e.g. a HF safetensors model) is
  in the test matrix. The 10 unit tests exercise the spec, not the
  ecosystem.
- **No memory-safety claim** beyond Zig's type+lifetime guarantees and
  the bounds-checks in the parser. No fuzz testing yet.
- **No big-endian host support claim**. Tested on x86_64-linux only.

## Next milestones

1. Add a real-model integration test: download a small HF safetensors
   file in CI (or vendor a 1-MB synthetic-from-real fixture) and
   exercise the full read path.
2. Add `huggingface/safetensors` (Rust) bench comparison. Likely path:
   wrap the Rust binary in a hyperfine call that processes the same
   fixture in both Rust and Zig and reports relative throughput.
3. Add mmap-backed tensor views for zero-copy reads on supported OSes.
4. Add fuzz tests: malformed JSON, malformed offsets, malformed shapes.
5. Promote to v0.1.0 once real-model fixtures land and bench parity
   (within 2x, ideally <1x) is documented.
