# safetensors-zig Agent Entry Point

Zig 0.16 reader for the HuggingFace safetensors tensor serialization format.

## Before you change anything

1. Read `README.md` and `STATUS.md`.
2. Run the proof gate before claiming correctness or speed.
3. Keep honest non-claims: speed ratios require matched harness + hardware class.

## Proof gate

```sh
zig build test --summary all
```

## Allowed work

- docs/status reconciliation
- parser correctness fixes with unit + fixture tests
- narrow performance work with re-measured benches
- CI / proof-gate maintenance

## Forbidden without explicit operator intent

- committing large real model weight files as published artifacts
- overclaiming vs HF Rust without a re-run of the stated harness
- shipping a writer path without tests and STATUS update
- release cuts without green CI

## Escalation

Ask before release claims, large fixture commits, or format-writer design changes.

