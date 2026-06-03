# NEXT — safetensors-zig

_Last update: 2026-06-02 (bootstrapped per C17)_

## What just shipped

- HEAD `5fbd402` — parser: explicit `@Vector(32, u8)` scan for quote/backslash; **Type-1 catch on 64-lane** (i.e., 64-lane variant was attempted but didn't beat 32-lane on this hardware)
- v0.3.0 series — FIXTURES.md calibration baseline + curation rationale; CHANGELOG describes the SIMD evolution

## Next concrete action

Bench the `@Vector(32, u8)` structural scan on Apple Silicon NEON to verdict `exp-1780011142-037633785` (claim: ≥2× over `std.mem.indexOfAny` on JSON-shape data). This is Mac-blocked from this Linux box — either ssh to mac host or move the bench harness. Once measured, file `/verdict` with median-of-N output.

## In-flight hypotheses

| exp-id | hypothesis | threshold | next measurement |
|--------|------------|-----------|------------------|
| `exp-1780011142-037633785` | @Vector(32,u8) scan beats std.mem.indexOfAny ≥2× on JSON on Apple Silicon NEON | ≥2× median throughput on Apple Silicon | bench/bench_simd_scan.zig median-of-5 on M-series mac |

## Context not obvious from git

- 21 tests at HEAD per `grep test "" src/ tests/`.
- The 32-lane vs 64-lane Type-1 is a substantive finding — encode it in STATUS.md before the next lever attempts to revisit 64-lane (avoid re-discovery cost).
- FIXTURES.md is the substrate calibration baseline — every bench claim cites the fixture used.
- v0.3.0 is current (memory was previously stale at v0.0.1).
- Pre-1.0; Zig 0.16 bound; AGPL.
- Public-surface repo — claim-scan PostToolUse hook fires on edits to this tree.
