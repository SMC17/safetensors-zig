#!/usr/bin/env bash
# Head-to-head bench: safetensors-zig vs huggingface/safetensors (Rust)
# against tests/fixtures/tinyllama-shape.safetensors (33MB Llama-shape).
#
# Usage:
#   bench/compare.sh
#
# Requires: zig 0.16+, cargo, python3, internet (for crates.io on first run).
# Build the fixture first: python3 tests/gen_fixture.py --zero

set -euo pipefail
cd "$(dirname "$0")/.."

FIXTURE=tests/fixtures/tinyllama-shape.safetensors
if [ ! -f "$FIXTURE" ]; then
  echo "[gen] $FIXTURE missing — building..."
  python3 tests/gen_fixture.py --zero
fi

echo "=== Zig (safetensors-zig) ==="
zig build bench-realmodel
ZIG_NS=$(zig build bench-realmodel 2>&1 | awk '/^ns_per_parse/ {print $2}')

echo
echo "=== Rust (huggingface/safetensors) ==="
(
  cd bench/bench_rust
  if [ ! -d target/release ]; then
    echo "[build] cargo build --release (one-time)..."
    cargo build --release
  fi
  ./target/release/bench-rust
)
RUST_NS=$(cd bench/bench_rust && ./target/release/bench-rust 2>&1 | awk '/^ns_per_parse/ {print $2}')

echo
echo "=== Comparison ==="
if [ -n "${ZIG_NS:-}" ] && [ -n "${RUST_NS:-}" ]; then
  RATIO=$(awk "BEGIN { printf \"%.3f\", $ZIG_NS / $RUST_NS }")
  echo "zig:   $ZIG_NS  ns/parse"
  echo "rust:  $RUST_NS  ns/parse"
  echo "zig/rust ratio: $RATIO"
  if (( $(awk "BEGIN { print ($RATIO < 1.0) }") )); then
    echo "→ Zig is faster (ratio < 1.0)"
  elif (( $(awk "BEGIN { print ($RATIO < 1.5) }") )); then
    echo "→ Within 1.5x — competitive"
  else
    echo "→ Rust faster by >1.5x — honest non-claim territory"
  fi
fi
