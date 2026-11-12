#!/usr/bin/env python3
"""Inspect a 128-byte original fixture; test explicit malformed-file refusals."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import tempfile


def require(condition, detail):
    if not condition:
        raise AssertionError(detail)


def verify(binary):
    binary = str(binary.resolve(strict=True))
    fixture = Path(__file__).resolve().parents[1] / "tests/fixtures/original-two-f32.safetensors"
    require(fixture.stat().st_size == 128, "The committed original fixture must remain 128 bytes")
    run = subprocess.run([binary, str(fixture)], capture_output=True, timeout=10)
    output = run.stdout + run.stderr
    require(run.returncode == 0, output)
    for marker in [b"TENSORS: 1", b"METADATA: 1", b"original-synthetic",
                   b"weights  F32  [2]  0..8  numel=2"]:
        require(marker in output, (marker, output))
    with tempfile.TemporaryDirectory(prefix="safetensors-task-") as temp:
        root = Path(temp)
        cases = [("short", b"abc", b"InvalidHeader"),
                 ("oversize", struct.pack("<Q", 2**64 - 1), b"HeaderTooLarge")]
        # Mutated copies are parser tests, not a format-writer API.
        for label, dtype, end, error in [
            ("dtype", "BOGUS", 8, b"UnknownDtype"),
            ("offset", "F32", 12, b"OffsetOutOfBounds"),
        ]:
            header = json.dumps({"weights": {"dtype": dtype, "shape": [2],
                                              "data_offsets": [0, end]}}, separators=(",", ":")).encode()
            header += b" " * (-len(header) % 8)
            cases.append((label, struct.pack("<Q", len(header)) + header + struct.pack("<2f", -1, 2), error))
        calls = []
        for name, data, error in cases:
            path = root / (name + ".safetensors")
            path.write_bytes(data)
            calls.append(([str(path)], error))
        calls += [([], b"usage:"), ([str(root / "missing")], b"FileNotFound")]
        for args, expected in calls:
            run = subprocess.run([binary, *args], capture_output=True, timeout=10)
            text = run.stdout + run.stderr
            require(run.returncode != 0 and expected in text, (args, run.returncode, text))
            require(b"panic:" not in text and b"Segmentation fault" not in text, text)
    print(json.dumps({"status": "PASS", "original_fixture_bytes": fixture.stat().st_size,
                      "tensors": 1, "shape": [2], "dtype": "F32", "data_bytes": 8,
                      "refusal_cases": len(calls), "scope": "Synthetic header inspection; no real-model or speed claim"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("zig-out/bin/safetensors-inspect"))
    args = parser.parse_args()
    verify(args.binary)
