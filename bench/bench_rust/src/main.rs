//! Real-model fixture parse bench (Rust side of the Zig-vs-Rust
//! comparison).
//!
//! Reads tests/fixtures/tinyllama-shape.safetensors entirely into
//! memory once, then runs N parse iterations from that buffer via
//! the official `safetensors::SafeTensors::deserialize` to isolate
//! parse cost from disk I/O. Reports ns/parse and parses/sec on the
//! same metric scale as the Zig companion bench at
//! ../bench_realmodel.zig.

use std::time::Instant;

const ITERS: u64 = 1_000;
const FIXTURE: &str = "../../tests/fixtures/tinyllama-shape.safetensors";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(FIXTURE).map_err(|e| {
        eprintln!(
            "fixture {} not present. Build it with:\n  python3 tests/gen_fixture.py --zero\n  (run from the safetensors-zig repo root)",
            FIXTURE
        );
        e
    })?;

    // Warmup
    {
        let _ = safetensors::SafeTensors::deserialize(&bytes)?;
    }

    let start = Instant::now();
    for _ in 0..ITERS {
        let _ = safetensors::SafeTensors::deserialize(&bytes)?;
    }
    let elapsed = start.elapsed();
    let elapsed_ns = elapsed.as_nanos() as u64;
    let ns_per_iter = elapsed_ns / ITERS;
    let parses_per_sec = (ITERS as u128 * 1_000_000_000u128 / elapsed_ns as u128) as u64;

    println!("fixture        {}", FIXTURE);
    println!("file_bytes     {}", bytes.len());
    println!("iters          {}", ITERS);
    println!("ns_per_parse   {}", ns_per_iter);
    println!("us_per_parse   {}", ns_per_iter / 1000);
    println!("parses_per_sec {}", parses_per_sec);

    Ok(())
}
