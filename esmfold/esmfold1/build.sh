#!/usr/bin/env bash
set -euo pipefail

export RUSTFLAGS="-C target-cpu=generic -C target-feature=+simd128,+bulk-memory"

cargo +nightly build \
  --target wasm64-unknown-unknown \
  -Z build-std=std,panic_abort \
  --release \
  --lib

# Run wasm-bindgen CLI directly to generate pkg bindings
wasm-bindgen \
  --target web \
  --out-dir pkg \
  ../../target/wasm64-unknown-unknown/release/esmfold1.wasm

echo "WASM64 build complete."
