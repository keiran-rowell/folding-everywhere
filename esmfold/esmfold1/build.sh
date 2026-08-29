#!/usr/bin/env bash
set -euo pipefail

echo "==> Building release WASM library..."
RUSTFLAGS="-C target-feature=+simd128" cargo build --lib --target wasm32-unknown-unknown --release

echo "==> Generating wasm-bindgen artifacts into pkg/..."
wasm-bindgen ../../target/wasm32-unknown-unknown/release/esmfold.wasm \
  --out-dir pkg \
  --target web \
  --typescript

echo "==> Build complete. Artifacts in pkg/:"
ls -lh pkg/*.wasm pkg/*.js
