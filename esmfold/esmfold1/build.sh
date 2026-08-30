#!/usr/bin/env bash
set -euo pipefail

export RUSTFLAGS="-C target-cpu=generic -C target-feature=+simd128,+bulk-memory,+atomics,+mutable-globals -C link-arg=--import-memory -C link-arg=--shared-memory -C link-arg=--max-memory=4294967296 -C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size -C link-arg=--export=__tls_align -C link-arg=--export=__tls_base"

cargo +nightly build \
  --target wasm32-unknown-unknown \
  -Z build-std=std,panic_abort \
  --release \
  --lib

# Run wasm-bindgen CLI directly to generate pkg bindings
wasm-bindgen \
  --target web \
  --out-dir pkg \
  ../../target/wasm32-unknown-unknown/release/esmfold1.wasm

echo "WASM multithreaded build complete."
