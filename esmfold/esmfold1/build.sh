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

# Patch wasm-bindgen-rayon snippet to import esmfold1.js explicitly instead of directory URL
python3 -c "
import glob
for f in glob.glob('pkg/snippets/wasm-bindgen-rayon-*/src/workerHelpers.js'):
    with open(f, 'r') as file:
        code = file.read()
    code = code.replace(\"import('../../..')\", \"import('../../../esmfold1.js')\")
    with open(f, 'w') as file:
        file.write(code)
    print(f'Patched {f}')
"

echo "WASM multithreaded build complete."
