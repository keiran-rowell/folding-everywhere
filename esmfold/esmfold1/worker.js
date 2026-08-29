import init, { alloc_bytes, fold_esmfold1_from_ptr } from './pkg/esmfold.js';

let wasmInstance = null;
let cachedPtr = null;
let cachedContentLength = 0;
let isLoaded = false;
let loadPromise = null;

async function ensureWeightsLoaded(weightsUrl) {
  if (isLoaded) return;
  if (loadPromise) return loadPromise;

  loadPromise = (async () => {
    self.postMessage({ type: 'status', message: 'Initialising WASM runtime...' });
    wasmInstance = await init();

    self.postMessage({ type: 'status', message: `Streaming weights from ${weightsUrl}...` });
    const response = await fetch(weightsUrl);
    if (!response.ok) throw new Error(`HTTP ${response.status} fetching weights`);

    const contentLength = parseInt(response.headers.get('Content-Length') || '0', 10);
    if (!contentLength) throw new Error("Server must return Content-Length header.");

    self.postMessage({
      type: 'status',
      message: `Allocating ${(contentLength / (1024 * 1024 * 1024)).toFixed(2)} GB in WASM linear memory...`
    });

    // Allocate once in WASM linear memory
    const ptr = alloc_bytes(BigInt(contentLength));
    if (!ptr || ptr === 0) {
      throw new Error("Failed to allocate linear memory for weights buffer.");
    }

    cachedPtr = ptr;
    cachedContentLength = contentLength;

    const reader = response.body.getReader();
    let currentPtr = Number(ptr);
    let bytesReceived = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      // Access wasmInstance.memory.buffer dynamically on each chunk in case of memory.grow
      new Uint8Array(wasmInstance.memory.buffer, currentPtr, value.byteLength).set(value);

      currentPtr += value.byteLength;
      bytesReceived += value.byteLength;

      if (bytesReceived % (200 * 1024 * 1024) === 0 || bytesReceived === contentLength) {
        self.postMessage({
          type: 'status',
          message: `Loaded ${(bytesReceived / (1024 * 1024)).toFixed(0)} MB / ${(contentLength / (1024 * 1024)).toFixed(0)} MB`
        });
      }
    }

    isLoaded = true;
    self.postMessage({ type: 'status', message: 'Weights loaded and pinned in memory. Engine ready.' });
  })();

  return loadPromise;
}

self.onmessage = async (e) => {
  const { fasta, weightsUrl } = e.data;

  try {
    // 1. Paid only on the first call (or skips straight through if already loaded)
    await ensureWeightsLoaded(weightsUrl);

    // 2. Fast inference path
    self.postMessage({ type: 'status', message: 'Starting fold inference...' });
    const startTime = performance.now();

    const onProgress = (stage, fraction) => {
      self.postMessage({
        type: 'status',
        message: `[${(fraction * 100).toFixed(0)}%] ${stage}`
      });
    };

    const pdb = fold_esmfold1_from_ptr(
      fasta,
      cachedPtr,
      BigInt(cachedContentLength),
      onProgress
    );
    const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);

    self.postMessage({ type: 'complete', pdb, elapsed });
  } catch (err) {
    console.error('Worker failed:', err);
    self.postMessage({ type: 'error', message: err.toString() });
  }
};
