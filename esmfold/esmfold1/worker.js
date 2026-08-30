import init, { alloc_bytes, fold_esmfold1_from_ptr, initThreadPool, init_webgpu_backend } from './pkg/esmfold1.js';

const REPORT_INTERVAL = 200 * 1024 * 1024;
let poolInitialized = false;

self.onmessage = async (e) => {
  const { fasta, weightsUrl, threads } = e.data;

  try {
    self.postMessage({ type: 'status', message: 'Initialising WASM runtime...' });
    const wasm = await init();

    if (typeof initThreadPool === 'function' && !poolInitialized) {
      const numThreads = threads || navigator.hardwareConcurrency || 4;
      await initThreadPool(numThreads);
      poolInitialized = true;
      self.postMessage({ type: 'status', message: `Initialized WASM Rayon thread pool (${numThreads} threads).` });
    }

    if (typeof init_webgpu_backend === 'function') {
      self.postMessage({ type: 'status', message: 'Detecting WebGPU adapter...' });
      const gpuAvailable = await init_webgpu_backend();
      if (gpuAvailable) {
        self.postMessage({ type: 'status', message: 'WebGPU backend initialized & compute pipeline ready!' });
      } else {
        self.postMessage({ type: 'status', message: 'WebGPU unavailable. Using multi-threaded SIMD CPU fallback.' });
      }
    }

    self.postMessage({ type: 'status', message: `Streaming weights from ${weightsUrl}...` });
    const response = await fetch(weightsUrl);
    if (!response.ok) throw new Error(`HTTP ${response.status} fetching weights`);

    const contentLength = parseInt(response.headers.get('Content-Length') || '0', 10);
    if (!contentLength) throw new Error("Server must return Content-Length header.");

    self.postMessage({
      type: 'status',
      message: `Allocating ${(contentLength / (1024 * 1024 * 1024)).toFixed(2)} GB in WASM linear memory...`
    });

    const ptr = alloc_bytes(BigInt(contentLength));
    if (!ptr || ptr === 0n) {
      throw new Error("Failed to allocate linear memory for weights buffer.");
    }

    const reader = response.body.getReader();
    let currentPtr = Number(ptr);
    let bytesReceived = 0;
    let lastReportedBytes = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      new Uint8Array(wasm.memory.buffer, currentPtr, value.byteLength).set(value);
      currentPtr += value.byteLength;
      bytesReceived += value.byteLength;

      if (bytesReceived - lastReportedBytes >= REPORT_INTERVAL || bytesReceived === contentLength) {
        lastReportedBytes = bytesReceived;
        self.postMessage({
          type: 'status',
          message: `Loaded ${(bytesReceived / (1024 * 1024)).toFixed(0)} MB / ${(contentLength / (1024 * 1024)).toFixed(0)} MB`
        });
      }
    }

    self.postMessage({ type: 'status', message: 'Weights loaded. Starting fold...' });
    const startTime = performance.now();

    const onProgress = (stage, fraction) => {
      self.postMessage({
        type: 'status',
        message: `[${(fraction * 100).toFixed(0)}%] ${stage}`
      });
    };

    const pdb = fold_esmfold1_from_ptr(fasta, ptr, contentLength, onProgress);
    const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);

    self.postMessage({ type: 'complete', pdb, elapsed });
  } catch (err) {
    console.error('Worker failed:', err);
    self.postMessage({ type: 'error', message: err.toString() });
  }
};
