import init, { alloc_bytes, fold_esmfold1_from_ptr, initThreadPool, init_webgpu_backend } from './pkg/esmfold1.js';

const REPORT_INTERVAL = 50 * 1024 * 1024; // Report every 50 MB
let poolInitialized = false;

self.onmessage = async (e) => {
  const { fasta, weightsUrl, threads } = e.data;

  try {
    self.postMessage({ type: 'telemetry', stage: 'init_wasm', message: 'Initialising WASM 64-bit runtime & panic hooks...' });
    const wasm = await init();

    if (typeof initThreadPool === 'function' && !poolInitialized) {
      const numThreads = threads || navigator.hardwareConcurrency || 4;
      await initThreadPool(numThreads);
      poolInitialized = true;
      self.postMessage({
        type: 'telemetry',
        stage: 'thread_pool',
        message: `Rayon Web Worker thread pool active (${numThreads} threads).`,
        threads: numThreads
      });
    }

    let gpuActive = false;
    if (typeof init_webgpu_backend === 'function') {
      self.postMessage({ type: 'telemetry', stage: 'gpu_detect', message: 'Querying WebGPU adapter & compiling WGSL shaders...' });
      gpuActive = await init_webgpu_backend();
      self.postMessage({
        type: 'telemetry',
        stage: 'gpu_status',
        gpuActive,
        message: gpuActive ? 'WebGPU WGSL FP8 Compute Pipeline Compiled & Ready' : 'WebGPU Unavailable (Using SIMD CPU Fallback)'
      });
    }

    self.postMessage({ type: 'telemetry', stage: 'stream_start', message: `Opening HTTP stream to ${weightsUrl}...` });
    const response = await fetch(weightsUrl);
    if (!response.ok) throw new Error(`HTTP ${response.status} fetching weights`);

    const contentLength = parseInt(response.headers.get('Content-Length') || '0', 10);
    if (!contentLength) throw new Error("Server must return Content-Length header.");

    const totalGb = (contentLength / (1024 * 1024 * 1024)).toFixed(2);
    self.postMessage({
      type: 'telemetry',
      stage: 'alloc_wasm',
      message: `Allocating ${totalGb} GB in WASM linear memory...`,
      contentLength
    });

    const ptr = alloc_bytes(BigInt(contentLength));
    if (!ptr || ptr === 0n) {
      throw new Error("Failed to allocate linear memory for weights buffer.");
    }

    const reader = response.body.getReader();
    let currentPtr = Number(ptr);
    let bytesReceived = 0;
    let lastReportedBytes = 0;
    const startTime = performance.now();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      new Uint8Array(wasm.memory.buffer, currentPtr, value.byteLength).set(value);
      currentPtr += value.byteLength;
      bytesReceived += value.byteLength;

      if (bytesReceived - lastReportedBytes >= REPORT_INTERVAL || bytesReceived === contentLength) {
        lastReportedBytes = bytesReceived;
        const elapsedSec = (performance.now() - startTime) / 1000;
        const mbps = elapsedSec > 0 ? ((bytesReceived / (1024 * 1024)) / elapsedSec).toFixed(1) : '0';
        const fraction = bytesReceived / contentLength;

        self.postMessage({
          type: 'telemetry',
          stage: 'streaming',
          bytesReceived,
          contentLength,
          fraction,
          mbps,
          message: `Streaming Weights: ${(bytesReceived / (1024 * 1024)).toFixed(0)} MB / ${(contentLength / (1024 * 1024)).toFixed(0)} MB (${mbps} MB/s)`
        });
      }
    }

    self.postMessage({ type: 'telemetry', stage: 'fold_start', message: 'Weights buffered in WASM VM memory. Launching ESMFold...' });
    const foldStartTime = performance.now();

    const onProgress = (stageName, fraction) => {
      self.postMessage({
        type: 'telemetry',
        stage: 'folding',
        foldStage: stageName,
        fraction,
        message: `[${(fraction * 100).toFixed(0)}%] ${stageName}`
      });
    };

    const pdb = fold_esmfold1_from_ptr(fasta, ptr, contentLength, onProgress);
    const elapsed = ((performance.now() - foldStartTime) / 1000).toFixed(1);

    self.postMessage({ type: 'complete', pdb, elapsed });
  } catch (err) {
    console.error('Worker failed:', err);
    self.postMessage({ type: 'error', message: err.toString() });
  }
};
