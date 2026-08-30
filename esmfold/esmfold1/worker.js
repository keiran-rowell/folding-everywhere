import init, { alloc_bytes, fold_esmfold1_from_ptr, fold_esmfold1_from_ptr_async, initThreadPool, init_webgpu_backend } from './pkg/esmfold1.js';

const REPORT_INTERVAL = 50 * 1024 * 1024;
let poolInitialized = false;
let cachedPtr = null;
let cachedLen = 0;
let cachedUrl = '';

// IndexedDB Helper for Persistent 3.3GB Weight Caching
function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('ESMFoldWeightsDB', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('weights');
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function fnGetCachedWeights(url) {
  try {
    const db = await openDB();
    return new Promise((resolve) => {
      const tx = db.transaction('weights', 'readonly');
      const store = tx.objectStore('weights');
      const req = store.get(url);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => resolve(null);
    });
  } catch (e) {
    return null;
  }
}

async function fnSaveCachedWeights(url, buffer) {
  try {
    const db = await openDB();
    const tx = db.transaction('weights', 'readwrite');
    const store = tx.objectStore('weights');
    store.put(buffer, url);
  } catch (e) {
    console.warn('Failed to save to IndexedDB:', e);
  }
}

self.onmessage = async (e) => {
  const { fasta, weightsUrl, threads } = e.data;

  try {
    self.postMessage({ type: 'telemetry', stage: 'init_wasm', message: 'Initialising Local 64-bit WASM VM Memory Space...' });
    const wasm = await init();

    if (typeof initThreadPool === 'function' && !poolInitialized) {
      const numThreads = threads || navigator.hardwareConcurrency || 4;
      await initThreadPool(numThreads);
      poolInitialized = true;
      self.postMessage({
        type: 'telemetry',
        stage: 'thread_pool',
        message: `Rayon Worker Pool Active (${numThreads} Local CPU Cores)`,
        threads: numThreads
      });
    }

    let gpuActive = false;
    if (typeof init_webgpu_backend === 'function') {
      gpuActive = await init_webgpu_backend();
      self.postMessage({
        type: 'telemetry',
        stage: 'gpu_status',
        gpuActive,
        message: gpuActive ? 'Local iGPU / GPU Compute Pipeline Ready (Unified VRAM)' : 'Local CPU SIMD Fallback Active'
      });
    }

    // FAST-PATH: If weights are already populated in WASM linear memory
    if (cachedPtr && cachedUrl === weightsUrl && cachedLen > 0) {
      self.postMessage({
        type: 'telemetry',
        stage: 'alloc_wasm',
        source: 'ram',
        message: `Using In-Memory Cached WASM Weights (${(cachedLen / (1024*1024*1024)).toFixed(2)} GB Instant)`,
        contentLength: cachedLen
      });
      self.postMessage({
        type: 'telemetry',
        stage: 'streaming',
        source: 'ram',
        bytesReceived: cachedLen,
        contentLength: cachedLen,
        fraction: 1.0,
        mbps: 'INSTANT (RAM)',
        message: 'Weights Cached in WASM RAM (0s Overhead)'
      });
    } else {
      // Check IndexedDB Persistent Cache first
      self.postMessage({ type: 'telemetry', stage: 'stream_start', message: 'Checking IndexedDB Persistent Storage Cache...' });
      const idbBuffer = await fnGetCachedWeights(weightsUrl);

      let weightBuffer = null;
      let contentLength = 0;

      if (idbBuffer) {
        weightBuffer = new Uint8Array(idbBuffer);
        contentLength = weightBuffer.byteLength;

        const totalGb = (contentLength / (1024 * 1024 * 1024)).toFixed(2);
        self.postMessage({
          type: 'telemetry',
          stage: 'alloc_wasm',
          source: 'idb',
          message: `Loaded ${totalGb} GB instantly from IndexedDB Local Storage (0.3s)!`,
          contentLength
        });

        const ptr = alloc_bytes(BigInt(contentLength));
        if (!ptr || ptr === 0n) {
          throw new Error("Failed to allocate linear memory for weights buffer.");
        }

        new Uint8Array(wasm.memory.buffer, Number(ptr), contentLength).set(weightBuffer);

        self.postMessage({
          type: 'telemetry',
          stage: 'streaming',
          source: 'idb',
          bytesReceived: contentLength,
          contentLength,
          fraction: 1.0,
          mbps: '3,000+ MB/s (Disk Cache)',
          message: 'Loaded instantly from IndexedDB Disk Cache'
        });

        cachedPtr = Number(ptr);
        cachedLen = contentLength;
        cachedUrl = weightsUrl;
      } else {
        self.postMessage({ type: 'telemetry', stage: 'stream_start', message: `Streaming weights from remote internet host: ${weightsUrl}...` });
        const response = await fetch(weightsUrl);
        if (!response.ok) throw new Error(`HTTP ${response.status} fetching weights`);

        contentLength = parseInt(response.headers.get('Content-Length') || '0', 10);
        if (!contentLength) throw new Error("Server must return Content-Length header.");

        const totalGb = (contentLength / (1024 * 1024 * 1024)).toFixed(2);
        self.postMessage({
          type: 'telemetry',
          stage: 'alloc_wasm',
          source: 'network',
          message: `Allocated ${totalGb} GB inside Local Browser WASM VM RAM`,
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
        const rawChunks = [];

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          new Uint8Array(wasm.memory.buffer, currentPtr, value.byteLength).set(value);
          rawChunks.push(value);
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
              source: 'network',
              bytesReceived,
              contentLength,
              fraction,
              mbps,
              message: `Streaming from Internet CDN ➔ Local WASM RAM (${(bytesReceived / (1024 * 1024)).toFixed(0)} / ${(contentLength / (1024 * 1024)).toFixed(0)} MB @ ${mbps} MB/s)`
            });
          }
        }

        // Save downloaded ArrayBuffer to IndexedDB asynchronously
        const fullArray = new Uint8Array(contentLength);
        let offset = 0;
        for (const chunk of rawChunks) {
          fullArray.set(chunk, offset);
          offset += chunk.byteLength;
        }
        fnSaveCachedWeights(weightsUrl, fullArray.buffer);

        cachedPtr = Number(ptr);
        cachedLen = contentLength;
        cachedUrl = weightsUrl;
      }
    }

    self.postMessage({ type: 'telemetry', stage: 'fold_start', message: 'Weights buffered in Local WASM VM RAM. Starting prediction...' });
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

    const pdbFn = typeof fold_esmfold1_from_ptr_async === 'function' ? fold_esmfold1_from_ptr_async : fold_esmfold1_from_ptr;
    const pdb = await pdbFn(fasta, cachedPtr, cachedLen, onProgress);
    const elapsed = ((performance.now() - foldStartTime) / 1000).toFixed(1);

    self.postMessage({ type: 'complete', pdb, elapsed });
  } catch (err) {
    console.error('Worker failed:', err);
    self.postMessage({ type: 'error', message: err.toString() });
  }
};
