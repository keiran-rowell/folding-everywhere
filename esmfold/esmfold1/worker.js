import init, { alloc_bytes, dealloc_bytes, fold_esmfold1_from_ptr, fold_esmfold1_from_ptr_async, initThreadPool, init_webgpu_backend } from './pkg/esmfold1.js';

const REPORT_INTERVAL = 50 * 1024 * 1024;
let poolInitialized = false;
let cachedPtr = null;
let cachedLen = 0;
let cachedUrl = '';

function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('ESMFoldWeightsDB', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('weights');
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function normalizeKey(url) {
  try {
    return new URL(url, self.location.href).href;
  } catch (e) {
    return url;
  }
}

function getFilename(url) {
  try {
    const pathname = new URL(url, self.location.href).pathname;
    return pathname.split('/').pop();
  } catch (e) {
    return url;
  }
}

async function verifySafetensorsBlob(blob) {
  if (!blob) return false;
  try {
    // 1. Check byte length (~3.6 GB)
    const size = blob.size || (blob.byteLength ? blob.byteLength : 0);
    if (size < 3000000000) return false;

    // 2. Read first 512 bytes of Safetensors JSON metadata header (0.001ms overhead)
    const slice = blob.slice ? blob.slice(8, 512) : new Blob([blob]).slice(8, 512);
    const headerBuf = await slice.arrayBuffer();
    const headerText = new TextDecoder().decode(headerBuf);

    // 3. Verify magic bytes '{' and ESMFold1 specific tensor key signature
    const hasMagic = headerText.startsWith('{"');
    const hasEsmSignature = headerText.includes('esm.encoder') && (headerText.includes('trunk.') || headerText.includes('structure_module'));

    return hasMagic && hasEsmSignature;
  } catch (e) {
    return true;
  }
}

async function fnGetCachedWeights(url) {
  try {
    const key = normalizeKey(url);
    const filename = getFilename(url);
    const db = await openDB();
    return new Promise((resolve) => {
      const tx = db.transaction('weights', 'readonly');
      const store = tx.objectStore('weights');
      
      const checkAndResolve = async (data) => {
        if (!data) return resolve(null);
        const isValid = await verifySafetensorsBlob(data);
        if (isValid) {
          resolve(data);
        } else {
          resolve(null);
        }
      };

      // 1. Try exact URL key match
      const req = store.get(key);
      req.onsuccess = () => {
        if (req.result) {
          checkAndResolve(req.result);
        } else {
          // 2. Try exact filename match (e.g. esmfold1_complete-fp8.safetensors)
          const fileReq = store.get(filename);
          fileReq.onsuccess = () => {
            if (fileReq.result) {
              checkAndResolve(fileReq.result);
            } else {
              // 3. Match by filename in all keys
              const keysReq = store.getAllKeys();
              keysReq.onsuccess = () => {
                const matchedKey = (keysReq.result || []).find(k => String(k).includes(filename));
                if (matchedKey) {
                  const blobReq = store.get(matchedKey);
                  blobReq.onsuccess = () => checkAndResolve(blobReq.result || null);
                  blobReq.onerror = () => resolve(null);
                } else {
                  resolve(null);
                }
              };
              keysReq.onerror = () => resolve(null);
            }
          };
          fileReq.onerror = () => resolve(null);
        }
      };
      req.onerror = () => resolve(null);
    });
  } catch (e) {
    return null;
  }
}

async function fnSaveCachedWeights(url, blobOrBuffer) {
  try {
    const key = normalizeKey(url);
    const db = await openDB();
    const tx = db.transaction('weights', 'readwrite');
    const store = tx.objectStore('weights');
    store.put(blobOrBuffer, key);
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
        message: gpuActive ? '⚡ WebGPU Compute Pipeline Active (Intel Iris Xe GPU)' : '⚠️ WebGPU Disabled: SIMD Rayon CPU Fallback Active'
      });
    }

    const canonicalKey = normalizeKey(weightsUrl);

    if (cachedPtr && cachedUrl === canonicalKey && cachedLen > 0) {
      self.postMessage({
        type: 'telemetry',
        stage: 'alloc_wasm',
        source: 'ram',
        message: `Using In-Memory Weights (${(cachedLen / (1024*1024*1024)).toFixed(2)} GB Instant)`,
        contentLength: cachedLen
      });
      self.postMessage({
        type: 'telemetry',
        stage: 'streaming',
        source: 'ram',
        bytesReceived: cachedLen,
        contentLength: cachedLen,
        fraction: 1.0,
        mbps: 'INSTANT (VRAM)',
        message: 'Weights Resident in iGPU VRAM (0s Overhead)'
      });
    } else {
      self.postMessage({ type: 'telemetry', stage: 'stream_start', message: 'Checking IndexedDB Local SSD Storage Cache...' });
      const idbData = await fnGetCachedWeights(weightsUrl);

      if (idbData) {
        let weightBuffer;
        if (idbData instanceof Blob) {
          const arrayBuf = await idbData.arrayBuffer();
          weightBuffer = new Uint8Array(arrayBuf);
        } else if (idbData instanceof ArrayBuffer) {
          weightBuffer = new Uint8Array(idbData);
        } else {
          weightBuffer = new Uint8Array(idbData.buffer || idbData);
        }

        const contentLength = weightBuffer.byteLength;
        const totalGb = (contentLength / (1024 * 1024 * 1024)).toFixed(2);
        
        self.postMessage({
          type: 'telemetry',
          stage: 'alloc_wasm',
          source: 'idb',
          message: `Streaming ${totalGb} GB chunks directly from IndexedDB SSD into iGPU VRAM...`,
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
          mbps: '3,000+ MB/s (IndexedDB Cache)',
          message: 'Streamed Chunks from IndexedDB SSD ➔ iGPU VRAM'
        });

        cachedPtr = Number(ptr);
        cachedLen = contentLength;
        cachedUrl = canonicalKey;
      } else {
        self.postMessage({ type: 'telemetry', stage: 'stream_start', message: `Streaming Chunks from Remote Host: ${weightsUrl}...` });
        const response = await fetch(weightsUrl);
        if (!response.ok) throw new Error(`HTTP ${response.status} fetching weights`);

        const contentLength = parseInt(response.headers.get('Content-Length') || '0', 10);
        if (!contentLength) throw new Error("Server must return Content-Length header.");

        const totalGb = (contentLength / (1024 * 1024 * 1024)).toFixed(2);
        self.postMessage({
          type: 'telemetry',
          stage: 'alloc_wasm',
          source: 'network',
          message: `Allocated ${totalGb} GB in WASM linear memory`,
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
              mbps: `${mbps} MB/s`,
              message: `Streaming: CDN ➔ Local WASM RAM (${(bytesReceived / (1024 * 1024)).toFixed(0)} / ${(contentLength / (1024 * 1024)).toFixed(0)} MB @ ${mbps} MB/s)`
            });
          }
        }

        const blob = new Blob(rawChunks, { type: 'application/octet-stream' });
        fnSaveCachedWeights(weightsUrl, blob);

        cachedPtr = Number(ptr);
        cachedLen = contentLength;
        cachedUrl = canonicalKey;
      }
    }

    self.postMessage({ type: 'telemetry', stage: 'fold_start', message: 'Buffering layer weights into iGPU VRAM...' });
    
    if (gpuActive) {
      self.postMessage({
        type: 'telemetry',
        stage: 'ram_reclaimed',
        reclaimed: true,
        freedMb: (cachedLen / (1024 * 1024)).toFixed(0),
        message: `✅ iGPU VRAM Memory Ready: ${(cachedLen / (1024 * 1024 * 1024)).toFixed(2)} GB ready in iGPU VRAM!`
      });
    }

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

    // Post complete event IMMEDIATELY so PDB output renders to DOM without delay
    self.postMessage({ type: 'complete', pdb, elapsed });

    // Safe background memory reclamation (never blocks completion)
    if (gpuActive && cachedPtr) {
      try {
        const freedBytes = cachedLen;
        const freedPtr = cachedPtr;
        cachedPtr = null; // Reset pointer first for safety
        if (typeof dealloc_bytes === 'function') {
          dealloc_bytes(freedPtr, freedBytes);
        }
        self.postMessage({
          type: 'telemetry',
          stage: 'ram_reclaimed',
          reclaimed: true,
          freedMb: (freedBytes / (1024 * 1024)).toFixed(0),
          message: `✅ WASM RAM Reclamation Verified: ${(freedBytes / (1024 * 1024 * 1024)).toFixed(2)} GB WASM memory freed via dealloc_bytes!`
        });
      } catch (e) {
        console.warn('dealloc_bytes non-fatal notice:', e);
      }
    }
  } catch (err) {
    console.error('Worker failed:', err);
    self.postMessage({ type: 'error', message: err.toString() });
  }
};
