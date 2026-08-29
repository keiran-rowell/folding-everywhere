import init, { alloc_bytes, fold_esmfold1_from_ptr } from './pkg/esmfold.js';

self.onmessage = async (e) => {
  const { fasta, weightsUrl } = e.data;

  try {
    self.postMessage({ type: 'status', message: 'Initialising 64-bit WASM runtime...' });
    const wasm = await init();

    self.postMessage({ type: 'status', message: `Streaming weights from ${weightsUrl}...` });
    const response = await fetch(weightsUrl);
    if (!response.ok) throw new Error(`HTTP ${response.status} fetching weights`);

    const contentLength = parseInt(response.headers.get('Content-Length') || '0', 10);
    if (!contentLength) throw new Error("Server must return Content-Length header.");

    self.postMessage({
      type: 'status',
      message: `Allocating ${(contentLength / (1024 * 1024 * 1024)).toFixed(2)} GB in WASM linear memory...`
    });

    // Pass BigInt for wasm64 memory size, get pointer offset
    const ptr = alloc_bytes(BigInt(contentLength));
    if (!ptr || ptr === 0) {
      throw new Error("Failed to allocate linear memory for weights buffer.");
    }

    const reader = response.body.getReader();
    let currentPtr = Number(ptr);
    let bytesReceived = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      // Always access wasm.memory.buffer dynamically on each chunk
      new Uint8Array(wasm.memory.buffer, currentPtr, value.byteLength).set(value);

      currentPtr += value.byteLength;
      bytesReceived += value.byteLength;

      if (bytesReceived % (200 * 1024 * 1024) === 0 || bytesReceived === contentLength) {
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

    // Pass ptr (number) and length (BigInt)
    const pdb = fold_esmfold1_from_ptr(fasta, ptr, BigInt(contentLength), onProgress);
    const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);

    self.postMessage({ type: 'complete', pdb, elapsed });
  } catch (err) {
    console.error('Worker failed:', err);
    self.postMessage({ type: 'error', message: err.toString() });
  }
};
