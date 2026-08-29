import init, { fold_esmfold1 } from './pkg/esmfold.js';

self.onmessage = async (e) => {
  const { fasta, weightsUrl } = e.data;

  try {
    self.postMessage({ type: 'status', message: 'Initialising 64-bit WASM runtime...' });
    await init();

    self.postMessage({ type: 'status', message: `Downloading weights from ${weightsUrl}...` });
    const response = await fetch(weightsUrl);
    if (!response.ok) throw new Error(`HTTP ${response.status} fetching weights`);

    const arrayBuffer = await response.arrayBuffer();
    const weightBytes = new Uint8Array(arrayBuffer);

    self.postMessage({ type: 'status', message: 'Running ESMFold1 inference...' });
    const startTime = performance.now();
    
    // Ingests bytes and executes full pipeline -> returns PDB string
    const pdb = fold_esmfold1(fasta, weightBytes);
    const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);

    self.postMessage({ type: 'complete', pdb, elapsed });
  } catch (err) {
    self.postMessage({ type: 'error', message: err.toString() });
  }
};
