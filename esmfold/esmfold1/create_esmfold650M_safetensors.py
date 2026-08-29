import torch
import gc
from safetensors.torch import save_file
import urllib.request
import os

os.makedirs("weights", exist_ok=True)

# 1. Fetch ESM-2 650M and ESMFold Trunk Checkpoints
print("Downloading ESM-2 650M & ESMFold trunk weights...")
torch.hub.download_url_to_file(
    "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt",
    "weights/esm2_t33_650M_UR50D.pt"
)
torch.hub.download_url_to_file(
    "https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v1.pt",
    "weights/esmfold_trunk.pt"
)

# 2. Load with memory mapping
esm_raw = torch.load("weights/esm2_t33_650M_UR50D.pt", map_location="cpu", weights_only=False)
trunk_raw = torch.load("weights/esmfold_trunk.pt", map_location="cpu", weights_only=False)

esm_state = esm_raw.get("model", esm_raw)
trunk_state = trunk_raw.get("model", trunk_raw)

combined = {}

# Map ESM-2 650M keys to HuggingFace/Rust naming scheme in BF16
for k, v in esm_state.items():
    if isinstance(v, torch.Tensor):
        key = k
        if not key.startswith("esm."):
            key = f"esm.{key}"
        # Standardize naming aliases
        if key == "esm.embed_tokens.weight":
            key = "esm.embeddings.word_embeddings.weight"
        if "esm.layers." in key:
            key = key.replace("esm.layers.", "esm.encoder.layer.")
        combined[key] = v.to(torch.bfloat16).contiguous()

# Add trunk tensors in BF16
for k, v in trunk_state.items():
    if isinstance(v, torch.Tensor):
        combined[k] = v.to(torch.bfloat16).contiguous()

del esm_raw, trunk_raw, esm_state, trunk_state
gc.collect()

output_file = "weights/esmfold_650M_v1.safetensors"
save_file(combined, output_file)
size_mb = os.path.getsize(output_file) / (1024 * 1024)
print(f"Export complete: {len(combined)} tensors saved to {output_file} ({size_mb:.1f} MB)")
