import torch
from safetensors.torch import save_file

# PyTorch 2.6+ defaults to weights_only=True, which blocks OmegaConf objects
state = torch.load("weights/esmfold_3B_v1.pt", map_location="cpu", weights_only=False)

if "model" in state:
    state = state["model"]

# Keep only float32 contiguous tensors
clean_state = {k: v.contiguous().float() for k, v in state.items() if isinstance(v, torch.Tensor)}

save_file(clean_state, "weights/esmfold_3B_v1.safetensors")
print(f"Exported {len(clean_state)} tensors to weights/esmfold_3B_v1.safetensors")
