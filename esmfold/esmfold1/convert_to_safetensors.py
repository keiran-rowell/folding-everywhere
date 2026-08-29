import torch
from safetensors.torch import save_file

print("Loading local esmfold_v1_complete.bin...")
state = torch.load("weights/esmfold_v1_complete.bin", map_location="cpu", weights_only=False)

# Standardize to fp32 / contiguous
clean_state = {k: v.contiguous().float() for k, v in state.items() if isinstance(v, torch.Tensor)}

save_file(clean_state, "weights/esmfold_v1_complete.safetensors")
print(f"Saved {len(clean_state)} tensors to weights/esmfold_v1_complete.safetensors")
