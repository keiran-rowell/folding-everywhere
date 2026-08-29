import gc
import torch
from safetensors.torch import save_file

print("1. Loading esmfold_v1_complete.bin with memory mapping...")
# mmap=True avoids copying entire file into memory before unpickling
state = torch.load("weights/esmfold_v1_complete.bin", map_location="cpu", weights_only=False, mmap=True)

print("2. Converting in-place to bfloat16 (halves memory footprint)...")
for k in list(state.keys()):
    v = state[k]
    if isinstance(v, torch.Tensor):
        # Convert in-place and replace in dictionary
        state[k] = v.to(torch.bfloat16).contiguous()
    else:
        del state[k]

gc.collect()

print("3. Saving to safetensors (~4.4 GB)...")
save_file(state, "weights/esmfold_v1_complete.safetensors")
print("Done! Saved weights/esmfold_v1_complete.safetensors")
