import torch
from safetensors.torch import save_file
from transformers import EsmForProteinFolding

print("Loading facebook/esmfold_v1 from Hugging Face...")
model = EsmForProteinFolding.from_pretrained(
    "facebook/esmfold_v1",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)

state = {k: v.contiguous().float() for k, v in model.state_dict().items()}
total_params = sum(v.numel() for v in state.values())

print(f"Total parameters: {total_params / 1e9:.2f}B across {len(state)} tensors")

output_path = "weights/esmfold_3B_v1.safetensors"
save_file(state, output_path)
print(f"Saved complete weights to {output_path}")
