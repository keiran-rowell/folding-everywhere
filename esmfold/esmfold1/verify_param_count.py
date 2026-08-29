import torch

state = torch.load("weights/esmfold_3B_v1.pt", map_location="cpu", weights_only=False)
if "model" in state:
    state = state["model"]

total_params = sum(v.numel() for v in state.values() if isinstance(v, torch.Tensor))
print(f"Total parameters: {total_params / 1e6:.2f}M")
print("Top-level keys in state dict:", [k.split('.')[0] for k in list(state.keys())[:20]])
