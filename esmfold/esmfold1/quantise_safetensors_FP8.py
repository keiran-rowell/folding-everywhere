#!/usr/bin/env python3
"""Quantize ESMFold v1 3B SafeTensors weights to FP8 E4M3 with per-tensor scales."""

import torch
from safetensors.torch import load_file, save_file
import sys

def quantize_esmfold(input_path: str, output_path: str):
    print(f"Loading {input_path}...")
    weights = load_file(input_path)
    quantized_dict = {}

    for name, tensor in weights.items():
        # Keep small vectors (1D biases, LayerNorm weights, embeddings) in FP32
        if tensor.ndim < 2 or "norm" in name or "bias" in name or "embedding" in name:
            quantized_dict[name] = tensor.to(torch.float32)
            continue

        # Large 2D linear weight matrices -> FP8 (E4M3)
        t_f32 = tensor.to(torch.float32)
        max_val = torch.max(torch.abs(t_f32))
        scale = max_val / 448.0  # Max representable in e4m3fn is 448.0
        
        if scale == 0:
            scale = 1.0
            
        scaled_t = t_f32 / scale
        # Clamp & cast to float8_e4m3fn
        clamped_t = torch.clamp(scaled_t, -448.0, 448.0)
        t_fp8 = clamped_t.to(torch.float8_e4m3fn)

        quantized_dict[name] = t_fp8
        quantized_dict[f"{name}._scale"] = scale.unsqueeze(0).to(torch.float32)

    print(f"Saving quantized weights to {output_path}...")
    save_file(quantized_dict, output_path)
    print("Done!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python quantize_fp8.py <input.safetensors> <output_fp8.safetensors>")
        sys.exit(1)
    quantize_esmfold(sys.argv[1], sys.argv[2])
