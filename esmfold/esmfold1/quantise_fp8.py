#!/usr/bin/env python3
"""Aggressive FP8 (E4M3) quantization for ESMFold v1 SafeTensors."""

import sys
import torch
from safetensors.torch import load_file, save_file

def quantize_esmfold(input_path: str, output_path: str):
    print(f"Loading {input_path}...")
    weights = load_file(input_path)
    quantized_dict = {}
    
    total_orig_bytes = 0
    total_quant_bytes = 0

    for name, tensor in weights.items():
        orig_bytes = tensor.nelement() * tensor.element_size()
        total_orig_bytes += orig_bytes

        # Keep 1D tensors (biases, LN weights) and small tables in FP32
        # Only quantize 2D weight matrices (ndim == 2)
        if tensor.ndim < 2 or tensor.numel() < 1024:
            quantized_dict[name] = tensor.to(torch.float32)
            total_quant_bytes += tensor.nelement() * 4
            continue

        # All 2D linear weight matrices -> FP8 (E4M3)
        t_f32 = tensor.to(torch.float32)
        max_val = torch.max(torch.abs(t_f32))
        scale = max_val / 448.0  # Max value for e4m3fn
        
        if scale == 0:
            scale = 1.0
            
        scaled_t = t_f32 / scale
        clamped_t = torch.clamp(scaled_t, -448.0, 448.0)
        t_fp8 = clamped_t.to(torch.float8_e4m3fn)

        quantized_dict[name] = t_fp8
        quantized_dict[f"{name}._scale"] = scale.unsqueeze(0).to(torch.float32)
        
        # 1 byte per FP8 element + 4 bytes for scale
        total_quant_bytes += tensor.nelement() * 1 + 4

    print(f"Original size:  {total_orig_bytes / (1024**3):.2f} GB")
    print(f"Quantized size: {total_quant_bytes / (1024**3):.2f} GB")
    print(f"Saving to {output_path}...")
    save_file(quantized_dict, output_path)
    print("Done!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python quantize_fp8.py <input.safetensors> <output_fp8.safetensors>")
        sys.exit(1)
    quantize_esmfold(sys.argv[1], sys.argv[2])
