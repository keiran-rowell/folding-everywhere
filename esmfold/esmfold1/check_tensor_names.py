from safetensors import safe_open

with safe_open("weights/esmfold_3B_v1.safetensors", framework="pt", device="cpu") as f:
    keys = list(f.keys())

# Search for embedding keys
emb_keys = [k for k in keys if "embed" in k.lower()]
print("Embedding keys found:", emb_keys)

# Sample the first 15 keys to see naming scheme
print("\nSample keys:", keys[:15])
