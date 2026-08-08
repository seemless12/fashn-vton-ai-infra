import sys
sys.path.insert(0, "/opt/fashn-vton")
import os
os.chdir("/opt/fashn-vton")

from fashn_vton import TryOnPipeline
p = TryOnPipeline(weights_dir="/opt/fashn-vton/weights")

# List all public attributes
print("=== PUBLIC ATTRIBUTES ===")
attrs = [a for a in dir(p) if not a.startswith("_")]
for a in attrs:
    obj = getattr(p, a)
    print(f"  {a}: {type(obj).__name__}")

# Check for torch.nn.Module submodules
print("\n=== TORCH MODULES (compilable) ===")
import torch
for name, module in vars(p).items():
    if isinstance(module, torch.nn.Module):
        print(f"  {name}: {type(module).__name__} (params: {sum(p.numel() for p in module.parameters())/1e6:.1f}M)")

# Check for attention processors
print("\n=== ATTENTION CHECK ===")
for name in ["model", "unet", "transformer", "net", "network"]:
    if hasattr(p, name):
        m = getattr(p, name)
        print(f"  Found: p.{name} -> {type(m).__name__}")
        if hasattr(m, "set_attn_processor"):
            print(f"    Has set_attn_processor!")
        if hasattr(m, "enable_xformers_memory_efficient_attention"):
            print(f"    Has enable_xformers_memory_efficient_attention!")
