"""
Detailed per-stage profiling for the FASHN VTON pipeline.
Measures: preprocessing, DWPose, Human Parser, Diffusion, VAE decode, postprocessing.
"""
import os
import sys
import time
import json

# Ensure we're running from the right directory
os.chdir("/opt/fashn-vton")
sys.path.insert(0, "/opt/fashn-vton")

import torch
from PIL import Image
from engine import VTONEngine, TryOnRequest

def profile_detailed():
    results = {}

    # --- Model Loading ---
    print("=" * 60)
    print("STAGE 0: Model Loading")
    print("=" * 60)
    t0 = time.time()
    engine = VTONEngine(weights_dir="./weights")
    t1 = time.time()
    results["model_load_time"] = round(t1 - t0, 2)
    print(f"  Model load time: {results['model_load_time']}s")

    # Check VRAM after model load
    vram_after_load = torch.cuda.memory_allocated() / (1024**3)
    vram_reserved = torch.cuda.memory_reserved() / (1024**3)
    results["vram_after_load_gb"] = round(vram_after_load, 2)
    results["vram_reserved_gb"] = round(vram_reserved, 2)
    print(f"  VRAM Allocated: {vram_after_load:.2f} GB")
    print(f"  VRAM Reserved:  {vram_reserved:.2f} GB")

    # --- Create test inputs ---
    person = Image.new('RGB', (768, 1024), color='white')
    garment = Image.new('RGB', (768, 1024), color='red')

    # --- Warmup pass (torch.compile needs this) ---
    print("\n" + "=" * 60)
    print("WARMUP PASS (torch.compile graph capture)")
    print("=" * 60)
    req = TryOnRequest(
        person_image=person,
        garment_image=garment,
        category="tops",
        mode="maskless",
        num_timesteps=15
    )
    torch.cuda.synchronize()
    t0 = time.time()
    _ = engine.run(req)
    torch.cuda.synchronize()
    t1 = time.time()
    results["warmup_time"] = round(t1 - t0, 2)
    print(f"  Warmup time: {results['warmup_time']}s")

    # --- Timed runs at different step counts ---
    step_counts = [12, 15, 20, 25, 30]
    for steps in step_counts:
        print(f"\n{'=' * 60}")
        print(f"BENCHMARK: {steps} steps")
        print("=" * 60)

        req = TryOnRequest(
            person_image=person,
            garment_image=garment,
            category="tops",
            mode="maskless",
            num_timesteps=steps
        )

        # Run 3 times and average
        times = []
        for run in range(3):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            result = engine.run(req)
            torch.cuda.synchronize()
            t1 = time.time()
            elapsed = t1 - t0
            times.append(elapsed)
            print(f"  Run {run+1}: {elapsed:.2f}s")

        avg = sum(times) / len(times)
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        results[f"steps_{steps}_avg"] = round(avg, 2)
        results[f"steps_{steps}_peak_vram_gb"] = round(peak_vram, 2)
        print(f"  Average: {avg:.2f}s")
        print(f"  Peak VRAM: {peak_vram:.2f} GB")

    # --- Check xformers ---
    print(f"\n{'=' * 60}")
    print("ENVIRONMENT CHECK")
    print("=" * 60)
    try:
        import xformers
        results["xformers_version"] = xformers.__version__
        print(f"  xformers: {xformers.__version__}")
    except ImportError:
        results["xformers_version"] = "NOT INSTALLED"
        print("  xformers: NOT INSTALLED")

    results["pytorch_version"] = torch.__version__
    results["cuda_version"] = torch.version.cuda
    results["gpu_name"] = torch.cuda.get_device_name(0)
    results["compute_capability"] = str(torch.cuda.get_device_capability(0))
    results["is_ampere_plus"] = engine.is_ampere_plus
    results["torch_compile_enabled"] = hasattr(engine.pipeline, 'transformer') and hasattr(engine.pipeline.transformer, '_torchdynamo_orig_callable')

    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"  Ampere+: {engine.is_ampere_plus}")

    # --- Final summary ---
    print(f"\n{'=' * 60}")
    print("SUMMARY (JSON)")
    print("=" * 60)
    print(json.dumps(results, indent=2))

    return results

if __name__ == "__main__":
    profile_detailed()
