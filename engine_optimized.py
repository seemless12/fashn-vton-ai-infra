"""
engine_optimized.py
-------------------
Optimized FASHN VTON engine with all critical performance fixes applied.

Fixes applied:
1. torch.compile targets `tryon_model` (not `transformer`)
2. SDPA attention (native PyTorch 2.x) instead of broken xformers call
3. Removed empty_cache() to preserve CUDA allocator cache & enable CUDA Graphs
4. Warmup pass at init time to pre-compile graphs
5. Channels Last memory format for Tensor Core efficiency
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from image_utils import (
    autocrop_garment,
    classify_garment,
    laplacian_sharpness,
    normalize_category,
    normalize_orientation,
    to_pil,
)
from refiner import DetailRefiner, build_garment_mask

logger = logging.getLogger("vton_engine")

LOOSE_EXTENT_THRESHOLD = 0.55
BORDERLINE_BAND = 0.08


@dataclass
class TryOnRequest:
    person_image: Image.Image
    garment_image: Image.Image
    category: Optional[str] = None
    garment_photo_type: str = "model"
    mode: str = "auto"
    num_samples: int = 1
    num_timesteps: int = 30
    steps: Optional[int] = None
    guidance_scale: float = 1.5
    seed: int = 42
    autocrop: bool = True
    refine: bool = False
    refine_strength: float = 0.25
    refine_guidance_scale: float = 3.0
    refine_steps: int = 25

    def __post_init__(self):
        if self.steps is not None:
            self.num_timesteps = self.steps
        else:
            self.steps = self.num_timesteps


@dataclass
class TryOnResult:
    image: Image.Image
    category_used: str
    segmentation_free_used: bool
    extent_ratio: float
    dominant_class: str
    candidates_tried: int
    score: Optional[float] = None
    refined: bool = False


class VTONEngine:
    def __init__(self, weights_dir: str = "./weights", device: Optional[str] = None):
        import torch
        from fashn_vton import TryOnPipeline
        from fashn_human_parser import FashnHumanParser

        # --- Standard perf toggles ---
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        self.is_ampere_plus = False
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            self.is_ampere_plus = major >= 8
            logger.info(
                "GPU: %s (compute capability %s.%s) -- %s",
                torch.cuda.get_device_name(0),
                major, minor,
                "Ampere+ bf16 tensor cores available" if self.is_ampere_plus else "pre-Ampere (Turing T4 / sm_75)",
            )

        # --- Enable SDPA (Scaled Dot Product Attention) ---
        # FlashAttention strictly requires Ampere+ (sm_80+).
        # On Tesla T4 (Turing, sm_75), Flash is unsupported and MUST be False.
        # Memory-Efficient Attention (Cutlass) is supported on sm_75.
        # Math Attention MUST be True so PyTorch can fall back whenever Cutlass constraints are not met!
        torch.backends.cuda.enable_flash_sdp(self.is_ampere_plus)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        logger.info(
            "Configured PyTorch SDPA: Flash=%s, MemEfficient=True, Math=True",
            self.is_ampere_plus,
        )

        # Default force_fp16 to True on pre-Ampere GPUs (like T4) for memory & speed, or via env
        self.force_fp16 = (
            os.environ.get("FASHN_FORCE_FP16", "1" if not self.is_ampere_plus else "0") == "1"
        )
        if self.force_fp16:
            logger.info("FP16 autocast enabled (optimal for Tesla T4 Tensor Cores)")

        logger.info("Loading FASHN VTON pipeline from %s ...", weights_dir)
        self._weights_dir = weights_dir
        self._device = device
        self.pipeline = TryOnPipeline(weights_dir=weights_dir, device=device)

        # --- FIX #2: Apply Channels Last memory format ---
        # This improves Tensor Core utilization on modern NVIDIA GPUs
        if hasattr(self.pipeline, "tryon_model"):
            try:
                self.pipeline.tryon_model = self.pipeline.tryon_model.to(
                    memory_format=torch.channels_last
                )
                logger.info("Applied channels_last memory format to tryon_model")
            except Exception as e:
                logger.warning(f"Could not apply channels_last: {e}")

        # --- FIX #3: torch.compile ---
        # Disabled by default on pre-Ampere (T4) to avoid CUDA Graph conflicts and slow compilation
        enable_compile = os.environ.get("FASHN_ENABLE_COMPILE", "0") == "1"
        if enable_compile and hasattr(self.pipeline, "tryon_model"):
            try:
                compile_mode = "default" if not self.is_ampere_plus else "reduce-overhead"
                self.pipeline.tryon_model = torch.compile(
                    self.pipeline.tryon_model,
                    mode=compile_mode,
                    fullgraph=False
                )
                logger.info(f"torch.compile enabled on tryon_model (mode={compile_mode})")
            except Exception as e:
                logger.warning(f"Could not torch.compile tryon_model: {e}")
        else:
            logger.info("torch.compile disabled (optimal for stable eager SDPA inference)")

        logger.info("Loading FASHN Human Parser ...")
        self.parser = FashnHumanParser()
        self.refiner = DetailRefiner(device=device or "cuda")

        # --- FIX #5: Warmup pass at init ---
        logger.info("Running warmup inference pass...")
        warmup_start = time.time()
        try:
            dummy_person = Image.new('RGB', (768, 1024), color='white')
            dummy_garment = Image.new('RGB', (768, 1024), color='red')
            self._call_pipeline(
                person_image=dummy_person,
                garment_image=dummy_garment,
                category="tops",
                garment_photo_type="model",
                num_samples=1,
                num_timesteps=5,  # Minimal steps for warmup
                guidance_scale=1.5,
                seed=42,
                segmentation_free=True,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logger.info(f"Warmup completed in {time.time() - warmup_start:.1f}s. Engine ready.")
        except Exception as e:
            logger.warning(f"Warmup pass failed (non-critical): {e}")

        logger.info("Engine ready (optimized).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, req: TryOnRequest) -> TryOnResult:
        person = normalize_orientation(to_pil(req.person_image))
        garment = normalize_orientation(to_pil(req.garment_image))

        if req.autocrop and req.garment_photo_type == "flat-lay":
            garment = autocrop_garment(garment)

        canonical_category, loose_hint = normalize_category(req.category)

        need_parser = (canonical_category is None) or (
            loose_hint is None and req.mode == "auto"
        )
        if need_parser:
            dominant_class, auto_category, extent_ratio = classify_garment(
                garment if req.garment_photo_type == "model" else person,
                self.parser,
            )
        else:
            dominant_class, auto_category, extent_ratio = "skipped (category hint used)", None, 0.0

        category = canonical_category or auto_category

        if req.mode == "masked":
            plan = [False]
        elif req.mode == "maskless":
            plan = [True]
        elif req.mode == "both":
            plan = [True, False]
        elif loose_hint is not None:
            plan = [loose_hint]
        else:
            loose = extent_ratio >= LOOSE_EXTENT_THRESHOLD
            plan = [loose]

        candidates = []
        for seg_free in plan:
            result = self._call_pipeline(
                person_image=person,
                garment_image=garment,
                category=category,
                garment_photo_type=req.garment_photo_type,
                num_samples=req.num_samples,
                num_timesteps=req.num_timesteps,
                guidance_scale=req.guidance_scale,
                seed=req.seed,
                segmentation_free=seg_free,
            )
            out_img = result.images[0]
            score = laplacian_sharpness(out_img)
            candidates.append((out_img, seg_free, score))
            logger.info(
                "candidate: segmentation_free=%s sharpness=%.2f", seg_free, score
            )

        best_img, best_seg_free, best_score = max(candidates, key=lambda c: c[2])
        refined = False

        if req.refine:
            try:
                best_img = self._call_refiner(
                    base_image=best_img,
                    garment_reference_image=garment,
                    dominant_class=dominant_class if dominant_class in ("top", "dress", "skirt", "pants") else None,
                    strength=req.refine_strength,
                    guidance_scale=req.refine_guidance_scale,
                    num_inference_steps=req.refine_steps,
                    seed=req.seed,
                )
                refined = True
            except Exception:
                logger.exception(
                    "Detail refine step failed; returning un-refined FASHN output instead."
                )

        return TryOnResult(
            image=best_img,
            category_used=category,
            segmentation_free_used=best_seg_free,
            extent_ratio=extent_ratio,
            dominant_class=dominant_class,
            candidates_tried=len(candidates),
            score=best_score,
            refined=refined,
        )

    def _call_refiner(
        self,
        base_image: Image,
        garment_reference_image: Image,
        dominant_class: Optional[str],
        strength: float,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int,
    ) -> Image:
        import gc
        import torch

        logger.info("Refine requested: releasing FASHN VRAM to load the detail refiner...")
        del self.pipeline
        self.pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            mask_image = build_garment_mask(base_image, self.parser, dominant_class)
            self.refiner.load()
            refined_image = self.refiner.refine(
                base_image=base_image,
                mask_image=mask_image,
                garment_reference_image=garment_reference_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
            )
            return refined_image
        finally:
            self.refiner.unload()
            logger.info("Reloading FASHN VTON pipeline after refine step...")
            from fashn_vton import TryOnPipeline
            self.pipeline = TryOnPipeline(weights_dir=self._weights_dir, device=self._device)

    def _call_pipeline(self, **kwargs):
        """
        FIX #4: Removed empty_cache() — it was destroying the CUDA allocator
        cache after every request, adding ~0.5s overhead and breaking CUDA Graphs
        (which torch.compile mode=reduce-overhead relies on).
        """
        import torch

        with torch.inference_mode():
            if self.force_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    return self.pipeline(**kwargs)
            return self.pipeline(**kwargs)
