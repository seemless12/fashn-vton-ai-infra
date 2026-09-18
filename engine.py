"""
engine.py
---------
Wraps FASHN VTON v1.5 + the FASHN Human Parser and adds an adaptive layer
on top to address the fitted-garment (t-shirt / western dress) hallucination
you're seeing, without touching model weights.

Core idea
=========
`segmentation_free=True` (maskless) is v1.5's default and is what your
kurta/long-garment results are using successfully. For LOOSE garments this
is great: volume is under-constrained, but slack fabric hides small body-
shape estimation errors.

For FITTED garments (t-shirts, short/western dresses) that same freedom is
exactly what lets the model hallucinate: wrong volume, warped logos,
invented wrinkles. Masked inference (`segmentation_free=False`) constrains
generation to a segmentation-derived region, trading some of the flexibility
maskless mode is designed to give you for much tighter fidelity -- which is
what fitted garments need most.

Strategy implemented here (mode="auto", the default):
  1. Parse the garment reference (model-worn photo) with the Human Parser.
  2. Classify it as loose/long or fitted/short via vertical extent ratio.
  3. Loose  -> segmentation_free=True  (matches your working kurta setup)
     Fitted -> segmentation_free=False (constrained, reduces hallucination)
  4. For borderline cases, optionally run BOTH and auto-pick the sharper /
     more plausible result via a cheap heuristic scorer (no extra models).

Everything is override-able per-request so you can tune thresholds against
your own eval set once you're on a GPU.
"""

from __future__ import annotations

import logging
import os
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

# Below this vertical-extent ratio, a garment is considered "fitted/short"
# (t-shirt, mini/knee-length western dress) and routed to masked inference.
# Above it, it's considered "loose/long" (kurta, maxi dress, gown) and
# routed to maskless inference. Tune this once you have real outputs to look at.
LOOSE_EXTENT_THRESHOLD = 0.55

# Garment classes we treat as "risky" enough to consider a dual-run when
# mode="auto" and the classification lands close to the threshold.
BORDERLINE_BAND = 0.08


@dataclass
class TryOnRequest:
    person_image: Image.Image
    garment_image: Image.Image
    category: Optional[str] = None          # "tops" | "bottoms" | "one-pieces" | None (auto)
    garment_photo_type: str = "model"        # "model" | "flat-lay"
    mode: str = "auto"                       # "auto" | "masked" | "maskless" | "both"
    num_samples: int = 1
    steps: int = 15
    num_timesteps: Optional[int] = None
    guidance_scale: float = 1.5
    seed: int = 42
    autocrop: bool = True
    # Optional free, local detail-refine pass (see refiner.py). Adds real
    # wall-clock time (model swap + its own generation steps) since it
    # can't run alongside FASHN in VRAM at the same time on a single T4 --
    # this trades time for texture fidelity, not money.
    refine: bool = False
    refine_strength: float = 0.25
    refine_guidance_scale: float = 3.0
    refine_steps: int = 25

    def __post_init__(self):
        if self.num_timesteps is not None:
            self.steps = self.num_timesteps
        else:
            self.num_timesteps = self.steps


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
        # Imports are deferred so this module can be imported (e.g. for tests)
        # without requiring a GPU / the weights to be present.
        import torch
        from fashn_vton import TryOnPipeline
        from fashn_human_parser import FashnHumanParser

        # Standard, safe perf toggles -- no precision/behavior tradeoff on
        # Ampere+ (they're TF32-specific), harmless no-ops on older GPUs.
        # cudnn.benchmark autotunes conv algorithms for fixed input shapes,
        # which this pipeline has (fixed 576x864 output).
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # FASHN VTON's own docs: weights run bf16 on Ampere+ GPUs; on older
        # hardware they convert to full fp32 automatically. fp32 has no
        # tensor-core acceleration on pre-Ampere cards, which is the single
        # biggest cost driver on something like a T4 -- no TF32 either
        # (also Ampere+ only), so the toggles above are no-ops there.
        self.is_ampere_plus = False
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            self.is_ampere_plus = major >= 8
            logger.info(
                "GPU: %s (compute capability %s.%s) -- %s",
                torch.cuda.get_device_name(0),
                major,
                torch.cuda.get_device_capability(0)[1],
                "bf16 tensor cores available"
                if self.is_ampere_plus
                else "pre-Ampere: pipeline will run in fp32 unless FASHN_FORCE_FP16 is set",
            )

        # EXPERIMENTAL, opt-in only: on pre-Ampere GPUs (e.g. T4), attempt
        # fp16 autocast instead of the library's fp32 fallback. T4-class
        # cards DO have native fp16 tensor cores (unlike bf16/TF32), so this
        # can be a real 2-4x win -- but it's not an officially supported
        # path for this pipeline, so treat it as unverified until you've
        # compared output quality against a known-good fp32 run. Using
        # autocast (rather than hard-casting model weights) is the safer
        # of the two approaches: it doesn't require knowing/guessing the
        # pipeline's internal attribute names, and it lets PyTorch decide
        # per-op which precision is numerically safe.
        self.force_fp16 = (
            not self.is_ampere_plus
            and os.environ.get("FASHN_FORCE_FP16", "0") == "1"
        )
        if self.force_fp16:
            logger.warning(
                "FASHN_FORCE_FP16=1: running inference under fp16 autocast. "
                "This is experimental and unverified against this pipeline "
                "-- compare a few outputs against fp32 before trusting it."
            )

        logger.info("Loading FASHN VTON pipeline from %s ...", weights_dir)
        self._weights_dir = weights_dir
        self._device = device
        self.pipeline = TryOnPipeline(weights_dir=weights_dir, device=device)
        
        # Performance Optimizations: Xformers / SDPA and torch.compile
        try:
            self.pipeline.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory efficient attention.")
        except Exception as e:
            logger.warning(f"Could not enable xformers: {e}")
            
        try:
            if hasattr(self.pipeline, "transformer"):
                self.pipeline.transformer = torch.compile(self.pipeline.transformer, mode="reduce-overhead", fullgraph=True)
                logger.info("Enabled torch.compile on the transformer.")
        except Exception as e:
            logger.warning(f"Could not torch.compile the transformer: {e}")

        logger.info("Loading FASHN Human Parser ...")
        self.parser = FashnHumanParser()
        # Lazily loaded, NOT loaded here -- see _call_refiner(). Loading it
        # eagerly at startup would hold VRAM that FASHN needs, for a
        # feature most requests won't use.
        self.refiner = DetailRefiner(device=device or "cuda")
        logger.info("Engine ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, req: TryOnRequest) -> TryOnResult:
        person = normalize_orientation(to_pil(req.person_image))
        garment = normalize_orientation(to_pil(req.garment_image))

        if req.autocrop and req.garment_photo_type == "flat-lay":
            garment = autocrop_garment(garment)

        canonical_category, loose_hint = normalize_category(req.category)

        # Only pay for a Human Parser forward pass when we actually need it:
        # either the category itself is unrecognized (need auto_category),
        # or we're in "auto" mode and the category name alone doesn't tell
        # us loose vs fitted.
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
            # The caller's own category name conclusively tells us which
            # mode to use (e.g. "kurta" -> loose -> maskless) -- trust it
            # over the parser heuristic, and skip the dual-run entirely.
            plan = [loose_hint]
        else:
            loose = extent_ratio >= LOOSE_EXTENT_THRESHOLD
            near_boundary = abs(extent_ratio - LOOSE_EXTENT_THRESHOLD) <= BORDERLINE_BAND
            if near_boundary:
                plan = [True, False]  # dual-run and let the scorer decide
            else:
                plan = [loose]  # True = maskless, False = masked

        candidates = []
        for seg_free in plan:
            result = self._call_pipeline(
                person_image=person,
                garment_image=garment,
                category=category,
                garment_photo_type=req.garment_photo_type,
                num_samples=req.num_samples,
                num_timesteps=req.steps,
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
                # Refining is a bonus pass, not a required one -- if it
                # breaks (OOM, download failure, etc.) fall back to FASHN's
                # own output rather than failing the whole job.
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
        """
        Runs the optional detail-refine pass. FASHN's pipeline and the
        refiner don't both fit in VRAM on a single 16GB T4, so this frees
        FASHN's GPU memory first, runs the refiner, then reloads FASHN
        before returning -- always, even if refining fails, via try/finally,
        so the server is never left without a usable pipeline for the next
        job. Freeing FASHN uses only `del` + gc + empty_cache on the whole
        pipeline object (its documented public API), never guessing at
        internal module attribute names.
        """
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
        Runs one pipeline call with the safe perf context always applied
        (inference_mode -- no autograd bookkeeping, real memory + speed
        win, zero correctness risk), plus the experimental fp16 autocast
        path if explicitly enabled via FASHN_FORCE_FP16=1 on a pre-Ampere
        GPU. Frees the CUDA allocator cache afterward so a long-running
        server doesn't accumulate fragmentation across many jobs.
        """
        import torch

        try:
            with torch.inference_mode():
                if self.force_fp16:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        return self.pipeline(**kwargs)
                return self.pipeline(**kwargs)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
