"""
refiner.py
----------
Optional second stage: a FREE, local, open-weight detail refiner that
targets ONLY the garment region of a FASHN VTON output, to fix pattern/
texture warping (e.g. the sequin-pattern breakup on fitted garments) that
FASHN's own generation sometimes introduces.

Why this exists
================
FASHN gives you correct fit, pose, and drape. When its output shows warped
or broken-up patterning on the garment, that's a texture-fidelity problem,
not a placement problem. Rather than fight that inside FASHN's own
generation (different failure mode, different fix), we run a second,
narrowly-scoped pass:

  1. Take FASHN's output image as-is (structure/placement already correct).
  2. Mask ONLY the garment region (via the Human Parser we already run --
     zero extra cost to get this mask).
  3. Inpaint just that region with Stable Diffusion 1.5 inpainting +
     IP-Adapter, using the ORIGINAL garment photo as a visual reference
     (IP-Adapter conditions on a reference image, not just a text prompt --
     this is what lets it "copy" the real pattern/texture rather than
     inventing a generic one).
  4. LOW denoising strength (~0.3-0.4): enough to redraw texture detail,
     not enough to redraw the garment's shape or touch anything outside
     the mask -- so this cannot reintroduce identity/body distortion.

All models here are free, open-weight, and run locally:
  - runwayml/stable-diffusion-inpainting (CreativeML Open RAIL-M license)
  - h94/IP-Adapter (Apache 2.0)
No API key, no per-call cost.

The real cost is VRAM + time, not money
========================================
FASHN alone uses most of a 16GB T4's VRAM. This refiner (SD1.5 + IP-Adapter,
~4-5GB in fp16) does NOT fit alongside FASHN at the same time on that
hardware. So this module is designed to be loaded AFTER releasing FASHN's
GPU memory, and unloaded again before FASHN reloads for the next job --
see VTONEngine._call_refiner() in engine.py, which handles that swap.
This means using the refiner adds real wall-clock time per request (model
load/unload + its own generation steps) on top of FASHN's own generation
time. That's the honest tradeoff for "free" here: time, not dollars.
"""

from __future__ import annotations

import logging
from typing import Optional

from PIL import Image, ImageFilter

logger = logging.getLogger("vton_refiner")

# Garment classes from the FASHN Human Parser label set (see image_utils.py)
GARMENT_CLASS_IDS = {"top": 3, "dress": 4, "skirt": 5, "pants": 6}


def build_garment_mask(output_image: Image.Image, parser, dominant_class: Optional[str] = None) -> Image.Image:
    """
    Runs the Human Parser on FASHN's OUTPUT image (not the reference garment
    photo -- we need the mask where the garment actually landed after
    generation) and returns a white-on-black PIL mask of the garment region,
    slightly feathered so the inpaint blends rather than showing a hard edge.
    """
    import numpy as np

    seg = parser.predict(output_image)

    if dominant_class and dominant_class in GARMENT_CLASS_IDS:
        class_id = GARMENT_CLASS_IDS[dominant_class]
    else:
        # Fall back to whichever garment class has the most pixels.
        counts = {name: int((seg == cid).sum()) for name, cid in GARMENT_CLASS_IDS.items()}
        class_id = GARMENT_CLASS_IDS[max(counts, key=counts.get)]

    mask_arr = (seg == class_id).astype("uint8") * 255
    mask_img = Image.fromarray(mask_arr, mode="L")
    # Feather the mask edges so the refined region blends into the rest of
    # the image rather than showing a visible seam.
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=6))
    return mask_img


class DetailRefiner:
    """
    Lazily-loaded SD1.5 inpainting + IP-Adapter pipeline. Call load() right
    before use and unload() right after -- see engine.py for why this can't
    just stay resident alongside FASHN on a single 16GB GPU.
    """

    MODEL_ID = "runwayml/stable-diffusion-inpainting"
    IP_ADAPTER_REPO = "h94/IP-Adapter"
    IP_ADAPTER_WEIGHT = "ip-adapter_sd15.bin"

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.pipe = None

    def load(self):
        import torch
        from diffusers import AutoPipelineForInpainting

        if self.pipe is not None:
            return  # already loaded

        logger.info("Loading detail refiner (SD1.5 inpainting + IP-Adapter)...")
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(self.device)
        self.pipe.load_ip_adapter(
            self.IP_ADAPTER_REPO, subfolder="models", weight_name=self.IP_ADAPTER_WEIGHT
        )
        self.pipe.set_ip_adapter_scale(0.6)
        logger.info("Detail refiner loaded.")

    def unload(self):
        import gc
        import torch

        if self.pipe is None:
            return
        del self.pipe
        self.pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Detail refiner unloaded, VRAM freed.")

    def refine(
        self,
        base_image: Image.Image,
        mask_image: Image.Image,
        garment_reference_image: Image.Image,
        strength: float = 0.25,
        guidance_scale: float = 3.0,
        num_inference_steps: int = 25,
        seed: int = 42,
    ) -> Image.Image:
        """
        base_image: FASHN's output (the try-on image to refine)
        mask_image: white = refine this region, black = leave untouched
                    (from build_garment_mask, scoped to garment pixels only)
        garment_reference_image: the ORIGINAL garment photo, used as an
                    IP-Adapter visual reference so the refiner reproduces
                    the real pattern/texture instead of inventing one
        strength: how much the masked region is allowed to change.
                    LOW on purpose (default 0.25) -- enough to correct
                    texture, not enough to redraw the garment's shape.
        guidance_scale: classifier-free guidance for the refine pass. Kept
                    LOW on purpose (default 3.0, well below SD1.5's usual
                    ~7.5) -- high CFG here reliably over-sharpens and
                    oversaturates, especially on simple/plain garments that
                    don't have much real texture for the model to recover.
                    If a garment looks under-detailed at these defaults,
                    raise this gradually (try 4.0 before jumping to 5+).
        """
        import torch

        if self.pipe is None:
            raise RuntimeError("DetailRefiner.load() must be called before refine().")

        target_size = base_image.size  # (W, H)
        mask_resized = mask_image.resize(target_size)
        garment_ref_resized = garment_reference_image.resize(target_size)

        generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            # Deliberately restrained -- "sharp"/"detailed" wording combined
            # with any real guidance strength tends to invent texture on
            # plain garments rather than just correcting existing warping.
            prompt="accurate clean fabric texture matching the reference, photorealistic clothing",
            negative_prompt="oversaturated, over-sharpened, harsh contrast, artifacts, blurry, distorted pattern",
            image=base_image,
            mask_image=mask_resized,
            ip_adapter_image=garment_ref_resized,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        return result.images[0]
