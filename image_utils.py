"""
image_utils.py
---------------
Preprocessing helpers used by the VTON engine.

Two jobs:
1. Clean up garment product photos (flat-lays / model shots) before they hit
   the try-on pipeline — tight crop + background cleanup reduces noise that
   otherwise gets baked into the diffusion conditioning.
2. Use the FASHN Human Parser to classify a garment as "loose/long" vs
   "fitted/short", which drives the masked-vs-maskless decision in engine.py.
"""

from __future__ import annotations

import io
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

# Garment-relevant classes from the FASHN Human Parser label set
GARMENT_CLASS_IDS = {
    "top": 3,
    "dress": 4,
    "skirt": 5,
    "pants": 6,
}

# Maps a dominant parsed class -> the category string the VTON pipeline expects
CLASS_TO_CATEGORY = {
    "top": "tops",
    "dress": "one-pieces",
    "skirt": "bottoms",
    "pants": "bottoms",
}

# The pipeline only accepts "tops" | "bottoms" | "one-pieces". Callers
# (e.g. notebooks) often send more natural-language labels -- normalize
# those here rather than rejecting the request.
#
# Each entry is (canonical_category, loose_hint):
#   loose_hint = True   -> known loose/long garment (kurta, gown, abaya...):
#                           route straight to maskless, matching the setup
#                           that already works well for these.
#   loose_hint = False  -> known fitted/short garment (t-shirt, crop top...):
#                           route straight to masked, to constrain
#                           hallucination on tight-fitting silhouettes.
#   loose_hint = None   -> ambiguous length for this name alone (e.g. plain
#                           "shirt" or "dress" could be either) -- fall back
#                           to the parser-based extent-ratio heuristic.
#
# This is intentionally the primary signal: the caller's own category name
# is far more reliable than inferring looseness from a garment photo, and
# using it also skips an extra Human Parser forward pass when the hint is
# conclusive, which saves real wall-clock time on slow GPUs.
CATEGORY_ALIASES = {
    "tops": ("tops", None),
    "top": ("tops", None),
    "t-shirt": ("tops", False),
    "tshirt": ("tops", False),
    "shirt": ("tops", None),
    "blouse": ("tops", False),
    "crop top": ("tops", False),
    "croptop": ("tops", False),
    "tank top": ("tops", False),
    "polo": ("tops", False),
    "kurta": ("tops", True),
    "kurti": ("tops", True),
    "sherwani": ("tops", True),
    "jubba": ("tops", True),
    "tunic": ("tops", True),
    "sweater": ("tops", None),
    "hoodie": ("tops", None),
    "jacket": ("tops", None),
    "bottoms": ("bottoms", None),
    "bottom": ("bottoms", None),
    "pants": ("bottoms", None),
    "jeans": ("bottoms", None),
    "trousers": ("bottoms", None),
    "skirt": ("bottoms", None),
    "shorts": ("bottoms", False),
    "one-pieces": ("one-pieces", None),
    "one-piece": ("one-pieces", None),
    "dress": ("one-pieces", None),
    "full dress": ("one-pieces", True),
    "full-dress": ("one-pieces", True),
    "full outfit": ("one-pieces", True),
    "kurta pajama": ("one-pieces", True),
    "kurta pajam": ("one-pieces", True),
    "kurta-pajama": ("one-pieces", True),
    "kurta-pajam": ("one-pieces", True),
    "kurta shalwar": ("one-pieces", True),
    "kurta-shalwar": ("one-pieces", True),
    "shalwar kameez": ("one-pieces", True),
    "kameez shalwar": ("one-pieces", True),
    "shalwar-kameez": ("one-pieces", True),
    "kameez-shalwar": ("one-pieces", True),
    "suit": ("one-pieces", True),
    "eastern wear": ("one-pieces", True),
    "2pc": ("one-pieces", True),
    "2-pc": ("one-pieces", True),
    "2 piece": ("one-pieces", True),
    "mini dress": ("one-pieces", False),
    "bodycon dress": ("one-pieces", False),
    "gown": ("one-pieces", True),
    "maxi dress": ("one-pieces", True),
    "frock": ("one-pieces", None),
    "burkha": ("one-pieces", True),
    "burqa": ("one-pieces", True),
    "abaya": ("one-pieces", True),
    "caftan": ("one-pieces", True),
    "jumpsuit": ("one-pieces", None),
    "romper": ("one-pieces", False),
}


def normalize_category(raw: Optional[str]) -> Tuple[Optional[str], Optional[bool]]:
    """
    Map a freeform category string (e.g. 't-shirt', 'kurta', 'burkha') to
    (canonical_category, loose_hint):
      canonical_category -- the value the pipeline's `category` param
        expects ("tops"/"bottoms"/"one-pieces"), or None if unrecognized
        (triggering auto-detection upstream).
      loose_hint -- True/False if this specific garment name conclusively
        implies loose or fitted, else None (ambiguous -> use the parser
        heuristic instead).
    """
    if not raw:
        return None, None
    key = raw.strip().lower()
    return CATEGORY_ALIASES.get(key, (None, None))


def to_pil(image) -> Image.Image:
    """Accept bytes, file path, or PIL.Image and always return a PIL RGB image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("RGB")
    # assume path-like
    return Image.open(image).convert("RGB")


def autocrop_garment(image: Image.Image, pad_frac: float = 0.04) -> Image.Image:
    """
    Tight-crop a garment product photo to its content bounding box.

    Most flat-lay / ghost-mannequin shots sit on a near-uniform light
    background. Large empty margins waste patch budget and, in our testing,
    correlate with the model over-extending garment volume to "fill" the
    frame. This crops to content with a small safety pad.

    Falls back to the original image untouched if no clear background can
    be found (e.g. busy background, non-flatlay editorial shot).
    """
    arr = np.array(image.convert("L"))
    h, w = arr.shape

    # Sample the four corners to guess the background color/brightness.
    corner_px = np.concatenate(
        [
            arr[:5, :5].ravel(),
            arr[:5, -5:].ravel(),
            arr[-5:, :5].ravel(),
            arr[-5:, -5:].ravel(),
        ]
    )
    bg_level = np.median(corner_px)
    bg_std = np.std(corner_px)

    # Background must be reasonably uniform for this heuristic to be safe.
    if bg_std > 18:
        return image

    diff = np.abs(arr.astype(np.int16) - int(bg_level))
    mask = diff > max(15, bg_std * 3)

    if mask.sum() < 0.01 * mask.size:
        # Nothing detected as foreground -- bail out safely.
        return image

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    pad_y = int((y1 - y0) * pad_frac)
    pad_x = int((x1 - x0) * pad_frac)
    y0 = max(0, y0 - pad_y)
    y1 = min(h, y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(w, x1 + pad_x)

    return image.crop((x0, y0, x1, y1))


def normalize_orientation(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation so downstream pose/parsing isn't fed a rotated image."""
    return ImageOps.exif_transpose(image)


def classify_garment(
    person_or_model_image: Image.Image,
    parser,
) -> Tuple[str, str, float]:
    """
    Run the FASHN Human Parser over an image that shows the garment being
    worn (either the person-image for a same-garment sanity check, or a
    model-worn garment reference photo) and return:

        (dominant_class, mapped_category, vertical_extent_ratio)

    vertical_extent_ratio is the fraction of image height the dominant
    garment class spans -- a cheap proxy for "loose/long" (kurta, gown,
    maxi dress) vs "fitted/short" (t-shirt, mini/knee dress).

    If nothing garment-like is detected, defaults to ("top", "tops", 0.3).
    """
    seg = parser.predict(person_or_model_image)
    h = seg.shape[0]

    best_class, best_count = "top", 0
    for name, cls_id in GARMENT_CLASS_IDS.items():
        count = int((seg == cls_id).sum())
        if count > best_count:
            best_class, best_count = name, count

    if best_count == 0:
        return "top", "tops", 0.3

    mask = seg == GARMENT_CLASS_IDS[best_class]
    rows_with_garment = np.where(mask.any(axis=1))[0]
    extent_ratio = float((rows_with_garment.max() - rows_with_garment.min()) / h)

    return best_class, CLASS_TO_CATEGORY[best_class], extent_ratio


def laplacian_sharpness(image: Image.Image) -> float:
    """
    Cheap, dependency-light sharpness score (variance of a 3x3 Laplacian)
    used to penalize blurry/hallucinated garment regions when auto-scoring
    two candidate outputs against each other. No cv2 dependency required.
    """
    arr = np.array(image.convert("L"), dtype=np.float32)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

    # Manual 3x3 convolution (valid mode) -- avoids pulling in scipy/cv2.
    h, w = arr.shape
    out = (
        arr[0:h-2, 1:w-1] * kernel[0, 1]
        + arr[1:h-1, 0:w-2] * kernel[1, 0]
        + arr[1:h-1, 1:w-1] * kernel[1, 1]
        + arr[1:h-1, 2:w] * kernel[1, 2]
        + arr[2:h, 1:w-1] * kernel[2, 1]
    )
    return float(out.var())
