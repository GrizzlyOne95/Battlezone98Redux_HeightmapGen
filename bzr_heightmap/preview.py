"""Preview rendering for HG2 height and LGT lighting.

Separation of concerns: generation -> contrast -> height display -> lighting -> UI resize.

All preview functions are deterministic and do not mutate source heights.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

from bzr_heightmap.analysis import make_shaded_image
from bzr_heightmap.hg2 import HG2_SAFE_MAX_HEIGHT
from bzr_heightmap.lgt import compute_lgt_lightmap, lgt_to_brightness

# Fixed range for HG2 Height preview: authoring-safe 0..4095 maps to 0..255.
# This makes contrast changes visually comparable across previews.
HG2_HEIGHT_DISPLAY_MIN = 0
HG2_HEIGHT_DISPLAY_MAX = HG2_SAFE_MAX_HEIGHT


def make_hg2_height_image(heights: np.ndarray) -> Image.Image:
    """Raw BZ height representation (fixed 0..4095 mapping).

    Does not percentile-normalize per preview; a flat at 500 looks darker than
    a plateau at 2500, and changing vertical contrast visibly changes the image
    because the underlying heights change, not because the display window does.
    """
    a = np.asarray(heights, dtype=np.float32)
    # Fixed mapping: 0 -> 0, 4095 -> 255
    norm = np.clip((a - HG2_HEIGHT_DISPLAY_MIN) / max(HG2_HEIGHT_DISPLAY_MAX - HG2_HEIGHT_DISPLAY_MIN, 1), 0.0, 1.0)
    img = (norm * 255.0).astype(np.uint8)
    return Image.fromarray(img, mode="L").convert("RGB")


def make_lgt_preview_image(
    heights: np.ndarray,
    zones_x: int | None = None,
    zones_z: int | None = None,
    lgt_zone_size: int = 128,
) -> Image.Image:
    """LGT-style lighting preview derived from heights.

    Uses the same sun (315° az, 45° alt) and ambient floor (25%) as the LGT
    file definition. The returned display image maps ambient-only to 25%
    gray and full sun to white. Upscaled to HG2 dimensions for
    side-by-side alignment if lgt_zone_size is 128.
    """
    lgt = compute_lgt_lightmap(heights, zones_x, zones_z, lgt_zone_size=lgt_zone_size)
    # Both arrays are south-first, so the height and lighting tabs remain
    # spatially aligned. Displaying south at the top also mirrors file order.
    display = np.rint(lgt_to_brightness(lgt) * 255.0).astype(np.uint8)
    img = Image.fromarray(display, mode="L").convert("RGB")
    # If LGT is half-res (128 per zone vs 256 hg), upscale nearest-neighbor to
    # match HG2 dimensions so side-by-side comparison is pixel-aligned.
    h, w = heights.shape
    if img.size != (w, h):
        img = img.resize((w, h), Image.Resampling.NEAREST)
    return img


def make_shaded_preview(heights: np.ndarray, max_size: Tuple[int, int] = (960, 760)) -> Image.Image:
    """Combined elevation + hillshade view (retained from analysis.make_preview)."""
    preview = make_shaded_image(heights)
    preview.thumbnail(max_size, Image.Resampling.LANCZOS)
    return preview


def make_shaded_preview_fullres(heights: np.ndarray) -> Image.Image:
    """Full-resolution shaded preview without thumbnail (for export)."""
    return make_shaded_image(heights)
