from __future__ import annotations

import numpy as np

from .hg2 import HG2_SAFE_MAX_HEIGHT, HG2Map


def apply_vertical_scale(terrain: HG2Map, scale: float) -> HG2Map:
    """Compress or expand vertical relief without changing horizontal topology.

    The transformation is anchored at the median terrain height so highlands and
    lowlands move toward or away from the same central elevation. Exact flat
    regions remain flat, while every vertical difference is scaled uniformly.

    ``scale=1.0`` preserves the generated terrain, ``0.75`` keeps 75% of its
    vertical contrast, and values above 1.0 exaggerate relief.
    """
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("vertical_scale must be a finite value greater than 0")
    if abs(scale - 1.0) < 1e-9:
        return terrain

    source = terrain.heights.astype(np.float32)
    anchor = float(np.median(source))
    scaled = anchor + (source - anchor) * scale
    scaled = np.clip(np.rint(scaled), 0, HG2_SAFE_MAX_HEIGHT).astype(np.uint16)
    return HG2Map(
        scaled,
        terrain.zones_x,
        terrain.zones_z,
        terrain.zone_bits,
        terrain.structure_version,
        terrain.map_version,
    )
