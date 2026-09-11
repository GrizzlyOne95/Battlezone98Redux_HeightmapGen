from __future__ import annotations

import zlib

import numpy as np

from .hg2 import HG2Map, HG2_SAFE_MAX_HEIGHT
from .settings import GeneratorSettings
from .stock_detail import STOCK_DETAIL_STYLES
from .stock_detail_v2 import PROFILES, _clustered_micro_features, repair_with_saddles
from .stock_detail_v3 import (
    POST_SADDLES,
    _calm_traversable_substrate,
    _extra_discrete_events,
    _patchy_micro_surface,
)


def enhance_stock_terrain(terrain: HG2Map, settings: GeneratorSettings, style: str) -> HG2Map:
    profile = PROFILES.get(style)
    if profile is None:
        return terrain

    seed = (int(settings.seed) ^ (zlib.crc32(style.encode("utf-8")) & 0xFFFFFFFF) ^ 0x6D4B2A19) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    a = _calm_traversable_substrate(np.asarray(terrain.heights, dtype=np.float32), style)

    a, protected = repair_with_saddles(
        a,
        rng,
        target_map_fraction=profile.target_map_fraction,
        max_saddles=profile.max_saddles,
    )
    _clustered_micro_features(a, protected, profile, settings, rng)
    _extra_discrete_events(a, protected, style, settings, rng)
    _patchy_micro_surface(a, protected, style, settings, rng)

    final_saddles = POST_SADDLES.get(style, 0)
    if final_saddles:
        a, _ = repair_with_saddles(a, rng, profile.target_map_fraction, final_saddles)

    heights = np.clip(np.rint(a), 0, HG2_SAFE_MAX_HEIGHT).astype(np.uint16)
    return HG2Map(
        heights,
        terrain.zones_x,
        terrain.zones_z,
        terrain.zone_bits,
        terrain.structure_version,
        terrain.map_version,
    )


__all__ = ["STOCK_DETAIL_STYLES", "enhance_stock_terrain"]
