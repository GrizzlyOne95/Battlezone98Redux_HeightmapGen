from __future__ import annotations

import math
import zlib

import numpy as np
from scipy import ndimage

from .hg2 import HG2Map, HG2_SAFE_MAX_HEIGHT
from .noise import fbm
from .settings import GeneratorSettings
from .stock_detail import (
    STOCK_DETAIL_STYLES,
    _add_crater,
    _add_gaussian_feature,
    _add_ledge,
    _add_small_ramp,
    _slope_degrees,
)
from .stock_detail_v2 import (
    PROFILES,
    _clustered_micro_features,
    _component_state,
    _layered_surface_detail,
    _random_centers,
    repair_with_saddles,
)

# Retain the large-scale recipe composition, but suppress the continuous
# mid-frequency procedural relief on the styles where it fragments otherwise
# useful ground. Exact authored flats and hard cliffs are explicitly retained.
MACRO_RESIDUAL_FACTORS = {
    "Terraced Labyrinth": 1.00,
    "Cratered Divide": 0.88,
    "Ravine Network": 0.80,
    "Mountain Basin": 0.52,
    "Radial Badlands": 0.82,
    "Ridged Wastes": 0.46,
    "Serpentine Canyon": 1.00,
    "Natural Badlands": 0.62,
}

# Extra discrete stock-style events beyond the general v2 profile. These are
# intentionally local terrain objects rather than more full-map fBm.
EXTRA_EVENT_COUNTS = {
    "Terraced Labyrinth": (22, 8, 10),
    "Cratered Divide": (34, 22, 16),
    "Ravine Network": (26, 18, 18),
    "Mountain Basin": (34, 22, 20),
    "Radial Badlands": (32, 18, 18),
    "Ridged Wastes": (20, 12, 18),
    "Serpentine Canyon": (28, 12, 12),
    "Natural Badlands": (42, 26, 24),
}

POST_SADDLES = {
    "Terraced Labyrinth": 2,
    "Cratered Divide": 2,
    "Ravine Network": 0,
    "Mountain Basin": 3,
    "Radial Badlands": 2,
    "Ridged Wastes": 3,
    "Serpentine Canyon": 0,
    "Natural Badlands": 3,
}


def _calm_traversable_substrate(a: np.ndarray, style: str) -> np.ndarray:
    factor = float(MACRO_RESIDUAL_FACTORS.get(style, 1.0))
    if factor >= 0.995:
        return a.copy()
    original = np.asarray(a, dtype=np.float32)
    smooth = ndimage.gaussian_filter(original, sigma=7.0, mode="reflect")
    slope = _slope_degrees(original)
    local_max = ndimage.maximum_filter(original, size=5, mode="reflect")
    local_min = ndimage.minimum_filter(original, size=5, mode="reflect")
    exact_flat = (local_max - local_min) <= 1.0

    # Full effect on gentle/moderate ground, fading out before hard cliffs so
    # macro escarpments, ravines, rims and shelf boundaries keep their identity.
    blend = np.clip((27.0 - slope) / 17.0, 0.0, 1.0).astype(np.float32)
    amount = (1.0 - factor) * blend
    out = original * (1.0 - amount) + smooth * amount
    out[exact_flat] = original[exact_flat]
    return out.astype(np.float32)


def _extra_discrete_events(
    a: np.ndarray,
    protected: np.ndarray,
    style: str,
    settings: GeneratorSettings,
    rng: np.random.Generator,
) -> None:
    counts = EXTRA_EVENT_COUNTS[style]
    area_scale = max(0.08, float(a.size) / float(768 * 768))
    density = 0.65 + 0.85 * float(np.clip(settings.feature_density, 0.0, 1.0))
    slope = _slope_degrees(a)
    eligible = slope <= 16.0
    border = max(10, int(min(a.shape) * 0.012))
    eligible[:border] = False
    eligible[-border:] = False
    eligible[:, :border] = False
    eligible[:, -border:] = False
    if np.any(protected):
        eligible &= ~ndimage.binary_dilation(protected, iterations=4)

    shelf_count = int(round(counts[0] * area_scale * density))
    crater_count = int(round(counts[1] * area_scale * density))
    ramp_count = int(round(counts[2] * area_scale * density))

    # Tiny shelves/steps are a strong authored-BZ signature: localized flats
    # offset by roughly 1-6 world metres, with short irregular lips around them.
    for cy, cx in _random_centers(eligible, shelf_count, rng):
        offset = float(rng.choice([-1.0, 1.0]) * rng.uniform(12.0, 62.0))
        rx = float(rng.uniform(3.5, 10.0))
        ry = float(rng.uniform(2.5, 7.0))
        angle = float(rng.uniform(0.0, math.tau))
        _add_ledge(a, cy, cx, rx, ry, offset, angle)
        if rng.random() < 0.38:
            # Break a subset of shelf silhouettes with an overlapping smaller
            # step so these do not all read as isolated geometric ovals.
            _add_ledge(
                a,
                cy + float(rng.uniform(-4.0, 4.0)),
                cx + float(rng.uniform(-4.0, 4.0)),
                rx * float(rng.uniform(0.45, 0.72)),
                ry * float(rng.uniform(0.45, 0.72)),
                offset * float(rng.uniform(0.45, 0.75)),
                angle + float(rng.uniform(-0.7, 0.7)),
            )

    for cy, cx in _random_centers(eligible, crater_count, rng):
        radius = float(rng.uniform(2.4, 8.5))
        _add_crater(
            a,
            cy,
            cx,
            radius,
            float(rng.uniform(10.0, 38.0)),
            float(rng.uniform(4.0, 17.0)),
            float(rng.uniform(0.65, 1.50)),
        )
        if rng.random() < 0.30:
            child_angle = float(rng.uniform(0.0, math.tau))
            child_dist = radius * float(rng.uniform(0.8, 1.6))
            _add_crater(
                a,
                cy + math.sin(child_angle) * child_dist,
                cx + math.cos(child_angle) * child_dist,
                radius * float(rng.uniform(0.35, 0.65)),
                float(rng.uniform(6.0, 20.0)),
                float(rng.uniform(2.0, 9.0)),
                float(rng.uniform(0.75, 1.35)),
            )

    for cy, cx in _random_centers(eligible, ramp_count, rng):
        _add_small_ramp(
            a,
            cy,
            cx,
            float(rng.uniform(8.0, 22.0)),
            float(rng.uniform(2.8, 7.5)),
            float(rng.uniform(12.0, 44.0)),
            float(rng.uniform(0.0, math.tau)),
        )


def _patchy_micro_surface(
    a: np.ndarray,
    protected: np.ndarray,
    style: str,
    settings: GeneratorSettings,
    rng: np.random.Generator,
) -> None:
    detail_strength = float(np.clip(settings.detail, 0.0, 1.5))
    if detail_strength <= 0:
        return

    slope = _slope_degrees(a)
    local_max = ndimage.maximum_filter(a, size=5, mode="reflect")
    local_min = ndimage.minimum_filter(a, size=5, mode="reflect")
    exact_flat = (local_max - local_min) <= 1.0
    patch_field = fbm(a.shape, 52.0, rng, octaves=3, persistence=0.52)

    flat_fraction = 0.18 if style in {"Ridged Wastes", "Serpentine Canyon", "Ravine Network"} else 0.30
    if np.any(exact_flat):
        threshold = float(np.quantile(patch_field[exact_flat], 1.0 - flat_fraction))
    else:
        threshold = 1.0
    active_flat = exact_flat & (patch_field >= threshold)
    active = (slope <= 24.0) & ~protected & (~exact_flat | active_flat)

    # These amplitudes are HG2 units (0.1 world-unit vertical increments).
    # The 3-12 sample wavelengths therefore create roughly 15-60 m bumps,
    # scallops and surface rolls with only metre-scale vertical relief.
    fine = fbm(a.shape, 4.2, rng, octaves=3, persistence=0.48)
    micro = fbm(a.shape, 9.0, rng, octaves=3, persistence=0.50)
    amount = (micro * 31.0 + fine * 24.0) * detail_strength
    amount *= np.clip((patch_field + 0.85) / 1.45, 0.20, 1.0)
    amount[~active] = 0.0
    a += amount

    # Add localized pebble/knoll fields rather than globally raising the noise
    # floor. They are deliberately low enough to remain driveable.
    anchors = _random_centers(active & (slope <= 12.0), max(1, int(a.size / (768 * 768) * 14)), rng)
    for ay, ax in anchors:
        for _ in range(int(rng.integers(4, 10))):
            cy = ay + float(rng.normal(0.0, 13.0))
            cx = ax + float(rng.normal(0.0, 13.0))
            if cy < 2 or cx < 2 or cy >= a.shape[0] - 2 or cx >= a.shape[1] - 2:
                continue
            _add_gaussian_feature(
                a,
                cy,
                cx,
                float(rng.uniform(2.0, 5.5)),
                float(rng.uniform(8.0, 28.0)),
                float(rng.uniform(0.65, 1.45)),
                float(rng.uniform(0.0, math.tau)),
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
    _layered_surface_detail(a, protected, profile, settings, rng)
    _patchy_micro_surface(a, protected, style, settings, rng)

    # Re-check after adding micro terrain. Organic broad saddles are permitted
    # on rough upland styles; deliberate ravine/canyon separation remains.
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
