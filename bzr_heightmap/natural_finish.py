from __future__ import annotations

from dataclasses import dataclass
import math
import zlib

import numpy as np
from scipy import ndimage

from .hg2 import HG2Map, HG2_SAFE_MAX_HEIGHT
from .noise import fbm
from .settings import GeneratorSettings
from .stock_detail import _add_crater, _add_gaussian_feature, _add_ledge, _add_small_ramp, _slope_degrees

REFERENCE_AREA = 768 * 768


@dataclass(frozen=True)
class NaturalFinishProfile:
    province_fraction: float
    knolls: int
    bowls: int
    craterlets: int
    ledges: int
    ramps: int
    max_slope: float = 12.0
    flat_touch_fraction: float = 0.16


# These recipes already have strong macro morphology. This final pass deliberately
# adds discrete native-HG2-scale events only; it does not smooth, reconnect, or
# paint global fBm over the map. Sparse Mission Field is intentionally excluded.
NATURAL_FINISH_PROFILES: dict[str, NaturalFinishProfile] = {
    "Campaign Canyon Network": NaturalFinishProfile(0.34, 48, 30, 16, 14, 10, 12.0, 0.14),
    "Compartmented Plateau": NaturalFinishProfile(0.30, 34, 22, 6, 18, 14, 11.0, 0.12),
    "Escarpment Stronghold": NaturalFinishProfile(0.36, 44, 28, 8, 20, 16, 12.0, 0.14),
    "Mars Rift": NaturalFinishProfile(0.30, 38, 24, 18, 8, 8, 11.0, 0.12),
}

NATURAL_FINISH_STYLES = frozenset(NATURAL_FINISH_PROFILES)


def _exact_flat_core(a: np.ndarray, size: int = 5) -> np.ndarray:
    hi = ndimage.maximum_filter(a, size=size, mode="reflect")
    lo = ndimage.minimum_filter(a, size=size, mode="reflect")
    return (hi - lo) <= 1.0


def _province_mask(shape: tuple[int, int], fraction: float, rng: np.random.Generator) -> np.ndarray:
    m = min(shape)
    field = fbm(shape, max(58.0, m * 0.18), rng, octaves=3, persistence=0.54)
    field += 0.30 * fbm(shape, max(24.0, m * 0.065), rng, octaves=2, persistence=0.50)
    threshold = float(np.quantile(field, 1.0 - float(np.clip(fraction, 0.08, 0.65))))
    mask = field >= threshold
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if n:
        counts = np.bincount(labels.ravel())
        keep = counts >= max(96, int(np.prod(shape) * 0.003))
        keep[0] = False
        mask = keep[labels]
    return mask


def _pick(mask: np.ndarray, count: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    ys, xs = np.nonzero(mask)
    if count <= 0 or ys.size == 0:
        return []
    n = min(int(count), int(ys.size))
    idx = rng.choice(ys.size, size=n, replace=False)
    return [(int(ys[i]), int(xs[i])) for i in np.atleast_1d(idx)]


def _scaled(base: int, area: int, density: float) -> int:
    area_scale = max(0.12, float(area) / float(REFERENCE_AREA))
    density_scale = 0.65 + 0.70 * float(np.clip(density, 0.0, 1.0))
    return max(1, int(round(float(base) * area_scale * density_scale)))


def enhance_natural_finish(terrain: HG2Map, settings: GeneratorSettings, style: str) -> HG2Map:
    profile = NATURAL_FINISH_PROFILES.get(style)
    if profile is None:
        return terrain

    style_hash = zlib.crc32(style.encode("utf-8")) & 0xFFFFFFFF
    seed = (int(settings.seed) ^ style_hash ^ 0x4E415446) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    a = np.asarray(terrain.heights, dtype=np.float32).copy()
    province = _province_mask(a.shape, profile.province_fraction, rng)
    slope = _slope_degrees(ndimage.gaussian_filter(a, 1.0, mode="reflect"))
    flat = _exact_flat_core(a)

    # Only a small subset of exact flats participates. Broad staging/mission
    # ground remains intact while feature provinces gain authored-looking events.
    gate_field = fbm(a.shape, max(36.0, min(a.shape) * 0.075), rng, octaves=2, persistence=0.50)
    if np.any(flat & province):
        q = float(np.quantile(gate_field[flat & province], 1.0 - profile.flat_touch_fraction))
        touch_flat = flat & province & (gate_field >= q)
    else:
        touch_flat = np.zeros_like(flat)

    eligible = province & (slope <= profile.max_slope) & (~flat | touch_flat)
    edge = max(8, int(min(a.shape) * 0.012))
    eligible[:edge] = False
    eligible[-edge:] = False
    eligible[:, :edge] = False
    eligible[:, -edge:] = False

    strength = 0.55 + float(np.clip(settings.detail, 0.0, 1.25))
    area = int(a.size)

    for cy, cx in _pick(eligible, _scaled(profile.knolls, area, settings.feature_density), rng):
        _add_gaussian_feature(
            a, cy, cx,
            float(rng.uniform(2.5, 9.5)),
            float(rng.uniform(7.0, 28.0)) * strength,
            float(rng.uniform(0.65, 1.65)),
            float(rng.uniform(0.0, math.tau)),
        )

    for cy, cx in _pick(eligible, _scaled(profile.bowls, area, settings.feature_density), rng):
        _add_gaussian_feature(
            a, cy, cx,
            float(rng.uniform(3.0, 11.0)),
            -float(rng.uniform(6.0, 24.0)) * strength,
            float(rng.uniform(0.70, 1.55)),
            float(rng.uniform(0.0, math.tau)),
        )

    for cy, cx in _pick(eligible, _scaled(profile.craterlets, area, settings.feature_density), rng):
        radius = float(rng.uniform(2.0, 7.5))
        _add_crater(
            a, cy, cx, radius,
            float(rng.uniform(6.0, 24.0)) * strength,
            float(rng.uniform(2.5, 10.0)) * strength,
            float(rng.uniform(0.72, 1.42)),
        )

    for cy, cx in _pick(eligible, _scaled(profile.ledges, area, settings.feature_density), rng):
        offset = float(rng.choice([-1.0, 1.0]) * rng.uniform(10.0, 34.0)) * strength
        angle = float(rng.uniform(0.0, math.tau))
        rx = float(rng.uniform(5.0, 13.0))
        ry = float(rng.uniform(3.0, 8.0))
        _add_ledge(a, cy, cx, rx, ry, offset, angle)
        if rng.random() < 0.72:
            _add_small_ramp(
                a,
                cy + math.sin(angle) * rx * 0.7,
                cx + math.cos(angle) * rx * 0.7,
                float(rng.uniform(10.0, 24.0)),
                float(rng.uniform(3.0, 7.0)),
                max(8.0, abs(offset) * float(rng.uniform(0.55, 0.90))),
                angle,
            )

    for cy, cx in _pick(eligible, _scaled(profile.ramps, area, settings.feature_density), rng):
        _add_small_ramp(
            a, cy, cx,
            float(rng.uniform(10.0, 26.0)),
            float(rng.uniform(3.5, 7.5)),
            float(rng.uniform(9.0, 30.0)) * strength,
            float(rng.uniform(0.0, math.tau)),
        )

    heights = np.clip(np.rint(a), 0, HG2_SAFE_MAX_HEIGHT).astype(np.uint16)
    return HG2Map(
        heights,
        terrain.zones_x,
        terrain.zones_z,
        terrain.zone_bits,
        terrain.structure_version,
        terrain.map_version,
    )


__all__ = ["NATURAL_FINISH_STYLES", "enhance_natural_finish"]
