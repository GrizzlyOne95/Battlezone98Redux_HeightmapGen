from __future__ import annotations

import math
import zlib

import numpy as np
from scipy import ndimage

from .hg2 import HG2Map, HG2_SAFE_MAX_HEIGHT
from .noise import fbm, smoothstep01
from .settings import GeneratorSettings
from .stock_detail import (
    STOCK_DETAIL_STYLES,
    _add_crater,
    _add_gaussian_feature,
    _add_small_ramp,
    _slope_degrees,
)
from .stock_detail_v2 import _component_state, _nearest_component_pair, _random_centers

# Stock campaign terrain is typically a calm connected substrate with highly
# localized authored features. These factors retain only this proportion of
# medium-scale procedural residual on gentle terrain outside feature provinces.
CALM_FACTORS = {
    "Terraced Labyrinth": 0.85,
    "Cratered Divide": 0.50,
    "Ravine Network": 0.42,
    "Mountain Basin": 0.30,
    "Radial Badlands": 0.42,
    "Ridged Wastes": 0.24,
    "Serpentine Canyon": 0.55,
    "Natural Badlands": 0.28,
}

# Fraction of the map allowed to retain stronger natural surface texture.
PROVINCE_FRACTIONS = {
    "Terraced Labyrinth": 0.30,
    "Cratered Divide": 0.30,
    "Ravine Network": 0.28,
    "Mountain Basin": 0.34,
    "Radial Badlands": 0.30,
    "Ridged Wastes": 0.42,
    "Serpentine Canyon": 0.22,
    "Natural Badlands": 0.38,
}

PASS_COUNTS = {
    "Terraced Labyrinth": 2,
    "Cratered Divide": 3,
    "Ravine Network": 0,
    "Mountain Basin": 4,
    "Radial Badlands": 3,
    "Ridged Wastes": 5,
    "Serpentine Canyon": 0,
    "Natural Badlands": 4,
}

TARGET_CONNECTED = {
    "Terraced Labyrinth": 0.45,
    "Cratered Divide": 0.50,
    "Ravine Network": 0.34,
    "Mountain Basin": 0.48,
    "Radial Badlands": 0.52,
    "Ridged Wastes": 0.42,
    "Serpentine Canyon": 0.42,
    "Natural Badlands": 0.50,
}


def _exact_flat_core(a: np.ndarray, size: int = 5) -> np.ndarray:
    local_max = ndimage.maximum_filter(a, size=size, mode="reflect")
    local_min = ndimage.minimum_filter(a, size=size, mode="reflect")
    return (local_max - local_min) <= 1.0


def _feature_province(shape: tuple[int, int], fraction: float, rng: np.random.Generator) -> np.ndarray:
    m = min(shape)
    field = fbm(shape, max(72.0, m * 0.20), rng, octaves=3, persistence=0.55)
    field += 0.35 * fbm(shape, max(32.0, m * 0.085), rng, octaves=2, persistence=0.50)
    threshold = float(np.quantile(field, 1.0 - float(np.clip(fraction, 0.08, 0.70))))
    mask = field >= threshold
    # Remove tiny islands so the active areas read as authored terrain regions.
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if n:
        counts = np.bincount(labels.ravel())
        min_size = max(128, int(np.prod(shape) * 0.004))
        keep = counts >= min_size
        keep[0] = False
        mask = keep[labels]
    return mask


def _regionalize_substrate(a: np.ndarray, style: str, province: np.ndarray) -> np.ndarray:
    original = np.asarray(a, dtype=np.float32)
    macro = ndimage.gaussian_filter(original, sigma=12.0, mode="reflect")
    residual = original - macro
    slope = _slope_degrees(original)
    flat = _exact_flat_core(original)
    base_factor = float(CALM_FACTORS[style])

    # Feature provinces keep much more of the existing natural texture. Quiet
    # ground is reduced toward the large-scale shape, which opens connected
    # vehicle space without globally flattening the actual map composition.
    factor = np.full(original.shape, base_factor, dtype=np.float32)
    factor[province] = np.maximum(factor[province], 0.78)
    # Preserve steep authored boundaries/ravines rather than rounding them away.
    cliff_keep = smoothstep01(np.clip((slope - 18.0) / 18.0, 0.0, 1.0))
    factor = factor * (1.0 - cliff_keep) + cliff_keep
    out = macro + residual * factor
    out[flat] = original[flat]
    return out.astype(np.float32)


def _organic_shelf(
    a: np.ndarray,
    cy: float,
    cx: float,
    rx: float,
    ry: float,
    offset: float,
    angle: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    margin = int(math.ceil(max(rx, ry) * 1.75 + 5.0))
    y0 = max(0, int(round(cy)) - margin)
    y1 = min(a.shape[0], int(round(cy)) + margin + 1)
    x0 = max(0, int(round(cx)) - margin)
    x1 = min(a.shape[1], int(round(cx)) + margin + 1)
    if y0 >= y1 or x0 >= x1:
        return cy, cx, angle
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    rel_y, rel_x = yy - float(cy), xx - float(cx)
    c, s = math.cos(angle), math.sin(angle)
    u = rel_x * c + rel_y * s
    v = -rel_x * s + rel_y * c
    radial = np.sqrt((u / max(rx, 1.0)) ** 2 + (v / max(ry, 1.0)) ** 2)
    local_rng = np.random.default_rng(int(rng.integers(1, 2**31 - 1)))
    warp = fbm(radial.shape, max(7.0, min(rx, ry) * 0.70), local_rng, octaves=3, persistence=0.52) * 0.14
    signed = radial + warp
    core = signed <= 0.58
    weight = np.zeros_like(signed, dtype=np.float32)
    weight[core] = 1.0
    feather = (signed > 0.58) & (signed < 1.10)
    weight[feather] = 1.0 - smoothstep01((signed[feather] - 0.58) / 0.52)
    area = a[y0:y1, x0:x1]
    base = float(np.median(area[core])) if np.any(core) else float(np.median(area))
    target = base + float(offset)
    area[:] = area * (1.0 - weight) + target * weight
    return cy, cx, angle


def _add_shelf_with_ramp(a: np.ndarray, cy: int, cx: int, rng: np.random.Generator, scale: str) -> None:
    if scale == "medium":
        rx = float(rng.uniform(13.0, 28.0))
        ry = float(rng.uniform(9.0, 20.0))
        offset = float(rng.choice([-1.0, 1.0]) * rng.uniform(28.0, 90.0))
    else:
        rx = float(rng.uniform(5.0, 13.0))
        ry = float(rng.uniform(3.5, 9.0))
        offset = float(rng.choice([-1.0, 1.0]) * rng.uniform(12.0, 52.0))
    angle = float(rng.uniform(0.0, math.tau))
    _organic_shelf(a, float(cy), float(cx), rx, ry, offset, angle, rng)

    # Most local shelves get one usable shallow approach. The ramp begins near
    # a random shelf edge and runs roughly normal to it; this creates the small
    # handmade-looking ramps visible throughout stock/custom maps.
    if rng.random() < 0.72:
        edge_angle = angle + float(rng.choice([0.0, math.pi])) + float(rng.uniform(-0.45, 0.45))
        ex = float(cx) + math.cos(edge_angle) * rx * 0.75
        ey = float(cy) + math.sin(edge_angle) * rx * 0.75
        rise = max(8.0, abs(offset) * float(rng.uniform(0.55, 0.95)))
        _add_small_ramp(
            a,
            ey,
            ex,
            float(rng.uniform(max(10.0, abs(offset) / 4.0), max(18.0, abs(offset) / 2.2))),
            float(rng.uniform(4.0, 8.5)),
            rise,
            edge_angle,
        )


def _place_regional_features(
    a: np.ndarray,
    style: str,
    province: np.ndarray,
    settings: GeneratorSettings,
    rng: np.random.Generator,
) -> None:
    slope = _slope_degrees(a)
    border = max(10, int(min(a.shape) * 0.012))
    eligible = slope <= 15.0
    eligible[:border] = False
    eligible[-border:] = False
    eligible[:, :border] = False
    eligible[:, -border:] = False
    regional = eligible & province
    quiet = eligible & ~province

    area_scale = max(0.10, float(a.size) / float(768 * 768))
    density = 0.65 + 0.85 * float(np.clip(settings.feature_density, 0.0, 1.0))

    # Moderate local shelves are the main source of small cliffs + staging
    # terraces. They occupy only parts of feature provinces, not the whole map.
    medium_shelves = int(round(8 * area_scale * density))
    small_shelves = int(round(34 * area_scale * density))
    crater_count = int(round(54 * area_scale * density))
    knoll_count = int(round(46 * area_scale * density))
    quiet_events = int(round(18 * area_scale * density))

    if style == "Ridged Wastes":
        medium_shelves = max(2, medium_shelves // 2)
        small_shelves = max(8, small_shelves // 2)
    elif style == "Terraced Labyrinth":
        crater_count = max(8, crater_count // 3)
    elif style == "Serpentine Canyon":
        medium_shelves = max(2, medium_shelves // 2)

    for cy, cx in _random_centers(regional, medium_shelves, rng):
        _add_shelf_with_ramp(a, cy, cx, rng, "medium")
    for cy, cx in _random_centers(regional, small_shelves, rng):
        _add_shelf_with_ramp(a, cy, cx, rng, "small")

    for cy, cx in _random_centers(regional, crater_count, rng):
        radius = float(rng.uniform(2.5, 10.5))
        _add_crater(a, cy, cx, radius, float(rng.uniform(9.0, 42.0)), float(rng.uniform(3.0, 18.0)), float(rng.uniform(0.62, 1.55)))
        if rng.random() < 0.26:
            ang = float(rng.uniform(0.0, math.tau))
            dist = radius * float(rng.uniform(0.8, 1.6))
            _add_crater(a, cy + math.sin(ang) * dist, cx + math.cos(ang) * dist, radius * float(rng.uniform(0.30, 0.62)), float(rng.uniform(5.0, 18.0)), float(rng.uniform(2.0, 8.0)), float(rng.uniform(0.75, 1.35)))

    for cy, cx in _random_centers(regional, knoll_count, rng):
        amplitude = float(rng.uniform(8.0, 36.0)) * (-1.0 if rng.random() < 0.35 else 1.0)
        _add_gaussian_feature(a, cy, cx, float(rng.uniform(2.5, 9.0)), amplitude, float(rng.uniform(0.60, 1.60)), float(rng.uniform(0.0, math.tau)))

    # A sparse set of small features outside provinces keeps traversal spaces
    # from feeling empty without destroying their broad readability.
    for cy, cx in _random_centers(quiet, quiet_events, rng):
        if rng.random() < 0.55:
            _add_gaussian_feature(a, cy, cx, float(rng.uniform(2.5, 6.0)), float(rng.uniform(7.0, 24.0)), float(rng.uniform(0.70, 1.45)), float(rng.uniform(0.0, math.tau)))
        else:
            radius = float(rng.uniform(2.5, 6.5))
            _add_crater(a, cy, cx, radius, float(rng.uniform(7.0, 24.0)), float(rng.uniform(2.0, 9.0)), float(rng.uniform(0.75, 1.35)))


def _regional_surface_detail(a: np.ndarray, province: np.ndarray, settings: GeneratorSettings, rng: np.random.Generator) -> None:
    strength = float(np.clip(settings.detail, 0.0, 1.5))
    if strength <= 0:
        return
    slope = _slope_degrees(a)
    flat = _exact_flat_core(a)

    # Retain exact staging flats by default. A small random subset of a province
    # receives micro texture; most flat cores remain untouched.
    flat_texture = fbm(a.shape, 48.0, rng, octaves=2, persistence=0.52)
    if np.any(flat & province):
        threshold = float(np.quantile(flat_texture[flat & province], 0.82))
        textured_flats = flat & province & (flat_texture >= threshold)
    else:
        textured_flats = np.zeros_like(flat)
    active = (slope <= 24.0) & province & (~flat | textured_flats)

    meso = fbm(a.shape, 10.0, rng, octaves=3, persistence=0.50)
    fine = fbm(a.shape, 4.0, rng, octaves=3, persistence=0.47)
    texture = (meso * 48.0 + fine * 35.0) * strength
    texture[~active] = 0.0
    a += texture

    # Broken cliff lips get a little high-frequency erosion; this raises local
    # shape complexity without putting the same noise over traversable flats.
    slope2 = _slope_degrees(a)
    cliff = (slope2 >= 18.0) & (slope2 <= 58.0)
    edge = fbm(a.shape, 5.0, rng, octaves=3, persistence=0.48)
    a += edge * 22.0 * strength * cliff


def _organic_pass_patch(a: np.ndarray, start: tuple[int, int], end: tuple[int, int], rng: np.random.Generator) -> None:
    y0, x0 = start
    y1, x1 = end
    dy, dx = float(y1 - y0), float(x1 - x0)
    distance = max(math.hypot(dy, dx), 1.0)
    cy, cx = (y0 + y1) * 0.5, (x0 + x1) * 0.5
    angle = math.atan2(dy, dx)
    c, s = math.cos(angle), math.sin(angle)
    half_len = distance * 0.5 + float(rng.uniform(12.0, 22.0))
    half_w = float(rng.uniform(12.0, 22.0))
    margin = int(math.ceil(half_len + half_w + 5.0))
    ya = max(0, int(round(cy)) - margin)
    yb = min(a.shape[0], int(round(cy)) + margin + 1)
    xa = max(0, int(round(cx)) - margin)
    xb = min(a.shape[1], int(round(cx)) + margin + 1)
    yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
    rx, ry = xx - cx, yy - cy
    u = rx * c + ry * s
    v = -rx * s + ry * c
    radial = np.sqrt((u / max(half_len, 1.0)) ** 2 + (v / max(half_w, 1.0)) ** 2)
    local_rng = np.random.default_rng(int(rng.integers(1, 2**31 - 1)))
    radial += fbm(radial.shape, max(9.0, half_w * 0.8), local_rng, octaves=3, persistence=0.52) * 0.10
    weight = np.zeros_like(radial, dtype=np.float32)
    core = radial <= 0.55
    weight[core] = 1.0
    feather = (radial > 0.55) & (radial < 1.05)
    weight[feather] = 1.0 - smoothstep01((radial[feather] - 0.55) / 0.50)

    h0 = float(np.median(a[max(0, y0 - 3):y0 + 4, max(0, x0 - 3):x0 + 4]))
    h1 = float(np.median(a[max(0, y1 - 3):y1 + 4, max(0, x1 - 3):x1 + 4]))
    t = np.clip((u + half_len) / max(2.0 * half_len, 1.0), 0.0, 1.0)
    target = h0 + (h1 - h0) * t
    area = a[ya:yb, xa:xb]
    area[:] = area * (1.0 - weight) + target * weight


def _repair_regions(a: np.ndarray, style: str, rng: np.random.Generator) -> None:
    if PASS_COUNTS.get(style, 0) <= 0:
        return
    target = float(TARGET_CONNECTED[style])
    for _ in range(PASS_COUNTS[style]):
        passable, labels, counts = _component_state(a, 15.0)
        if counts.size == 0:
            return
        largest = float(np.max(counts)) / float(a.size)
        if largest >= target:
            return
        pair = _nearest_component_pair(a, labels, counts, rng)
        if pair is None:
            return
        _organic_pass_patch(a, pair[0], pair[1], rng)


def enhance_stock_terrain(terrain: HG2Map, settings: GeneratorSettings, style: str) -> HG2Map:
    if style not in STOCK_DETAIL_STYLES:
        return terrain
    seed = (int(settings.seed) ^ (zlib.crc32(style.encode("utf-8")) & 0xFFFFFFFF) ^ 0x15C09A73) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    original = np.asarray(terrain.heights, dtype=np.float32)
    province = _feature_province(original.shape, PROVINCE_FRACTIONS[style], rng)
    a = _regionalize_substrate(original, style, province)
    _repair_regions(a, style, rng)
    _place_regional_features(a, style, province, settings, rng)
    _regional_surface_detail(a, province, settings, rng)

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
