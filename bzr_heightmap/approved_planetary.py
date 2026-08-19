from __future__ import annotations

import math
from typing import Callable, Dict

import numpy as np

from .builder import TerrainBuilder
from .hg2 import HG2Map
from .noise import meander_path, vary_widths
from .planetary import _edge_point, _masked_fbm, _path_side_ramps, _repair_connectivity
from .settings import GeneratorSettings


def _central_peak(b: TerrainBuilder, cx: float, cy: float, radius: float, height_scale: float) -> None:
    yy, xx = np.mgrid[0:b.h, 0:b.w].astype(np.float32)
    peak_r = max(radius * 0.18, 2.0)
    rr = np.sqrt(((xx - cx) / peak_r) ** 2 + ((yy - cy) / peak_r) ** 2)
    b.a += (np.exp(-0.5 * (rr / 0.85) ** 2) * (radius * height_scale)).astype(np.float32)


def mars_rift(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(2200)
    m = min(b.h, b.w)
    b.add_fbm(120, m * 0.34, ridged=False, octaves=3)
    trunk = meander_path(_edge_point(b, "left", 0.24), _edge_point(b, "right", 0.74), 16, m * 0.13 * s.naturalization, b.rng)
    widths = vary_widths(len(trunk), m * 0.030, 0.34, b.rng)
    b.carve_variable_corridor_level(trunk, 760, widths, bank=m * 0.060, rim_height=120, edge_irregularity=m * 0.006)
    for side in ["top", "bottom", "top"]:
        start = _edge_point(b, side, float(b.rng.uniform(0.16, 0.84)))
        target = trunk[int(b.rng.integers(len(trunk) // 5, len(trunk) * 4 // 5))]
        branch = meander_path(start, target, 9, m * 0.075 * s.naturalization, b.rng)
        b.carve_variable_corridor_level(branch, 760, vary_widths(len(branch), m * 0.019, 0.28, b.rng), bank=m * 0.040, rim_height=75, edge_irregularity=m * 0.005)
    _path_side_ramps(b, trunk, 3 + int(2 * s.feature_density), m * 0.020, m * 0.120, m * 0.0060, m * 0.012)
    _masked_fbm(b, 110, m * 0.11, m * 0.32, 0.26, ridged=True, softness=0.42)
    b.add_detail(6.0 * s.detail, m * 0.042)
    _repair_connectivity(b, 40, 0.90, 3, m * 0.0075, m * 0.013)
    return b.finalize(center_height=2550.0, preserve_flats=True)


def callisto_craterlands(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(860)
    m = min(b.h, b.w)
    b.add_fbm(40, m * 0.42, ridged=False, octaves=3)
    for _ in range(14 + int(12 * s.feature_density)):
        radius = float(b.rng.uniform(m * 0.020, m * 0.078))
        cx = float(b.rng.uniform(radius * 1.4, b.w - radius * 1.4))
        cy = float(b.rng.uniform(radius * 1.4, b.h - radius * 1.4))
        b.crater(cx, cy, radius, radius * float(b.rng.uniform(1.4, 2.1)), radius * float(b.rng.uniform(0.68, 1.02)), float(b.rng.uniform(0.965, 1.035)))
        if radius >= m * 0.050 and b.rng.random() < 0.28:
            _central_peak(b, cx, cy, radius, float(b.rng.uniform(0.08, 0.15)))
    for _ in range(45 + int(35 * s.feature_density)):
        radius = float(b.rng.uniform(m * 0.006, m * 0.016))
        cx = float(b.rng.uniform(radius * 1.4, b.w - radius * 1.4))
        cy = float(b.rng.uniform(radius * 1.4, b.h - radius * 1.4))
        b.crater(cx, cy, radius, radius * float(b.rng.uniform(1.1, 1.8)), radius * float(b.rng.uniform(0.30, 0.58)), float(b.rng.uniform(0.97, 1.03)))
    b.add_detail(1.8 * s.detail, m * 0.060)
    return b.finalize(center_height=1220.0, preserve_flats=True)


def europa_fracture_plains(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(840)
    m = min(b.h, b.w)
    b.add_fbm(22, m * 0.42, ridged=False, octaves=2)
    for _ in range(5 + int(2 * s.feature_density)):
        p0 = _edge_point(b, str(b.rng.choice(["left", "top", "right", "bottom"])), float(b.rng.uniform(0.12, 0.88)))
        p1 = _edge_point(b, str(b.rng.choice(["left", "top", "right", "bottom"])), float(b.rng.uniform(0.12, 0.88)))
        path = meander_path(p0, p1, 10, m * 0.05 * s.naturalization, b.rng)
        start = int(np.clip(round(float(b.rng.uniform(0.05, 0.18)) * (len(path) - 1)), 0, len(path) - 2))
        end = int(np.clip(round(float(b.rng.uniform(0.55, 0.92)) * (len(path) - 1)), start + 1, len(path) - 1))
        segment = [(float(x), float(y)) for x, y in path[start : end + 1]]
        b.carve_path(segment, depth=24, half_width=m * 0.0032, bank=m * 0.0065, rim=0)
        ridge1 = []
        ridge2 = []
        offset = m * float(b.rng.uniform(0.0045, 0.0070))
        for i, (x, y) in enumerate(segment):
            prev = segment[max(i - 1, 0)]
            nxt = segment[min(i + 1, len(segment) - 1)]
            tx, ty = nxt[0] - prev[0], nxt[1] - prev[1]
            mag = max(math.hypot(tx, ty), 1e-4)
            nx, ny = -ty / mag, tx / mag
            ridge1.append((x + nx * offset, y + ny * offset))
            ridge2.append((x - nx * offset, y - ny * offset))
        b.add_ridge_path(ridge1, height=float(b.rng.uniform(26, 58)), half_width=m * 0.0028, falloff=m * 0.006)
        b.add_ridge_path(ridge2, height=float(b.rng.uniform(26, 58)), half_width=m * 0.0028, falloff=m * 0.006)
    for _ in range(2 + int(s.feature_density > 0.60)):
        cx = float(b.rng.uniform(b.w * 0.18, b.w * 0.82))
        cy = float(b.rng.uniform(b.h * 0.18, b.h * 0.82))
        b.flatten_pad(cx, cy, m * float(b.rng.uniform(0.040, 0.072)), m * float(b.rng.uniform(0.030, 0.060)), target=860 + float(b.rng.uniform(-12, 16)), feather=m * 0.010, rectangular=False)
    _masked_fbm(b, 38, m * 0.030, m * 0.10, 0.18, ridged=True, softness=0.40)
    b.add_detail(1.4 * s.detail, m * 0.060)
    return b.finalize(center_height=1160.0, preserve_flats=True)


APPROVED_PLANETARY_RECIPES: Dict[str, Callable[[GeneratorSettings], HG2Map]] = {
    "Mars Rift": mars_rift,
    "Callisto Craterlands": callisto_craterlands,
    "Europa Fracture Plains": europa_fracture_plains,
}
