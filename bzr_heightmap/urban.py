from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from .builder import TerrainBuilder
from .hg2 import HG2Map
from .noise import meander_path, vary_widths
from .planetary import _masked_fbm, _ramp_path, _repair_connectivity
from .settings import GeneratorSettings


def _line_positions(length: int, start_frac: float, end_frac: float, count: int, rng: np.random.Generator, jitter_frac: float = 0.02) -> list[float]:
    vals = np.linspace(length * start_frac, length * end_frac, max(count, 2), dtype=np.float32)
    out: list[float] = []
    for i, v in enumerate(vals):
        if i == 0 or i == len(vals) - 1:
            out.append(float(v))
        else:
            out.append(float(v + rng.uniform(-length * jitter_frac, length * jitter_frac)))
    return sorted(out)


def _corridor(b: TerrainBuilder, points, level: float, width: float, bank: float, rim: float = 20.0) -> None:
    widths = vary_widths(len(points), width, 0.08, b.rng, cycles=2)
    b.carve_variable_corridor_level(points, level, widths, bank=bank, rim_height=rim, edge_irregularity=0.0, protect_floor=True)


def _urban_ramp(b: TerrainBuilder, start, end, sh: float, eh: float, width: float, feather: float) -> None:
    sx, sy = start
    ex, ey = end
    mx = (sx + ex) * 0.5 + (ey - sy) * 0.10
    my = (sy + ey) * 0.5 - (ex - sx) * 0.10
    _ramp_path(b, [start, (mx, my), end], sh, eh, width, feather, 0.76)


def dense_city_grid(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(880)
    m = min(b.h, b.w)
    yy, xx = np.mgrid[0:b.h, 0:b.w].astype(np.float32)
    b.a += ((xx / max(b.w - 1, 1) - 0.5) * 160.0).astype(np.float32)
    _masked_fbm(b, 55, m * 0.16, m * 0.42, 0.26, ridged=False, softness=0.48)
    xs = _line_positions(b.w, 0.12, 0.88, 6 + int(3 * s.feature_density), b.rng, 0.018)
    ys = _line_positions(b.h, 0.12, 0.88, 6 + int(3 * s.feature_density), b.rng, 0.018)
    road_level = 860.0
    for x in xs:
        _corridor(b, [(x, b.h * 0.06), (x, b.h * 0.94)], road_level + float(b.rng.uniform(-18, 18)), m * 0.0105, m * 0.016, 18)
    for y in ys:
        _corridor(b, [(b.w * 0.06, y), (b.w * 0.94, y)], road_level + float(b.rng.uniform(-18, 18)), m * 0.0105, m * 0.016, 18)
    if s.feature_density > 0.35:
        _corridor(b, [(b.w * 0.08, b.h * 0.28), (b.w * 0.92, b.h * 0.68)], road_level + 8, m * 0.012, m * 0.018, 20)
    if s.feature_density > 0.58:
        _corridor(b, [(b.w * 0.18, b.h * 0.92), (b.w * 0.76, b.h * 0.08)], road_level - 10, m * 0.011, m * 0.017, 18)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        for y0, y1 in zip(ys[:-1], ys[1:]):
            if (x1 - x0) < m * 0.05 or (y1 - y0) < m * 0.05:
                continue
            cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
            rx, ry = (x1 - x0) * 0.34, (y1 - y0) * 0.34
            target = road_level + float(b.rng.uniform(16, 70))
            if b.rng.random() < 0.18:
                target = road_level - float(b.rng.uniform(12, 28))
            b.flatten_pad(cx, cy, rx, ry, target=target, feather=m * 0.008, rectangular=True)
    for _ in range(2 + int(s.synthetic_pads > 0) + int(s.feature_density > 0.55)):
        cx = float(b.rng.uniform(b.w * 0.22, b.w * 0.78))
        cy = float(b.rng.uniform(b.h * 0.22, b.h * 0.78))
        b.flatten_pad(cx, cy, m * float(b.rng.uniform(0.05, 0.085)), m * float(b.rng.uniform(0.04, 0.07)), target=road_level + float(b.rng.uniform(10, 34)), feather=m * 0.009, rectangular=True)
    b.add_detail(3.5 * s.detail, m * 0.050)
    _repair_connectivity(b, 40, 0.94, 2, m * 0.007, m * 0.012)
    return b.finalize(center_height=1200.0, preserve_flats=True)


def hillside_mega_district(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(760)
    m = min(b.h, b.w)
    yy, xx = np.mgrid[0:b.h, 0:b.w].astype(np.float32)
    slope_dir = float(b.rng.uniform(-0.55, 0.55))
    b.a += ((yy / max(b.h - 1, 1)) * 780.0 + (xx / max(b.w - 1, 1) - 0.5) * 140.0 * slope_dir).astype(np.float32)
    _masked_fbm(b, 70, m * 0.18, m * 0.38, 0.22, ridged=True, softness=0.45)
    terrace_levels = [940, 1120, 1310, 1520]
    ys = [b.h * 0.18, b.h * 0.36, b.h * 0.56, b.h * 0.76]
    for level, y in zip(terrace_levels, ys):
        path = meander_path((b.w * 0.08, y), (b.w * 0.92, y + float(b.rng.uniform(-m * 0.02, m * 0.02))), 9, m * 0.025 * s.naturalization, b.rng)
        _corridor(b, path, level, m * 0.012, m * 0.018, 22)
        for frac in np.linspace(0.18, 0.82, 4):
            cx = b.w * frac + float(b.rng.uniform(-m * 0.02, m * 0.02))
            cy = y + float(b.rng.uniform(-m * 0.018, m * 0.018))
            b.flatten_pad(cx, cy, m * 0.055, m * 0.038, target=level + float(b.rng.uniform(18, 58)), feather=m * 0.010, rectangular=True)
    for i in range(len(terrace_levels) - 1):
        sx = b.w * float(0.20 + 0.22 * i) + float(b.rng.uniform(-m * 0.015, m * 0.015))
        ex = sx + float(b.rng.uniform(m * 0.08, m * 0.14))
        _urban_ramp(b, (sx, ys[i]), (ex, ys[i + 1]), terrace_levels[i], terrace_levels[i + 1], m * 0.010, m * 0.013)
    b.flatten_pad(b.w * 0.50, b.h * 0.14, m * 0.08, m * 0.05, target=860, feather=m * 0.011, rectangular=True)
    b.add_detail(3.2 * s.detail, m * 0.050)
    _repair_connectivity(b, 40, 0.95, 3, m * 0.0075, m * 0.013)
    return b.finalize(center_height=1750.0, preserve_flats=True)


def industrial_terrace(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(700)
    m = min(b.h, b.w)
    b.add_fbm(90, m * 0.28, ridged=False, octaves=3)
    for level, frac in zip([760, 980, 1220], [0.24, 0.19, 0.14]):
        b.stamp_blob_shelf(level, frac, m * 0.22, feather=m * 0.024, warp_px=m * 0.02 * s.naturalization, protect_core=True)
    for frac in [0.20, 0.44, 0.68, 0.84]:
        _corridor(b, [(b.w * 0.08, b.h * frac), (b.w * 0.92, b.h * frac)], 860 + frac * 120, m * 0.014, m * 0.020, 24)
    for frac in [0.24, 0.52, 0.80]:
        _corridor(b, [(b.w * frac, b.h * 0.08), (b.w * frac, b.h * 0.92)], 860 + frac * 80, m * 0.013, m * 0.019, 22)
    for _ in range(10):
        cx = float(b.rng.uniform(b.w * 0.15, b.w * 0.85)); cy = float(b.rng.uniform(b.h * 0.15, b.h * 0.85))
        rx = m * float(b.rng.uniform(0.045, 0.095)); ry = m * float(b.rng.uniform(0.035, 0.08))
        iy, ix = int(round(cy)), int(round(cx))
        target = float(b.a[np.clip(iy, 0, b.h - 1), np.clip(ix, 0, b.w - 1)]) + float(b.rng.uniform(10, 38))
        b.flatten_pad(cx, cy, rx, ry, target=target, feather=m * 0.010, rectangular=True)
    b.add_detail(2.6 * s.detail, m * 0.055)
    _repair_connectivity(b, 40, 0.95, 3, m * 0.008, m * 0.014)
    return b.finalize(center_height=1500.0, preserve_flats=True)


def sunken_expressway(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(900)
    m = min(b.h, b.w)
    _masked_fbm(b, 50, m * 0.19, m * 0.42, 0.18, ridged=False, softness=0.45)
    trunk = meander_path((b.w * 0.08, b.h * 0.40), (b.w * 0.92, b.h * 0.62), 12, m * 0.06 * s.naturalization, b.rng)
    b.carve_variable_corridor_level(trunk, 610, vary_widths(len(trunk), m * 0.019, 0.20, b.rng), bank=m * 0.030, rim_height=36, edge_irregularity=m * 0.003)
    spur = meander_path((b.w * 0.24, b.h * 0.08), trunk[len(trunk) // 2], 7, m * 0.04 * s.naturalization, b.rng)
    b.carve_variable_corridor_level(spur, 630, vary_widths(len(spur), m * 0.015, 0.16, b.rng), bank=m * 0.024, rim_height=26)
    xs = _line_positions(b.w, 0.16, 0.84, 5, b.rng, 0.015)
    ys = _line_positions(b.h, 0.16, 0.84, 5, b.rng, 0.015)
    for x in xs:
        _corridor(b, [(x, b.h * 0.06), (x, b.h * 0.94)], 900 + float(b.rng.uniform(-15, 15)), m * 0.010, m * 0.016, 18)
    for y in ys:
        _corridor(b, [(b.w * 0.06, y), (b.w * 0.94, y)], 900 + float(b.rng.uniform(-15, 15)), m * 0.010, m * 0.016, 18)
    for t in [0.28, 0.55, 0.76]:
        px, py = trunk[int(t * (len(trunk) - 1))]
        _urban_ramp(b, (px - m * 0.07, py - m * 0.03), (px, py), 900, 610, m * 0.009, m * 0.012)
        _urban_ramp(b, (px + m * 0.07, py + m * 0.03), (px, py), 900, 610, m * 0.009, m * 0.012)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        for y0, y1 in zip(ys[:-1], ys[1:]):
            b.flatten_pad((x0 + x1) * 0.5, (y0 + y1) * 0.5, (x1 - x0) * 0.31, (y1 - y0) * 0.31, target=935 + float(b.rng.uniform(8, 42)), feather=m * 0.009, rectangular=True)
    b.add_detail(2.7 * s.detail, m * 0.054)
    _repair_connectivity(b, 40, 0.96, 3, m * 0.007, m * 0.012)
    return b.finalize(center_height=1350.0, preserve_flats=True)


def arcology_edge(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(820)
    m = min(b.h, b.w)
    b.add_fbm(60, m * 0.24, ridged=False, octaves=3)
    side = str(b.rng.choice(["left", "right", "top", "bottom"]))
    if side in {"left", "right"}:
        cx, cy = b.w * (0.22 if side == "left" else 0.78), b.h * 0.52
        rx, ry = m * 0.18, m * 0.14
    else:
        cx, cy = b.w * 0.52, b.h * (0.22 if side == "top" else 0.78)
        rx, ry = m * 0.14, m * 0.18
    b.flatten_pad(cx, cy, rx, ry, target=980, feather=m * 0.014, rectangular=True)
    b.flatten_pad(cx, cy, rx * 0.56, ry * 0.56, target=1020, feather=m * 0.011, rectangular=True)
    loop = [(b.w * 0.12, b.h * 0.12), (b.w * 0.88, b.h * 0.12), (b.w * 0.88, b.h * 0.88), (b.w * 0.12, b.h * 0.88), (b.w * 0.12, b.h * 0.12)]
    _corridor(b, loop, 860, m * 0.011, m * 0.018, 20)
    for fracx in [0.28, 0.52, 0.74]:
        for fracy in [0.28, 0.52, 0.74]:
            if abs(fracx - cx / b.w) < 0.12 and abs(fracy - cy / b.h) < 0.12:
                continue
            b.flatten_pad(b.w * fracx, b.h * fracy, m * 0.05, m * 0.038, target=905 + float(b.rng.uniform(8, 40)), feather=m * 0.010, rectangular=True)
    for start in [(b.w * 0.12, cy), (cx, b.h * 0.12), (b.w * 0.88, cy)]:
        path = meander_path(start, (cx, cy), 7, m * 0.018 * s.naturalization, b.rng)
        _corridor(b, path, 900, m * 0.013, m * 0.020, 18)
    b.add_detail(2.4 * s.detail, m * 0.056)
    _repair_connectivity(b, 40, 0.96, 2, m * 0.007, m * 0.012)
    return b.finalize(center_height=1320.0, preserve_flats=True)


def cyberpunk_mixed_district(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(860)
    m = min(b.h, b.w)
    yy, xx = np.mgrid[0:b.h, 0:b.w].astype(np.float32)
    basin = np.hypot((xx - b.w * 0.55) / max(m * 0.62, 1), (yy - b.h * 0.46) / max(m * 0.54, 1))
    b.a -= (np.exp(-0.5 * (basin / 0.70) ** 2) * 170).astype(np.float32)
    b.a += (((yy / max(b.h - 1, 1)) - 0.5) * 220).astype(np.float32)
    _masked_fbm(b, 95, m * 0.14, m * 0.32, 0.24, ridged=True, softness=0.42)
    arterials = [
        ([(b.w * 0.08, b.h * 0.22), (b.w * 0.92, b.h * 0.30)], 840),
        ([(b.w * 0.10, b.h * 0.76), (b.w * 0.88, b.h * 0.58)], 885),
        ([(b.w * 0.28, b.h * 0.08), (b.w * 0.32, b.h * 0.92)], 960),
        ([(b.w * 0.70, b.h * 0.08), (b.w * 0.60, b.h * 0.92)], 1010),
    ]
    for path, level in arterials:
        _corridor(b, path, level, m * 0.013, m * 0.020, 20)
    for _ in range(3):
        p0 = (float(b.rng.uniform(b.w * 0.08, b.w * 0.30)), float(b.rng.uniform(b.h * 0.10, b.h * 0.90)))
        p1 = (float(b.rng.uniform(b.w * 0.70, b.w * 0.92)), float(b.rng.uniform(b.h * 0.10, b.h * 0.90)))
        _corridor(b, meander_path(p0, p1, 8, m * 0.030 * s.naturalization, b.rng), float(b.rng.uniform(860, 1000)), m * 0.0105, m * 0.016, 16)
    for _ in range(8):
        b.flatten_pad(float(b.rng.uniform(b.w * 0.14, b.w * 0.86)), float(b.rng.uniform(b.h * 0.14, b.h * 0.86)), m * float(b.rng.uniform(0.035, 0.08)), m * float(b.rng.uniform(0.03, 0.07)), target=float(b.rng.uniform(860, 1080)), feather=m * 0.009, rectangular=bool(b.rng.random() < 0.75))
    b.add_detail(3.2 * s.detail, m * 0.052)
    _repair_connectivity(b, 40, 0.95, 3, m * 0.007, m * 0.012)
    return b.finalize(center_height=1450.0, preserve_flats=True)


URBAN_RECIPES: Dict[str, Callable[[GeneratorSettings], HG2Map]] = {
    "Dense City Grid": dense_city_grid,
    "Hillside Mega-District": hillside_mega_district,
    "Industrial Terrace": industrial_terrace,
    "Sunken Expressway": sunken_expressway,
    "Arcology Edge": arcology_edge,
    "Cyberpunk Mixed District": cyberpunk_mixed_district,
}
