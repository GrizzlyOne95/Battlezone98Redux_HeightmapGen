from __future__ import annotations

import math
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
    b = TerrainBuilder(s).set_level(760)
    m = min(b.h, b.w)

    # Keep the terrain itself gentle; the industrial character should come from
    # stepped yards, service cuts, and long grade connections rather than noise.
    b.add_fbm(52, m * 0.34, ridged=False, octaves=3)

    # Four broad industrial elevation bands distribute useful terrain across the
    # whole map instead of leaving a few isolated shelves in a mostly empty field.
    row_levels = [820.0, 950.0, 1080.0, 1210.0]
    ys = _line_positions(b.h, 0.15, 0.85, len(row_levels), b.rng, 0.018)
    xs = _line_positions(b.w, 0.14, 0.86, 4 + int(s.feature_density > 0.58), b.rng, 0.024)

    for row, (level, y) in enumerate(zip(row_levels, ys)):
        end_y = y + float(b.rng.uniform(-m * 0.018, m * 0.018))
        service = meander_path(
            (b.w * 0.06, y),
            (b.w * 0.94, end_y),
            10,
            m * 0.018 * s.naturalization,
            b.rng,
        )
        _corridor(b, service, level, m * 0.0125, m * 0.020, 20)

        for col, x in enumerate(xs):
            cx = x + float(b.rng.uniform(-m * 0.018, m * 0.018))
            cy = y + float(b.rng.uniform(-m * 0.022, m * 0.022))
            large_yard = (col + row) % 3 == 0
            if large_yard:
                rx = m * float(b.rng.uniform(0.078, 0.115))
                ry = m * float(b.rng.uniform(0.052, 0.080))
            else:
                rx = m * float(b.rng.uniform(0.052, 0.082))
                ry = m * float(b.rng.uniform(0.036, 0.060))
            target = level + float(b.rng.uniform(8, 34))
            b.flatten_pad(cx, cy, rx, ry, target=target, feather=m * 0.010, rectangular=True)

    # Long, broad ramps connect every terrace band at several different columns.
    # Alternating offsets keep the layout industrial rather than perfectly gridded.
    ramp_columns = [0.20, 0.47, 0.74]
    if s.feature_density > 0.70:
        ramp_columns.append(0.86)
    for row in range(len(row_levels) - 1):
        for j, frac in enumerate(ramp_columns):
            x0 = b.w * frac + float(b.rng.uniform(-m * 0.020, m * 0.020))
            x1 = x0 + (m * 0.055 if (row + j) % 2 == 0 else -m * 0.055)
            _urban_ramp(
                b,
                (x0, ys[row]),
                (x1, ys[row + 1]),
                row_levels[row],
                row_levels[row + 1],
                m * 0.0105,
                m * 0.015,
            )

    # Cross-district service routes provide alternative circulation without
    # turning this family into another Dense City Grid.
    cross_routes = [
        ((0.08, 0.31), (0.64, 0.43), 920.0),
        ((0.34, 0.58), (0.92, 0.49), 1040.0),
        ((0.10, 0.73), (0.70, 0.88), 1140.0),
    ]
    for (sx, sy), (ex, ey), level in cross_routes:
        path = meander_path(
            (b.w * sx, b.h * sy),
            (b.w * ex, b.h * ey),
            8,
            m * 0.022 * s.naturalization,
            b.rng,
        )
        _corridor(b, path, level + float(b.rng.uniform(-18, 18)), m * 0.0105, m * 0.017, 16)

    # Smaller yards and loading/service pads fill interstitial space while
    # preserving open ground for bases, factories, and later building placement.
    secondary_count = 8 + int(8 * s.feature_density)
    for _ in range(secondary_count):
        cx = float(b.rng.uniform(b.w * 0.10, b.w * 0.90))
        cy = float(b.rng.uniform(b.h * 0.10, b.h * 0.90))
        iy = int(np.clip(round(cy), 0, b.h - 1))
        ix = int(np.clip(round(cx), 0, b.w - 1))
        target = float(b.a[iy, ix]) + float(b.rng.uniform(4, 24))
        b.flatten_pad(
            cx,
            cy,
            m * float(b.rng.uniform(0.032, 0.062)),
            m * float(b.rng.uniform(0.025, 0.050)),
            target=target,
            feather=m * 0.009,
            rectangular=True,
        )

    b.add_detail(2.2 * s.detail, m * 0.060)
    _repair_connectivity(b, 40, 0.97, 4, m * 0.008, m * 0.014)
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


def _stepped_podium(
    b: TerrainBuilder,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    base_level: float,
    tiers: int = 3,
    step_height: float = 32.0,
    feather: float = 2.8,
) -> None:
    """Generate a multi-tier ziggurat/setback building pad."""
    for tier in range(tiers):
        scale = 1.0 - tier * 0.25
        if rx * scale < 2.5 or ry * scale < 2.5:
            break
        target = base_level + tier * step_height
        b.flatten_pad(cx, cy, rx * scale, ry * scale, target=target, feather=feather, rectangular=True)


def _stair_steps(
    b: TerrainBuilder,
    start: tuple[float, float],
    end: tuple[float, float],
    sh: float,
    eh: float,
    step_count: int = 4,
    step_width: float = 14.0,
    step_length: float = 10.0,
    feather: float = 2.0,
) -> None:
    """Create a sequence of stepped shelf terraces climbing a slope."""
    sx, sy = start
    ex, ey = end
    for i in range(step_count):
        t = (i + 0.5) / max(step_count, 1)
        px = sx + (ex - sx) * t
        py = sy + (ey - sy) * t
        h = sh + (eh - sh) * (i / max(step_count - 1, 1))
        b.flatten_pad(px, py, step_width * 0.5, step_length * 0.5, target=h, feather=feather, rectangular=True)


def cyberpunk_megacity(s: GeneratorSettings) -> HG2Map:
    """Dense, multi-level cyberpunk megacity with maximum macro and micro detail."""
    b = TerrainBuilder(s).set_level(860)
    m = min(b.h, b.w)
    yy, xx = np.mgrid[0:b.h, 0:b.w].astype(np.float32)

    # 1. Macro terrain foundation: broad regional basin, cross-city tilt, and base relief
    basin = np.hypot((xx - b.w * 0.52) / max(m * 0.68, 1), (yy - b.h * 0.48) / max(m * 0.60, 1))
    b.a -= (np.exp(-0.5 * (basin / 0.72) ** 2) * 150).astype(np.float32)
    b.a += (((yy / max(b.h - 1, 1)) - 0.5) * 430).astype(np.float32)
    _masked_fbm(b, 75, m * 0.075, m * 0.24, 0.30, ridged=True, softness=0.42)

    density = float(np.clip(s.feature_density, 0.0, 1.0))
    # Scale density with map dimensions for 8x8 (10km) maps while maintaining playability
    scale_factor = max(b.w, b.h) / 256.0
    x_count = max(8, int(round((7.5 + 4.0 * density) * (0.85 + 0.15 * math.sqrt(scale_factor)))))
    y_count = max(9, int(round((8.5 + 4.0 * density) * (0.85 + 0.15 * math.sqrt(scale_factor)))))

    xs = _line_positions(b.w, 0.06, 0.94, x_count, b.rng, 0.007)
    ys = _line_positions(b.h, 0.06, 0.94, y_count, b.rng, 0.007)

    def street_level(y: float) -> float:
        t = float(np.clip(y / max(b.h - 1, 1), 0.0, 1.0))
        return float(np.interp(t, [0.0, 0.28, 0.54, 0.78, 1.0], [790, 900, 1030, 1180, 1320]))

    # 2. Horizontal avenues & tiered district thoroughfares
    for row, y in enumerate(ys):
        level = street_level(y) + float(b.rng.uniform(-8, 8))
        major = row % 3 == 0
        path = meander_path(
            (b.w * 0.03, y),
            (b.w * 0.97, y + float(b.rng.uniform(-m * 0.008, m * 0.008))),
            14,
            m * 0.008 * s.naturalization,
            b.rng,
        )
        _corridor(b, path, level, m * (0.0068 if major else 0.0044), m * 0.0075, 18 if major else 10)

    # 3. North/South graded boulevards spanning all tiers
    for col, x in enumerate(xs):
        sway = m * (0.008 if col % 2 else -0.008) * s.naturalization
        points = [(x, b.h * 0.03), (x + sway, b.h * 0.50), (x, b.h * 0.97)]
        _ramp_path(
            b,
            points,
            street_level(b.h * 0.03),
            street_level(b.h * 0.97),
            m * (0.0060 if col % 3 == 0 else 0.0042),
            m * 0.0068,
            0.82,
        )

    # 4. Sunken Expressway Trunk & Multilevel Interchanges
    trunk = meander_path(
        (b.w * 0.02, b.h * 0.68),
        (b.w * 0.98, b.h * 0.32),
        20,
        m * 0.022 * s.naturalization,
        b.rng,
    )
    express_level = 680.0
    b.carve_variable_corridor_level(
        trunk,
        express_level,
        vary_widths(len(trunk), m * 0.0082, 0.12, b.rng, cycles=3),
        bank=m * 0.012,
        rim_height=32,
        edge_irregularity=m * 0.0012,
        protect_floor=True,
    )
    # Expressway center divider island / barrier
    for idx in range(2, len(trunk) - 2, 3):
        p_prev = trunk[idx]
        p_next = trunk[idx + 1]
        mx = (p_prev[0] + p_next[0]) * 0.5
        my = (p_prev[1] + p_next[1]) * 0.5
        b.flatten_pad(mx, my, m * 0.008, m * 0.0025, target=express_level + 18, feather=m * 0.0015, rectangular=True)

    # Multi-level on/off ramps connecting street grid to expressway
    for index in (3, 6, 10, 14, 17):
        if index >= len(trunk):
            continue
        px, py = trunk[index]
        for direction in (-1.0, 1.0):
            start = (px + direction * m * 0.055, py + m * 0.045 * direction)
            _urban_ramp(b, start, (px, py), street_level(start[1]), express_level, m * 0.0048, m * 0.0065)

    # Overpass causeway bridges crossing the sunken expressway
    for cross_x in (b.w * 0.36, b.w * 0.64):
        # find approximate y where expressway intersects cross_x
        ey = float(np.interp(cross_x, [p[0] for p in trunk], [p[1] for p in trunk]))
        overpass_level = street_level(ey)
        _ramp_path(b, [(cross_x, ey - m * 0.035), (cross_x, ey + m * 0.035)], overpass_level, overpass_level, m * 0.0055, m * 0.0035, 0.95)

    # 5. Southeast Hillside Megadistrict: 5 Cascading Terraces, Switchbacks, and Stair-Steps
    hill_levels = (820.0, 950.0, 1090.0, 1240.0, 1390.0)
    hill_ys = (0.09, 0.19, 0.30, 0.41, 0.52)
    for level, fy in zip(hill_levels, hill_ys):
        b.flatten_pad(b.w * 0.78, b.h * fy, m * 0.145, m * 0.026, target=level, feather=m * 0.005, rectangular=True)
        # Subdivided hillside lots perched along the terrace shelf
        for hx_frac in (0.68, 0.76, 0.84, 0.90):
            b.flatten_pad(b.w * hx_frac, b.h * (fy + 0.008), m * 0.024, m * 0.016, target=level + 24, feather=m * 0.003, rectangular=True)

    for index in range(len(hill_levels) - 1):
        x0 = b.w * (0.68 + (index % 3) * 0.08)
        x1 = x0 + m * (0.065 if index % 2 == 0 else -0.055)
        # Main vehicle switchback ramp
        _urban_ramp(
            b,
            (x0, b.h * hill_ys[index]),
            (x1, b.h * hill_ys[index + 1]),
            hill_levels[index],
            hill_levels[index + 1],
            m * 0.0052,
            m * 0.0075,
        )
        # Parallel micro stair-step climb on opposite side
        stair_x = b.w * (0.91 - (index % 2) * 0.06)
        _stair_steps(
            b,
            (stair_x, b.h * (hill_ys[index] + 0.015)),
            (stair_x, b.h * (hill_ys[index + 1] - 0.015)),
            hill_levels[index],
            hill_levels[index + 1],
            step_count=4,
            step_width=m * 0.012,
            step_length=m * 0.008,
            feather=m * 0.002,
        )

    # 6. Northeast Arcology Mega-Spire Complex (Multi-Tier Ziggurat + Moat + Bastions)
    arc_x, arc_y = b.w * 0.79, b.h * 0.77
    arc_levels = (1380.0, 1480.0, 1590.0, 1710.0, 1830.0)
    arc_radii = (
        (m * 0.125, m * 0.100),
        (m * 0.095, m * 0.076),
        (m * 0.068, m * 0.054),
        (m * 0.044, m * 0.035),
        (m * 0.024, m * 0.019),
    )
    for lvl, (rx, ry) in zip(arc_levels, arc_radii):
        b.flatten_pad(arc_x, arc_y, rx, ry, target=lvl, feather=m * 0.0045, rectangular=True)

    # 4 Corner Bastions with pads
    for bx_sign in (-1.0, 1.0):
        for by_sign in (-1.0, 1.0):
            bx = arc_x + bx_sign * m * 0.110
            by = arc_y + by_sign * m * 0.088
            b.flatten_pad(bx, by, m * 0.022, m * 0.020, target=arc_levels[1] + 16, feather=m * 0.0035, rectangular=True)
            # Ramped bridge to main podium
            _urban_ramp(b, (bx, by), (arc_x + bx_sign * m * 0.070, arc_y + by_sign * m * 0.055), arc_levels[1] + 16, arc_levels[2], m * 0.0040, m * 0.0045)

    # Grand axial stair-ramps on all 4 faces climbing the pyramid spire
    _urban_ramp(b, (arc_x, arc_y - m * 0.105), (arc_x, arc_y - m * 0.025), arc_levels[0], arc_levels[3], m * 0.0050, m * 0.0050)
    _urban_ramp(b, (arc_x, arc_y + m * 0.105), (arc_x, arc_y + m * 0.025), arc_levels[0], arc_levels[3], m * 0.0050, m * 0.0050)
    _urban_ramp(b, (arc_x - m * 0.125, arc_y), (arc_x - m * 0.035, arc_y), arc_levels[0], arc_levels[3], m * 0.0050, m * 0.0050)
    _urban_ramp(b, (arc_x + m * 0.125, arc_y), (arc_x + m * 0.035, arc_y), arc_levels[0], arc_levels[3], m * 0.0050, m * 0.0050)

    # Arcology perimeter sunken transit canal
    arc_moat = [
        (arc_x - m * 0.145, arc_y - m * 0.120),
        (arc_x + m * 0.145, arc_y - m * 0.120),
        (arc_x + m * 0.145, arc_y + m * 0.120),
        (arc_x - m * 0.145, arc_y + m * 0.120),
        (arc_x - m * 0.145, arc_y - m * 0.120),
    ]
    _corridor(b, arc_moat, arc_levels[0] - 38, m * 0.0060, m * 0.0080, 16)
    # Causeway approach bridges over the moat
    arc_approaches = (
        ((b.w * 0.58, arc_y), (arc_x - m * 0.125, arc_y)),
        ((arc_x, b.h * 0.58), (arc_x, arc_y - m * 0.100)),
        ((b.w * 0.96, arc_y), (arc_x + m * 0.125, arc_y)),
    )
    for start, end in arc_approaches:
        _urban_ramp(b, start, end, street_level(start[1]), arc_levels[0], m * 0.0055, m * 0.0075)

    # 7. Northwest Industrial & Logistics Mega-Sector
    for row, fy in enumerate((0.68, 0.78, 0.88)):
        ind_level = street_level(b.h * fy) + (24 if row % 2 else 60)
        # Sunken rail/freight service corridor
        service = meander_path(
            (b.w * 0.06, b.h * fy),
            (b.w * 0.46, b.h * (fy - 0.020)),
            10,
            m * 0.010 * s.naturalization,
            b.rng,
        )
        _corridor(b, service, ind_level - 22, m * 0.0050, m * 0.0075, 14)

        # Stepped warehouse pads & loading docks
        for fx in (0.13, 0.27, 0.40):
            yard_cx = b.w * fx
            yard_cy = b.h * (fy - 0.040)
            _stepped_podium(b, yard_cx, yard_cy, m * 0.042, m * 0.024, ind_level, tiers=2, step_height=22.0, feather=m * 0.0030)
            # Ramped loading dock from service trench up to warehouse pad
            _urban_ramp(b, (yard_cx, b.h * fy), (yard_cx, yard_cy), ind_level - 22, ind_level, m * 0.0040, m * 0.0045)

        # Cylindrical storage silos with blast containment rims
        for fx in (0.20, 0.34):
            silo_cx = b.w * fx
            silo_cy = b.h * (fy + 0.035)
            b.flatten_pad(silo_cx, silo_cy, m * 0.016, m * 0.016, target=ind_level + 36, feather=m * 0.0025, rectangular=False)

    # 8. High-Density Downtown & Commercial City Blocks with Micro-Detail
    for col, (x0, x1) in enumerate(zip(xs[:-1], xs[1:])):
        for row, (y0, y1) in enumerate(zip(ys[:-1], ys[1:])):
            cell_w, cell_h = x1 - x0, y1 - y0
            if cell_w < 20 or cell_h < 20:
                continue
            cx = (x0 + x1) * 0.5
            cy = (y0 + y1) * 0.5
            fx, fy = cx / b.w, cy / b.h

            # Skip Arcology, NW Industrial yards, and SE hillside terraces already authored
            if fx > 0.64 and fy > 0.60:
                continue
            if fx < 0.48 and fy > 0.64:
                continue
            if fx > 0.65 and fy < 0.54:
                continue

            base_st = street_level(cy)
            downtown = 0.24 < fx < 0.66 and 0.22 < fy < 0.64
            lot_target = base_st + float(b.rng.uniform(18, 55))
            if (row + col) % 5 == 0:
                lot_target += float(b.rng.uniform(40, 90))

            # Block layout grammar based on cell proportions and district location
            if downtown and cell_w > 36 and cell_h > 36:
                # Quad-tower block with central cross-alleys and multi-tier ziggurats
                rx = cell_w * 0.17
                ry = cell_h * 0.17
                tower_heights = []
                for dx_s in (-0.22, 0.22):
                    for dy_s in (-0.22, 0.22):
                        tcx = cx + cell_w * dx_s
                        tcy = cy + cell_h * dy_s
                        th = lot_target + float(b.rng.uniform(-10, 25))
                        tower_heights.append((tcx, tcy, th))
                        _stepped_podium(b, tcx, tcy, rx, ry, th, tiers=3, step_height=28.0, feather=m * 0.0022)
                        # Micro access ramp from street
                        ramp_start = (tcx, cy + (cell_h * 0.42 if dy_s > 0 else -cell_h * 0.42))
                        _urban_ramp(b, ramp_start, (tcx, tcy), base_st, th, m * 0.0035, m * 0.0035)

                # Skybridge catwalk connecting two towers in the block
                t1, t2 = tower_heights[0], tower_heights[1]
                _ramp_path(b, [(t1[0], t1[1]), (t2[0], t2[1])], t1[2] + 28, t2[2] + 28, m * 0.0030, m * 0.0025, 0.95)

            elif cell_w > cell_h * 1.25:
                # Dual-podium split block
                for offset in (-0.22, 0.22):
                    tcx = cx + cell_w * offset
                    th = lot_target + float(b.rng.uniform(-8, 16))
                    _stepped_podium(b, tcx, cy, cell_w * 0.18, cell_h * 0.32, th, tiers=2, step_height=26.0, feather=m * 0.0025)
                    # Access ramp
                    _urban_ramp(b, (tcx, cy + cell_h * 0.40), (tcx, cy), base_st, th, m * 0.0035, m * 0.0035)
            elif (row + col) % 6 == 2:
                # Sunken courtyard / plaza with stepped perimeter walkway
                plaza_level = base_st - 18.0
                b.flatten_pad(cx, cy, cell_w * 0.35, cell_h * 0.35, target=base_st + 12, feather=m * 0.0025, rectangular=True)
                b.flatten_pad(cx, cy, cell_w * 0.22, cell_h * 0.22, target=plaza_level, feather=m * 0.0022, rectangular=True)
                # Corner access ramps into sunken plaza
                _urban_ramp(b, (cx - cell_w * 0.32, cy - cell_h * 0.32), (cx - cell_w * 0.15, cy - cell_h * 0.15), base_st, plaza_level, m * 0.0030, m * 0.0030)
                _urban_ramp(b, (cx + cell_w * 0.32, cy + cell_h * 0.32), (cx + cell_w * 0.15, cy + cell_h * 0.15), base_st, plaza_level, m * 0.0030, m * 0.0030)
            else:
                # Standard tiered ziggurat building lot
                _stepped_podium(b, cx, cy, cell_w * 0.32, cell_h * 0.32, lot_target, tiers=2 + int(downtown), step_height=30.0, feather=m * 0.0025)
                # Driveway ramp from street
                _urban_ramp(b, (cx, cy + cell_h * 0.40), (cx, cy), base_st, lot_target, m * 0.0035, m * 0.0035)

    # 9. Plazas, helipads, and staging nodes
    plazas = (
        (0.18, 0.20, 0.032, 28.0),
        (0.50, 0.52, 0.048, 34.0),
        (0.36, 0.36, 0.038, 42.0),
        (0.82, 0.78, 0.034, 30.0),
        (0.76, 0.18, 0.028, 24.0),
    )
    for fracx, fracy, scale, pad_elev in plazas:
        px = b.w * fracx
        py = b.h * fracy
        b.flatten_pad(px, py, m * scale * 1.30, m * scale, target=street_level(py) + pad_elev, feather=m * 0.0035, rectangular=True)
        # Helipad core in center
        b.flatten_pad(px, py, m * scale * 0.45, m * scale * 0.45, target=street_level(py) + pad_elev + 16, feather=m * 0.0020, rectangular=False)

    # 10. High-frequency micro-relief & connectivity guarantee
    b.add_detail(6.5 * s.detail, m * 0.018)
    _repair_connectivity(b, 40, 0.98, 5, m * 0.0055, m * 0.0090)
    return b.finalize(center_height=1580.0, preserve_flats=True)


URBAN_RECIPES: Dict[str, Callable[[GeneratorSettings], HG2Map]] = {
    "Dense City Grid": dense_city_grid,
    "Hillside Mega-District": hillside_mega_district,
    "Industrial Terrace": industrial_terrace,
    "Sunken Expressway": sunken_expressway,
    "Arcology Edge": arcology_edge,
    "Cyberpunk Mixed District": cyberpunk_mixed_district,
    "Cyberpunk Megacity": cyberpunk_megacity,
}

