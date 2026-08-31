from __future__ import annotations

import math
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
from scipy import ndimage

from .builder import TerrainBuilder
from .hg2 import HG2Map
from .noise import fbm, meander_path, ridged_fbm, smoothstep01, vary_widths
from .settings import GeneratorSettings

Point = Tuple[float, float]


def _edge_point(b: TerrainBuilder, side: str, t: float) -> Point:
    if side == "left":
        return 0.0, t * (b.h - 1)
    if side == "right":
        return b.w - 1.0, t * (b.h - 1)
    if side == "top":
        return t * (b.w - 1), 0.0
    return t * (b.w - 1), b.h - 1.0


def _masked_fbm(
    b: TerrainBuilder,
    amplitude: float,
    feature_px: float,
    mask_feature_px: float,
    coverage: float,
    *,
    ridged: bool = False,
    softness: float = 0.5,
) -> None:
    detail = ridged_fbm(b.a.shape, feature_px, b.rng, 4) if ridged else fbm(b.a.shape, feature_px, b.rng, octaves=4)
    field = fbm(b.a.shape, mask_feature_px, b.rng, octaves=3, persistence=0.52)
    threshold = float(np.quantile(field, np.clip(1.0 - coverage, 0.05, 0.95)))
    spread = max(float(np.std(field)) * max(softness, 0.08), 1e-4)
    mask = smoothstep01(np.clip((field - threshold + spread) / (spread * 2.0), 0.0, 1.0))
    b.a += detail.astype(np.float32) * mask.astype(np.float32) * float(amplitude)


def _ramp_path(
    b: TerrainBuilder,
    points: Sequence[Point],
    start_height: float,
    end_height: float,
    half_width: float,
    feather: float,
    endpoint_taper: float = 0.72,
) -> None:
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) < 2:
        return
    seg = np.asarray(
        [math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)],
        dtype=np.float32,
    )
    total = float(np.sum(seg))
    if total < 1e-4:
        return
    cumulative = np.concatenate(([0.0], np.cumsum(seg[:-1]))).astype(np.float32)
    margin = float(half_width + feather + 2)
    minx = max(0, int(math.floor(min(x for x, _ in pts) - margin)))
    maxx = min(b.w - 1, int(math.ceil(max(x for x, _ in pts) + margin)))
    miny = max(0, int(math.floor(min(y for _, y in pts) - margin)))
    maxy = min(b.h - 1, int(math.ceil(max(y for _, y in pts) + margin)))
    if minx >= maxx or miny >= maxy:
        return
    yy, xx = np.mgrid[miny : maxy + 1, minx : maxx + 1].astype(np.float32)
    best = np.full(xx.shape, np.inf, dtype=np.float32)
    progress = np.zeros(xx.shape, dtype=np.float32)
    for i in range(len(pts) - 1):
        sx, sy = pts[i]
        ex, ey = pts[i + 1]
        vx, vy = ex - sx, ey - sy
        l2 = max(vx * vx + vy * vy, 1e-6)
        lt = np.clip(((xx - sx) * vx + (yy - sy) * vy) / l2, 0, 1)
        nx, ny = sx + lt * vx, sy + lt * vy
        d2 = (xx - nx) ** 2 + (yy - ny) ** 2
        better = d2 < best
        best[better] = d2[better]
        pr = (cumulative[i] + lt * seg[i]) / total
        progress[better] = pr[better]
    lateral = np.sqrt(best)
    t = np.clip(progress, 0, 1)
    taper = endpoint_taper + (1.0 - endpoint_taper) * np.power(np.clip(np.sin(np.pi * t), 0, 1), 0.72)
    local_width = np.maximum(float(half_width) * taper, 1.0)
    weight = 1.0 - smoothstep01((lateral - local_width) / max(float(feather), 1.0))
    weight[lateral >= local_width + feather] = 0.0
    target = float(start_height) + (float(end_height) - float(start_height)) * t
    region = b.a[miny : maxy + 1, minx : maxx + 1]
    region[:] = region * (1 - weight) + target * weight
    b.protected[miny : maxy + 1, minx : maxx + 1] |= lateral <= local_width * 0.34


def _path_side_ramps(
    b: TerrainBuilder,
    points: Sequence[Point],
    count: int,
    inner_offset: float,
    outer_offset: float,
    half_width: float,
    feather: float,
) -> None:
    """Place long diagonal/contour-following access ramps along a canyon wall."""
    if len(points) < 5:
        return
    candidates = np.linspace(0.12, 0.88, max(count * 5, count), dtype=np.float32)
    b.rng.shuffle(candidates)
    placed = 0
    for t in candidates:
        if placed >= count:
            break
        idx = int(np.clip(round(float(t) * (len(points) - 1)), 1, len(points) - 2))
        px, py = points[idx]
        ax, ay = points[idx - 1]
        bx, by = points[idx + 1]
        tx, ty = bx - ax, by - ay
        mag = math.hypot(tx, ty)
        if mag < 1e-5:
            continue
        tx, ty = tx / mag, ty / mag
        nx, ny = -ty, tx
        side = -1.0 if placed % 2 else 1.0
        span = max(outer_offset - inner_offset, 1.0)
        tangent = span * float(b.rng.uniform(1.8, 2.8))
        sign = -1.0 if b.rng.random() < 0.5 else 1.0
        sx = px + nx * inner_offset * side - tx * tangent * 0.48 * sign
        sy = py + ny * inner_offset * side - ty * tangent * 0.48 * sign
        ex = px + nx * outer_offset * side + tx * tangent * 0.52 * sign
        ey = py + ny * outer_offset * side + ty * tangent * 0.52 * sign
        margin = half_width + feather + 3
        if min(sx, ex) < margin or min(sy, ey) < margin or max(sx, ex) >= b.w - margin or max(sy, ey) >= b.h - margin:
            continue
        sh = float(b.a[int(round(sy)), int(round(sx))])
        eh = float(b.a[int(round(ey)), int(round(ex))])
        if abs(eh - sh) < 110:
            continue
        p1 = (sx + tx * tangent * 0.30 * sign + nx * span * 0.10 * side, sy + ty * tangent * 0.30 * sign + ny * span * 0.10 * side)
        p2 = (sx + tx * tangent * 0.70 * sign + nx * span * 0.58 * side, sy + ty * tangent * 0.70 * sign + ny * span * 0.58 * side)
        _ramp_path(b, [(sx, sy), p1, p2, (ex, ey)], sh, eh, half_width, feather, 0.68)
        placed += 1


def _repair_connectivity(
    b: TerrainBuilder,
    max_slope_deg: float = 38.0,
    target_pct: float = 0.92,
    max_repairs: int = 5,
    half_width: float = 7.0,
    feather: float = 12.0,
) -> None:
    """Heuristic terrain repair; it does not model the exact BZR vehicle/AI slope limit."""
    for _ in range(max_repairs):
        a = b.a.astype(np.float32)
        gy, gx = np.gradient(ndimage.gaussian_filter(a, 2.4) * 0.1, 5.0, 5.0)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        passable = slope <= max_slope_deg
        labels, n = ndimage.label(passable, structure=np.ones((3, 3), dtype=np.uint8))
        counts = np.bincount(labels.ravel())[1:]
        total = int(np.count_nonzero(passable))
        if n <= 1 or total <= 0:
            return
        main = int(np.argmax(counts)) + 1
        if float(counts[main - 1]) / total >= target_pct:
            return
        min_size = max(32, int(a.size * 0.008))
        others = [i + 1 for i, c in enumerate(counts) if i + 1 != main and c >= min_size]
        if not others:
            return
        other = max(others, key=lambda lab: counts[lab - 1])
        ma, ob = labels == main, labels == other
        mba = ma & ~ndimage.binary_erosion(ma)
        obb = ob & ~ndimage.binary_erosion(ob)
        my, mx = np.nonzero(mba)
        oy, ox = np.nonzero(obb)
        if not len(mx) or not len(ox):
            return
        step_m, step_o = max(1, len(mx) // 800), max(1, len(ox) // 800)
        mp = np.column_stack([mx[::step_m], my[::step_m]]).astype(np.float32)
        op = np.column_stack([ox[::step_o], oy[::step_o]]).astype(np.float32)
        best = None
        for i in range(0, len(op), 160):
            chunk = op[i : i + 160]
            d = ((chunk[:, None, :] - mp[None, :, :]) ** 2).sum(axis=2)
            pos = np.unravel_index(int(np.argmin(d)), d.shape)
            val = float(d[pos])
            if best is None or val < best[0]:
                best = (val, chunk[pos[0]], mp[pos[1]])
        if best is None:
            return
        p0, p3 = tuple(float(v) for v in best[1]), tuple(float(v) for v in best[2])
        mxp, myp = (p0[0] + p3[0]) * 0.5, (p0[1] + p3[1]) * 0.5
        ix, iy = int(np.clip(round(mxp), 0, b.w - 1)), int(np.clip(round(myp), 0, b.h - 1))
        gxx, gyy = float(gx[iy, ix]), float(gy[iy, ix])
        gm = math.hypot(gxx, gyy)
        if gm > 1e-5:
            tx, ty = -gyy / gm, gxx / gm
        else:
            dx, dy = p3[0] - p0[0], p3[1] - p0[1]
            dm = max(math.hypot(dx, dy), 1.0)
            tx, ty = -dy / dm, dx / dm
        direct = math.hypot(p3[0] - p0[0], p3[1] - p0[1])
        tangent = min(max(direct * 0.8, min(b.h, b.w) * 0.055), min(b.h, b.w) * 0.15)
        sign = -1.0 if b.rng.random() < 0.5 else 1.0
        p1 = (p0[0] * 0.72 + p3[0] * 0.28 + tx * tangent * sign, p0[1] * 0.72 + p3[1] * 0.28 + ty * tangent * sign)
        p2 = (p0[0] * 0.30 + p3[0] * 0.70 + tx * tangent * 0.72 * sign, p0[1] * 0.30 + p3[1] * 0.70 + ty * tangent * 0.72 * sign)
        sh = float(b.a[int(np.clip(round(p0[1]), 0, b.h - 1)), int(np.clip(round(p0[0]), 0, b.w - 1))])
        eh = float(b.a[int(np.clip(round(p3[1]), 0, b.h - 1)), int(np.clip(round(p3[0]), 0, b.w - 1))])
        _ramp_path(b, [p0, p1, p2, p3], sh, eh, half_width, feather, 0.72)


def _arc_points(cx: float, cy: float, rx: float, ry: float, a0: float, a1: float, count: int, jitter: float, rng) -> list[Point]:
    pts: list[Point] = []
    for a in np.linspace(a0, a1, count):
        rr = 1.0 + float(rng.uniform(-jitter, jitter))
        pts.append((cx + math.cos(float(a)) * rx * rr, cy + math.sin(float(a)) * ry * rr))
    return pts


def pluto_basin(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(1040)
    m = min(b.h, b.w)
    yy, xx = np.mgrid[0:b.h, 0:b.w].astype(np.float32)
    cx = b.w * float(b.rng.uniform(0.46, 0.54))
    cy = b.h * float(b.rng.uniform(0.47, 0.55))
    rx = b.w * float(b.rng.uniform(0.31, 0.38))
    ry = b.h * float(b.rng.uniform(0.28, 0.35))
    ell = np.sqrt(((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2)
    basin = np.exp(-0.5 * (ell / 0.74) ** 4)
    rim = np.exp(-0.5 * ((ell - 1.0) / 0.22) ** 2)
    b.a -= (basin * 380).astype(np.float32)
    b.a += (rim * 140).astype(np.float32)
    b.add_fbm(58, m * 0.56, ridged=False, octaves=3)
    # Preserve a broad lowland floor while keeping Pluto's basin/rim macro form.
    b.stamp_blob_shelf(target=720.0, area_fraction=0.13, feature_px=m * 0.42, feather=m * 0.028, warp_px=m * 0.018 * s.naturalization)
    rough = ridged_fbm(b.a.shape, m * 0.155, b.rng, 4)
    outer = smoothstep01(np.clip((ell - 0.78) / 0.50, 0, 1))
    patch = smoothstep01(np.clip((fbm(b.a.shape, m * 0.36, b.rng, octaves=3, persistence=0.52) + 0.18) / 0.8, 0, 1))
    angle = np.arctan2((yy - cy) / max(ry, 1), (xx - cx) / max(rx, 1))
    saddle = np.ones_like(b.a)
    for base_ang in [float(b.rng.uniform(-0.7, 0.3)), float(b.rng.uniform(2.3, 3.5))]:
        da = np.arctan2(np.sin(angle - base_ang), np.cos(angle - base_ang))
        saddle *= 1.0 - 0.72 * np.exp(-0.5 * (da / 0.22) ** 2) * np.exp(-0.5 * ((ell - 1.0) / 0.34) ** 2)
    b.a += (rough * outer * patch * saddle * (150 + 55 * s.feature_density)).astype(np.float32)
    for _ in range(3 + int(4 * s.feature_density)):
        rad = float(b.rng.uniform(m * 0.018, m * 0.045))
        ang = float(b.rng.uniform(0, math.tau))
        rr = float(b.rng.uniform(0.80, 1.20))
        px, py = cx + math.cos(ang) * rx * rr, cy + math.sin(ang) * ry * rr
        if rad < px < b.w - rad and rad < py < b.h - rad:
            b.crater(px, py, rad, rad * 1.05, rad * 0.30, float(b.rng.uniform(0.82, 1.22)))
    b.add_detail(3 * s.detail, m * 0.060)
    _repair_connectivity(b, 38, 0.94, 3, m * 0.009, m * 0.016)
    return b.finalize(center_height=1380, preserve_flats=True)


def venus_shield(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(720)
    m = min(b.h, b.w)
    yy, xx = np.mgrid[0:b.h, 0:b.w].astype(np.float32)
    cx = b.w * float(b.rng.uniform(0.45, 0.55))
    cy = b.h * float(b.rng.uniform(0.44, 0.56))
    rx = m * float(b.rng.uniform(0.29, 0.35))
    ry = m * float(b.rng.uniform(0.25, 0.31))
    r = np.sqrt(((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2)
    shield = np.clip(1 - r, 0, 1)
    shield = shield * shield * (3 - 2 * shield)
    b.a += (shield * 700 + np.exp(-0.5 * ((r - 0.92) / 0.34) ** 2) * 85).astype(np.float32)
    b.crater(cx, cy, m * 0.060, 280, 115, float(b.rng.uniform(0.90, 1.12)))
    b.crater(cx + m * 0.018, cy - m * 0.010, m * 0.025, 95, 34, 1.05)
    b.add_fbm(60, m * 0.46, ridged=False, octaves=3)
    phase = float(b.rng.uniform(0, math.tau))
    for i in range(5 + int(2 * s.feature_density)):
        rr = m * float(b.rng.uniform(0.12, 0.29))
        span = float(b.rng.uniform(0.45, 1.0))
        mid = phase + i * math.tau / 6.0 + float(b.rng.uniform(-0.35, 0.35))
        pts = _arc_points(cx, cy, rr, rr * float(b.rng.uniform(0.82, 1.15)), mid - span / 2, mid + span / 2, 8, 0.055, b.rng)
        b.add_ridge_path(pts, float(b.rng.uniform(30, 62)), m * 0.0045, m * 0.022)
    for _ in range(3):
        ang = float(b.rng.uniform(0, math.tau))
        p0 = (cx + math.cos(ang) * m * 0.12, cy + math.sin(ang) * m * 0.11)
        p1 = (cx + math.cos(ang + b.rng.uniform(-0.20, 0.20)) * m * 0.39, cy + math.sin(ang + b.rng.uniform(-0.20, 0.20)) * m * 0.36)
        b.carve_path(meander_path(p0, p1, 10, m * 0.034 * s.naturalization, b.rng), float(b.rng.uniform(30, 52)), m * 0.012, m * 0.048, 4)
    b.add_random_craters(2 + int(3 * s.feature_density), (m * 0.010, m * 0.025), 0.9, 0.25)
    b.add_detail(3 * s.detail, m * 0.055)
    _repair_connectivity(b, 38, 0.94, 3, m * 0.009, m * 0.015)
    return b.finalize(center_height=1420, preserve_flats=True)


def lunar_catena(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(590)
    m = min(b.h, b.w)
    b.add_fbm(66, m * 0.50, ridged=False, octaves=3)
    b.add_random_craters(3 + int(4 * s.feature_density), (m * 0.060, m * 0.105), 0.92, 0.36)
    b.add_random_craters(7 + int(9 * s.feature_density), (m * 0.023, m * 0.050), 1.00, 0.37)
    b.add_random_craters(12 + int(16 * s.feature_density), (m * 0.007, m * 0.019), 1.08, 0.30)
    chain_count = 1 + int(s.feature_density > 0.70)
    for ci in range(chain_count):
        if ci == 0:
            st = _edge_point(b, "left", float(b.rng.uniform(0.30, 0.68)))
            en = _edge_point(b, "right", float(b.rng.uniform(0.30, 0.72)))
        else:
            st = _edge_point(b, "top", float(b.rng.uniform(0.25, 0.70)))
            en = _edge_point(b, "bottom", float(b.rng.uniform(0.28, 0.75)))
        path = meander_path(st, en, 16, m * 0.055 * s.naturalization, b.rng)
        chosen = list(range(2, len(path) - 2, 2))[:9]
        for idx in chosen:
            if b.rng.random() < 0.18:
                continue
            px, py = path[idx]
            px += float(b.rng.uniform(-m * 0.010, m * 0.010))
            py += float(b.rng.uniform(-m * 0.010, m * 0.010))
            rad = m * float(b.rng.uniform(0.013, 0.025))
            b.crater(px, py, rad, rad * float(b.rng.uniform(0.95, 1.30)), rad * float(b.rng.uniform(0.24, 0.40)), float(b.rng.uniform(0.84, 1.18)))
    yy, xx = np.mgrid[0:b.h, 0:b.w].astype(np.float32)
    for _ in range(2):
        cx = float(b.rng.uniform(m * 0.18, b.w - m * 0.18))
        cy = float(b.rng.uniform(m * 0.18, b.h - m * 0.18))
        rad = m * float(b.rng.uniform(0.13, 0.21))
        b.a += (np.exp(-0.5 * (np.hypot(xx - cx, yy - cy) / max(rad, 1) / 0.90) ** 2) * float(b.rng.uniform(60, 110))).astype(np.float32)
    b.add_detail(2.5 * s.detail, m * 0.060)
    _repair_connectivity(b, 38, 0.95, 2, m * 0.008, m * 0.014)
    return b.finalize(center_height=1080, preserve_flats=False)


def mars_rift(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(2320)
    m = min(b.h, b.w)
    b.add_fbm(125, m * 0.43, ridged=False, octaves=3)
    trunk = meander_path(_edge_point(b, "left", float(b.rng.uniform(0.18, 0.34))), _edge_point(b, "right", float(b.rng.uniform(0.66, 0.82))), 18, m * 0.13 * (0.45 + 0.55 * s.naturalization), b.rng)
    widths = vary_widths(len(trunk), m * 0.038, 0.40 * s.naturalization, b.rng, cycles=6)
    b.carve_variable_corridor_level(trunk, 720, widths, m * 0.095, 105, m * 0.008 * s.naturalization)
    for i in range(3 + int(3 * s.feature_density)):
        branch = meander_path(_edge_point(b, "top" if i % 2 == 0 else "bottom", float(b.rng.uniform(0.12, 0.88))), trunk[int(b.rng.integers(3, len(trunk) - 3))], 10, m * 0.075 * s.naturalization, b.rng)
        widths2 = vary_widths(len(branch), m * float(b.rng.uniform(0.018, 0.026)), 0.34 * s.naturalization, b.rng)
        b.carve_variable_corridor_level(branch, 760, widths2, m * 0.070, 58, m * 0.005 * s.naturalization)
        _path_side_ramps(b, branch, 1, m * 0.020, m * 0.115, m * 0.0075, m * 0.014)
    _path_side_ramps(b, trunk, 5 + int(3 * s.feature_density), m * 0.025, m * 0.135, m * 0.0085, m * 0.015)
    _masked_fbm(b, 235, m * 0.10, m * 0.28, 0.34, ridged=True, softness=0.46)
    b.add_random_craters(3 + int(5 * s.feature_density), (m * 0.012, m * 0.032), 0.95, 0.28)
    b.add_detail(6 * s.detail, m * 0.046)
    _repair_connectivity(b, 36, 0.95, 8, m * 0.009, m * 0.016)
    return b.finalize(center_height=2480, preserve_flats=True)


def callisto_craterlands(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(720)
    m = min(b.h, b.w)
    b.add_fbm(100, m * 0.50, ridged=False, octaves=3)
    for count, rrange, depth, rim in [(2 + int(3 * s.feature_density), (m * 0.075, m * 0.135), 0.66, 0.29), (8 + int(10 * s.feature_density), (m * 0.028, m * 0.070), 0.80, 0.31), (18 + int(22 * s.feature_density), (m * 0.009, m * 0.030), 0.90, 0.28)]:
        b.add_random_craters(count, rrange, depth, rim)
    yy, xx = np.mgrid[0:b.h, 0:b.w].astype(np.float32)
    for _ in range(3):
        cx = float(b.rng.uniform(m * 0.14, b.w - m * 0.14))
        cy = float(b.rng.uniform(m * 0.14, b.h - m * 0.14))
        rad = m * float(b.rng.uniform(0.10, 0.18))
        rr = np.hypot(xx - cx, yy - cy) / max(rad, 1)
        b.a += (np.exp(-0.5 * ((rr - 1) / 0.23) ** 2) * float(b.rng.uniform(48, 85))).astype(np.float32)
    b.smooth(0.72, 0.34)
    b.add_detail(2.5 * s.detail, m * 0.060)
    _repair_connectivity(b, 38, 0.96, 2, m * 0.008, m * 0.014)
    return b.finalize(center_height=1180, preserve_flats=False)


def titan_basin_network(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(850)
    m = min(b.h, b.w)
    yy, xx = np.mgrid[0:b.h, 0:b.w].astype(np.float32)
    b.add_fbm(190, m * 0.52, ridged=False, octaves=3)
    basins: list[Point] = []
    for fx, fy, rx, ry in [(0.30, 0.34, 0.23, 0.18), (0.67, 0.40, 0.26, 0.21), (0.50, 0.72, 0.28, 0.19)]:
        cx = b.w * (fx + float(b.rng.uniform(-0.04, 0.04)))
        cy = b.h * (fy + float(b.rng.uniform(-0.04, 0.04)))
        rr = np.sqrt(((xx - cx) / max(b.w * rx, 1)) ** 2 + ((yy - cy) / max(b.h * ry, 1)) ** 2)
        b.a -= (np.exp(-0.5 * (rr / 0.78) ** 4) * float(b.rng.uniform(155, 235))).astype(np.float32)
        basins.append((cx, cy))
    _masked_fbm(b, 190, m * 0.21, m * 0.42, 0.30, ridged=False, softness=0.70)
    # Preserve exact staging floors inside each basin without flattening the network.
    for cx, cy in basins:
        b.stamp_blob_shelf(target=float(np.mean(b.a[int(cy - m * 0.08) : int(cy + m * 0.08), int(cx - m * 0.08) : int(cx + m * 0.08)])), area_fraction=0.04, feature_px=m * 0.14, feather=m * 0.016, warp_px=m * 0.012 * s.naturalization)
    anchors = [_edge_point(b, "left", float(b.rng.uniform(0.35, 0.70))), *basins, _edge_point(b, "right", float(b.rng.uniform(0.30, 0.68)))]
    for ia, ib in [(0, 1), (1, 2), (2, 3), (3, 4), (1, 3)]:
        b.carve_path(meander_path(anchors[ia], anchors[ib], 13, m * 0.095 * s.naturalization, b.rng), float(b.rng.uniform(30, 58)), m * 0.032, m * 0.090, float(b.rng.uniform(0, 6)))
    for _ in range(6 + int(5 * s.feature_density)):
        cx = float(b.rng.uniform(m * 0.10, b.w - m * 0.10))
        cy = float(b.rng.uniform(m * 0.10, b.h - m * 0.10))
        rad = m * float(b.rng.uniform(0.028, 0.060))
        b.a += (np.exp(-0.5 * (np.hypot(xx - cx, yy - cy) / max(rad, 1) / 0.90) ** 2) * float(b.rng.uniform(65, 145))).astype(np.float32)
    b.add_random_craters(2 + int(4 * s.feature_density), (m * 0.015, m * 0.035), 0.68, 0.20)
    b.add_detail(3 * s.detail, m * 0.060)
    _repair_connectivity(b, 38, 0.96, 2, m * 0.008, m * 0.014)
    return b.finalize(center_height=1320, preserve_flats=True)


def europa_fracture_plains(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(720)
    m = min(b.h, b.w)
    b.add_fbm(68, m * 0.54, ridged=False, octaves=3)
    _masked_fbm(b, 70, m * 0.22, m * 0.42, 0.38, ridged=False, softness=0.82)
    for i in range(4 + int(s.feature_density > 0.68)):
        sx = float(b.rng.uniform(m * 0.06, b.w - m * 0.06))
        sy = float(b.rng.uniform(m * 0.06, b.h - m * 0.06))
        ang = float(b.rng.uniform(0, math.tau))
        length = float(b.rng.uniform(m * 0.24, m * 0.55))
        ex = float(np.clip(sx + math.cos(ang) * length, m * 0.05, b.w - m * 0.05))
        ey = float(np.clip(sy + math.sin(ang) * length, m * 0.05, b.h - m * 0.05))
        path = meander_path((sx, sy), (ex, ey), 11, m * 0.085 * s.naturalization, b.rng)
        split = int(b.rng.integers(4, 7))
        chunks = [path[j : j + split] for j in range(0, len(path) - 1, split - 1)]
        for chunk in chunks:
            if len(chunk) < 2 or b.rng.random() < 0.28:
                continue
            if i % 2:
                b.add_ridge_path(chunk, float(b.rng.uniform(24, 42)), m * 0.004, m * 0.022)
                if b.rng.random() < 0.55:
                    b.carve_path(chunk, float(b.rng.uniform(8, 16)), m * 0.0025, m * 0.014, 0)
            else:
                b.carve_path(chunk, float(b.rng.uniform(22, 42)), m * 0.005, m * 0.027, float(b.rng.uniform(7, 16)))
    yy, xx = np.mgrid[0:b.h, 0:b.w].astype(np.float32)
    for _ in range(3):
        cx = float(b.rng.uniform(m * 0.15, b.w - m * 0.15))
        cy = float(b.rng.uniform(m * 0.15, b.h - m * 0.15))
        rx = m * float(b.rng.uniform(0.07, 0.12))
        ry = m * float(b.rng.uniform(0.05, 0.10))
        rr = np.sqrt(((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2)
        b.a += (np.exp(-0.5 * (rr / 0.78) ** 4) * float(b.rng.uniform(-38, 42))).astype(np.float32)
    b.add_random_craters(1 + int(2 * s.feature_density), (m * 0.012, m * 0.028), 0.62, 0.18)
    b.add_detail(2 * s.detail, m * 0.065)
    _repair_connectivity(b, 38, 0.97, 2, m * 0.008, m * 0.014)
    return b.finalize(center_height=1120, preserve_flats=True)


PLANETARY_RECIPES: Dict[str, Callable[[GeneratorSettings], HG2Map]] = {
    "Pluto Basin": pluto_basin,
    "Venus Shield": venus_shield,
    "Lunar Catena": lunar_catena,
    "Mars Rift": mars_rift,
    "Callisto Craterlands": callisto_craterlands,
    "Titan Basin Network": titan_basin_network,
    "Europa Fracture Plains": europa_fracture_plains,
}
