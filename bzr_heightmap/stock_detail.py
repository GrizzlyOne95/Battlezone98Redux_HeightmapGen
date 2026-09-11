from __future__ import annotations

from dataclasses import dataclass
import math
import zlib

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .hg2 import HG2Map, HG2_SAFE_MAX_HEIGHT
from .noise import fbm, smoothstep01
from .settings import GeneratorSettings

HORIZONTAL_SAMPLE_SPACING = 5.0
VERTICAL_UNIT_SCALE = 0.1
REFERENCE_AREA = 768 * 768


@dataclass(frozen=True)
class StockDetailProfile:
    target_map_fraction: float
    target_passable_fraction: float
    max_connectors: int
    crater_count: int
    knoll_count: int
    bowl_count: int
    ledge_count: int
    ramp_count: int
    meso_amplitude: float
    fine_amplitude: float


STOCK_DETAIL_PROFILES: dict[str, StockDetailProfile] = {
    "Terraced Labyrinth": StockDetailProfile(0.38, 0.66, 3, 4, 7, 4, 14, 8, 14, 8),
    "Cratered Divide": StockDetailProfile(0.46, 0.70, 4, 24, 10, 8, 8, 10, 16, 10),
    "Ravine Network": StockDetailProfile(0.46, 0.70, 4, 10, 14, 14, 12, 12, 18, 10),
    "Mountain Basin": StockDetailProfile(0.44, 0.68, 5, 8, 20, 10, 12, 14, 17, 10),
    "Radial Badlands": StockDetailProfile(0.46, 0.70, 4, 16, 14, 10, 10, 12, 16, 9),
    "Ridged Wastes": StockDetailProfile(0.38, 0.66, 6, 4, 9, 5, 6, 14, 11, 6),
    "Serpentine Canyon": StockDetailProfile(0.48, 0.72, 3, 12, 12, 8, 10, 10, 17, 10),
    "Natural Badlands": StockDetailProfile(0.48, 0.72, 4, 14, 18, 12, 12, 14, 20, 12),
}

STOCK_DETAIL_STYLES = frozenset(STOCK_DETAIL_PROFILES)


def _slope_degrees(heightmap: np.ndarray) -> np.ndarray:
    a = np.asarray(heightmap, dtype=np.float32)
    gy, gx = np.gradient(
        a * VERTICAL_UNIT_SCALE,
        HORIZONTAL_SAMPLE_SPACING,
        HORIZONTAL_SAMPLE_SPACING,
    )
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


def stock_connectivity_metrics(heightmap: np.ndarray, max_slope_deg: float = 15.0) -> dict[str, float]:
    a = np.asarray(heightmap, dtype=np.float32)
    passable = _slope_degrees(a) <= float(max_slope_deg)
    labels, n = ndimage.label(passable, structure=np.ones((3, 3), dtype=np.uint8))
    passable_count = int(np.count_nonzero(passable))
    if n == 0 or passable_count == 0:
        return {
            "passable_fraction": 0.0,
            "largest_map_fraction": 0.0,
            "largest_passable_fraction": 0.0,
            "components": 0.0,
        }
    counts = np.bincount(labels.ravel())[1:]
    largest = int(np.max(counts)) if counts.size else 0
    return {
        "passable_fraction": float(passable_count) / float(a.size),
        "largest_map_fraction": float(largest) / float(a.size),
        "largest_passable_fraction": float(largest) / float(passable_count),
        "components": float(n),
    }


def _path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    delta = np.diff(points, axis=0)
    return float(np.sum(np.hypot(delta[:, 0], delta[:, 1])))


def _make_graded_path(
    start: tuple[float, float],
    end: tuple[float, float],
    required_length: float,
    shape: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    y0, x0 = start
    y1, x1 = end
    dy, dx = y1 - y0, x1 - x0
    direct = max(math.hypot(dy, dx), 1.0)
    uy, ux = dy / direct, dx / direct
    py, px = -ux, uy
    count = max(96, int(direct * 3.0))
    t = np.linspace(0.0, 1.0, count, dtype=np.float32)
    phase = float(rng.uniform(0.0, math.tau))
    cycles = int(rng.integers(2, 5))
    max_amp = min(shape) * 0.16
    amplitude = 0.0
    if required_length > direct * 1.03:
        amplitude = min(max_amp, max(6.0, math.sqrt(max(required_length * required_length - direct * direct, 0.0)) / max(cycles * 2.2, 1.0)))

    best = None
    for _ in range(6):
        base_y = y0 + dy * t
        base_x = x0 + dx * t
        taper = np.sin(np.pi * t)
        wave = np.sin(t * math.tau * cycles + phase)
        offset = amplitude * taper * wave
        yy = np.clip(base_y + py * offset, 1.0, shape[0] - 2.0)
        xx = np.clip(base_x + px * offset, 1.0, shape[1] - 2.0)
        points = np.column_stack((yy, xx)).astype(np.float32)
        best = points
        if _path_length(points) >= required_length * 0.97 or amplitude >= max_amp:
            break
        amplitude = min(max_amp, max(6.0, amplitude * 1.35 + 3.0))
    return best if best is not None else np.asarray([[y0, x0], [y1, x1]], dtype=np.float32)


def _apply_connector(
    a: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    rng: np.random.Generator,
    half_width: float = 4.0,
    bank: float = 11.0,
    max_grade_deg: float = 10.5,
) -> np.ndarray:
    y0, x0 = start
    y1, x1 = end
    h0 = float(np.median(a[max(0, y0 - 2) : y0 + 3, max(0, x0 - 2) : x0 + 3]))
    h1 = float(np.median(a[max(0, y1 - 2) : y1 + 3, max(0, x1 - 2) : x1 + 3]))
    direct = max(math.hypot(y1 - y0, x1 - x0), 1.0)
    max_step_per_sample = math.tan(math.radians(max_grade_deg)) * HORIZONTAL_SAMPLE_SPACING / VERTICAL_UNIT_SCALE
    required = max(direct, abs(h1 - h0) / max(max_step_per_sample, 1e-3) * 1.10)
    path = _make_graded_path((float(y0), float(x0)), (float(y1), float(x1)), required, a.shape, rng)

    seg = np.diff(path, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1]) if len(path) > 1 else np.asarray([], dtype=np.float32)
    cumulative = np.concatenate(([0.0], np.cumsum(seg_len)))
    total = max(float(cumulative[-1]), 1.0)
    path_targets = h0 + (h1 - h0) * (cumulative / total)

    seed_mask = np.zeros(a.shape, dtype=bool)
    seed_target = np.zeros(a.shape, dtype=np.float32)
    draw_img = Image.new("1", (a.shape[1], a.shape[0]), 0)
    draw = ImageDraw.Draw(draw_img)
    xy = [(float(x), float(y)) for y, x in path]
    if len(xy) > 1:
        draw.line(xy, fill=1, width=1, joint="curve")
    else:
        draw.point(xy[0], fill=1)
    path_mask = np.asarray(draw_img, dtype=bool)

    for (y, x), target in zip(path, path_targets):
        yi = int(round(float(y)))
        xi = int(round(float(x)))
        yi = min(max(yi, 0), a.shape[0] - 1)
        xi = min(max(xi, 0), a.shape[1] - 1)
        seed_mask[yi, xi] = True
        seed_target[yi, xi] = float(target)

    if not np.any(seed_mask):
        return np.zeros_like(a, dtype=bool)

    _, seed_indices = ndimage.distance_transform_edt(~seed_mask, return_indices=True)
    interpolated_target = seed_target[seed_indices[0], seed_indices[1]]
    distance, _ = ndimage.distance_transform_edt(~path_mask, return_indices=True)
    weight = np.zeros_like(a, dtype=np.float32)
    floor = distance <= float(half_width)
    transition = (distance > float(half_width)) & (distance < float(half_width + bank))
    weight[floor] = 1.0
    if np.any(transition):
        t = (distance[transition] - float(half_width)) / max(float(bank), 1e-3)
        weight[transition] = 1.0 - smoothstep01(t)
    a[:] = a * (1.0 - weight) + interpolated_target * weight
    return distance <= float(half_width + bank * 0.45)


def repair_stock_connectivity(
    heightmap: np.ndarray,
    rng: np.random.Generator,
    target_map_fraction: float = 0.46,
    target_passable_fraction: float = 0.70,
    max_connectors: int = 4,
    max_slope_deg: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(heightmap, dtype=np.float32).copy()
    protected = np.zeros_like(a, dtype=bool)
    min_component_size = max(64, int(a.size * 0.004))

    for _ in range(max(0, int(max_connectors))):
        passable = _slope_degrees(a) <= float(max_slope_deg)
        labels, n = ndimage.label(passable, structure=np.ones((3, 3), dtype=np.uint8))
        if n <= 1:
            break
        counts = np.bincount(labels.ravel())[1:]
        passable_count = int(np.count_nonzero(passable))
        if counts.size == 0 or passable_count == 0:
            break
        largest_label = int(np.argmax(counts)) + 1
        largest_count = int(counts[largest_label - 1])
        largest_map_fraction = float(largest_count) / float(a.size)
        largest_passable_fraction = float(largest_count) / float(passable_count)
        if largest_map_fraction >= float(target_map_fraction) and largest_passable_fraction >= float(target_passable_fraction):
            break

        main_mask = labels == largest_label
        dist_to_main, nearest_main = ndimage.distance_transform_edt(~main_mask, return_indices=True)
        other_labels = [i + 1 for i in np.argsort(counts)[::-1] if i + 1 != largest_label and int(counts[i]) >= min_component_size]
        if not other_labels:
            break

        chosen = None
        for label_id in other_labels[:8]:
            ys, xs = np.nonzero(labels == label_id)
            if ys.size == 0:
                continue
            d = dist_to_main[ys, xs]
            sample_indices = np.arange(ys.size)
            if ys.size > 12000:
                sample_indices = rng.choice(ys.size, size=12000, replace=False)
            sy = ys[sample_indices]
            sx = xs[sample_indices]
            nearest_y = nearest_main[0, sy, sx]
            nearest_x = nearest_main[1, sy, sx]
            height_cost = np.abs(a[sy, sx] - a[nearest_y, nearest_x]) / 10.0
            score = dist_to_main[sy, sx] + height_cost * 0.35
            j = int(np.argmin(score))
            candidate = (
                float(score[j]) / max(math.sqrt(float(counts[label_id - 1])), 1.0),
                (int(sy[j]), int(sx[j])),
                (int(nearest_y[j]), int(nearest_x[j])),
            )
            if chosen is None or candidate[0] < chosen[0]:
                chosen = candidate
        if chosen is None:
            break
        _, start, end = chosen
        protected |= _apply_connector(a, start, end, rng)

    return a, protected


def _patch_bounds(shape: tuple[int, int], cy: float, cx: float, radius: float) -> tuple[slice, slice, np.ndarray, np.ndarray]:
    margin = max(2, int(math.ceil(radius + 3.0)))
    y0 = max(0, int(round(cy)) - margin)
    y1 = min(shape[0], int(round(cy)) + margin + 1)
    x0 = max(0, int(round(cx)) - margin)
    x1 = min(shape[1], int(round(cx)) + margin + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    return slice(y0, y1), slice(x0, x1), yy - float(cy), xx - float(cx)


def _add_crater(a: np.ndarray, cy: float, cx: float, radius: float, depth: float, rim_height: float, ellipse: float) -> None:
    sy, sx, yy, xx = _patch_bounds(a.shape, cy, cx, radius * 1.6)
    ex = max(radius * ellipse, 1.0)
    ey = max(radius / max(ellipse, 1e-3), 1.0)
    r = np.sqrt((xx / ex) ** 2 + (yy / ey) ** 2)
    bowl = np.clip(1.0 - r, 0.0, 1.0)
    bowl = bowl * bowl * (3.0 - 2.0 * bowl)
    rim = np.exp(-0.5 * ((r - 1.02) / 0.14) ** 2)
    rim[r > 1.45] = 0.0
    a[sy, sx] += -float(depth) * bowl + float(rim_height) * rim


def _add_gaussian_feature(a: np.ndarray, cy: float, cx: float, radius: float, amplitude: float, ellipse: float, angle: float) -> None:
    sy, sx, yy, xx = _patch_bounds(a.shape, cy, cx, radius * 1.8)
    c, s = math.cos(angle), math.sin(angle)
    u = xx * c + yy * s
    v = -xx * s + yy * c
    rx = max(radius * ellipse, 1.0)
    ry = max(radius / max(ellipse, 1e-3), 1.0)
    r2 = (u / rx) ** 2 + (v / ry) ** 2
    a[sy, sx] += float(amplitude) * np.exp(-0.5 * r2)


def _add_ledge(a: np.ndarray, cy: float, cx: float, rx: float, ry: float, offset: float, angle: float) -> None:
    radius = max(rx, ry) * 1.6
    sy, sx, yy, xx = _patch_bounds(a.shape, cy, cx, radius)
    c, s = math.cos(angle), math.sin(angle)
    u = xx * c + yy * s
    v = -xx * s + yy * c
    r = np.maximum(np.abs(u) / max(rx, 1.0), np.abs(v) / max(ry, 1.0))
    weight = 1.0 - smoothstep01(np.clip((r - 0.68) / 0.42, 0.0, 1.0))
    area = a[sy, sx]
    core = r <= 0.58
    base = float(np.median(area[core])) if np.any(core) else float(np.median(area))
    target = base + float(offset)
    area[:] = area * (1.0 - weight) + target * weight


def _add_small_ramp(a: np.ndarray, cy: float, cx: float, length: float, width: float, rise: float, angle: float) -> None:
    radius = math.hypot(length, width) * 0.75 + 3.0
    sy, sx, yy, xx = _patch_bounds(a.shape, cy, cx, radius)
    c, s = math.cos(angle), math.sin(angle)
    u = xx * c + yy * s
    v = -xx * s + yy * c
    t = np.clip((u + length * 0.5) / max(length, 1.0), 0.0, 1.0)
    along = smoothstep01(t) * 2.0 - 1.0
    side = 1.0 - smoothstep01(np.clip(np.abs(v) / max(width, 1.0), 0.0, 1.0))
    end = np.sin(np.pi * t)
    end = np.sqrt(np.clip(end, 0.0, 1.0))
    a[sy, sx] += float(rise) * along * side * end


def _pick_centers(
    eligible: np.ndarray,
    count: int,
    rng: np.random.Generator,
    avoid: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    mask = np.asarray(eligible, dtype=bool).copy()
    if avoid is not None:
        mask &= ~np.asarray(avoid, dtype=bool)
    ys, xs = np.nonzero(mask)
    if ys.size == 0 or count <= 0:
        return []
    count = min(int(count), int(ys.size))
    chosen = rng.choice(ys.size, size=count, replace=False)
    return [(int(ys[i]), int(xs[i])) for i in np.atleast_1d(chosen)]


def _apply_discrete_features(
    a: np.ndarray,
    protected: np.ndarray,
    profile: StockDetailProfile,
    settings: GeneratorSettings,
    rng: np.random.Generator,
) -> None:
    slope = _slope_degrees(a)
    eligible = slope <= 11.0
    edge = max(8, int(min(a.shape) * 0.012))
    eligible[:edge, :] = False
    eligible[-edge:, :] = False
    eligible[:, :edge] = False
    eligible[:, -edge:] = False
    if np.any(protected):
        protected_margin = ndimage.binary_dilation(protected, iterations=3)
    else:
        protected_margin = protected

    area_scale = max(0.20, float(a.size) / float(REFERENCE_AREA))
    density = 0.65 + 0.70 * float(np.clip(settings.feature_density, 0.0, 1.0))
    natural = 0.70 + 0.45 * float(np.clip(settings.naturalization, 0.0, 1.0))

    groups = [
        ("crater", int(round(profile.crater_count * area_scale * density))),
        ("knoll", int(round(profile.knoll_count * area_scale * density))),
        ("bowl", int(round(profile.bowl_count * area_scale * density))),
        ("ledge", int(round(profile.ledge_count * area_scale * density))),
        ("ramp", int(round(profile.ramp_count * area_scale * density))),
    ]

    for kind, count in groups:
        centers = _pick_centers(eligible, count, rng, protected_margin)
        for cy, cx in centers:
            if kind == "crater":
                radius = float(rng.uniform(4.0, 14.0) * natural)
                depth = float(rng.uniform(12.0, 42.0))
                rim = float(rng.uniform(5.0, 20.0))
                _add_crater(a, cy, cx, radius, depth, rim, float(rng.uniform(0.72, 1.38)))
            elif kind == "knoll":
                _add_gaussian_feature(
                    a,
                    cy,
                    cx,
                    float(rng.uniform(4.0, 12.0) * natural),
                    float(rng.uniform(10.0, 38.0)),
                    float(rng.uniform(0.65, 1.55)),
                    float(rng.uniform(0.0, math.tau)),
                )
            elif kind == "bowl":
                _add_gaussian_feature(
                    a,
                    cy,
                    cx,
                    float(rng.uniform(5.0, 15.0) * natural),
                    -float(rng.uniform(8.0, 32.0)),
                    float(rng.uniform(0.70, 1.45)),
                    float(rng.uniform(0.0, math.tau)),
                )
            elif kind == "ledge":
                _add_ledge(
                    a,
                    cy,
                    cx,
                    float(rng.uniform(5.0, 13.0)),
                    float(rng.uniform(3.0, 8.0)),
                    float(rng.choice([-1.0, 1.0]) * rng.uniform(12.0, 42.0)),
                    float(rng.uniform(0.0, math.tau)),
                )
            else:
                _add_small_ramp(
                    a,
                    cy,
                    cx,
                    float(rng.uniform(10.0, 24.0)),
                    float(rng.uniform(3.0, 7.0)),
                    float(rng.uniform(10.0, 34.0)),
                    float(rng.uniform(0.0, math.tau)),
                )


def _apply_layered_detail(
    a: np.ndarray,
    protected: np.ndarray,
    profile: StockDetailProfile,
    settings: GeneratorSettings,
    rng: np.random.Generator,
) -> None:
    detail_strength = float(np.clip(settings.detail, 0.0, 1.5))
    if detail_strength <= 0.0:
        return
    slope = _slope_degrees(a)
    local_max = ndimage.maximum_filter(a, size=5, mode="reflect")
    local_min = ndimage.minimum_filter(a, size=5, mode="reflect")
    flat_core = (local_max - local_min) <= 1.0
    eligible = (slope <= 24.0) & ~protected & ~flat_core
    if not np.any(eligible):
        return

    meso = fbm(a.shape, 18.0, rng, octaves=4, persistence=0.50)
    fine = fbm(a.shape, 6.0, rng, octaves=3, persistence=0.46)
    modulation = fbm(a.shape, 72.0, rng, octaves=2, persistence=0.5)
    weight = np.clip((modulation + 1.0) * 0.55, 0.15, 1.0).astype(np.float32)
    detail = (
        meso * float(profile.meso_amplitude) * detail_strength
        + fine * float(profile.fine_amplitude) * detail_strength
    ) * weight
    detail[~eligible] = 0.0
    a += detail


def enhance_stock_terrain(terrain: HG2Map, settings: GeneratorSettings, style: str) -> HG2Map:
    profile = STOCK_DETAIL_PROFILES.get(style)
    if profile is None:
        return terrain

    stable_style_seed = zlib.crc32(style.encode("utf-8")) & 0xFFFFFFFF
    seed = (int(settings.seed) ^ stable_style_seed ^ 0x5A17C0DE) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    base = np.asarray(terrain.heights, dtype=np.float32)

    repaired, protected = repair_stock_connectivity(
        base,
        rng,
        target_map_fraction=profile.target_map_fraction,
        target_passable_fraction=profile.target_passable_fraction,
        max_connectors=profile.max_connectors,
        max_slope_deg=15.0,
    )
    _apply_discrete_features(repaired, protected, profile, settings, rng)
    _apply_layered_detail(repaired, protected, profile, settings, rng)
    heights = np.clip(np.rint(repaired), 0, HG2_SAFE_MAX_HEIGHT).astype(np.uint16)
    return HG2Map(
        heights,
        terrain.zones_x,
        terrain.zones_z,
        terrain.zone_bits,
        terrain.structure_version,
        terrain.map_version,
    )
