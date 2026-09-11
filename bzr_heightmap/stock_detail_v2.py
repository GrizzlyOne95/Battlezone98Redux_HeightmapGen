from __future__ import annotations

from dataclasses import dataclass
import math
import zlib

import numpy as np
from scipy import ndimage

from .hg2 import HG2Map, HG2_SAFE_MAX_HEIGHT
from .noise import fbm, smoothstep01
from .settings import GeneratorSettings
from .stock_detail import (
    HORIZONTAL_SAMPLE_SPACING,
    REFERENCE_AREA,
    STOCK_DETAIL_STYLES,
    VERTICAL_UNIT_SCALE,
    _add_crater,
    _add_gaussian_feature,
    _add_ledge,
    _add_small_ramp,
    _slope_degrees,
)


@dataclass(frozen=True)
class NaturalDetailProfile:
    target_map_fraction: float
    max_saddles: int
    crater_count: int
    cluster_count: int
    ledge_count: int
    ramp_count: int
    meso_amplitude: float
    fine_amplitude: float
    edge_amplitude: float
    flat_detail_fraction: float


PROFILES: dict[str, NaturalDetailProfile] = {
    "Terraced Labyrinth": NaturalDetailProfile(0.30, 3, 10, 5, 24, 18, 22, 16, 18, 0.28),
    "Cratered Divide": NaturalDetailProfile(0.38, 3, 42, 7, 16, 18, 28, 19, 20, 0.30),
    "Ravine Network": NaturalDetailProfile(0.32, 1, 18, 7, 20, 20, 31, 21, 22, 0.22),
    "Mountain Basin": NaturalDetailProfile(0.32, 4, 18, 8, 20, 22, 30, 21, 24, 0.27),
    "Radial Badlands": NaturalDetailProfile(0.46, 3, 28, 7, 18, 20, 28, 19, 20, 0.28),
    "Ridged Wastes": NaturalDetailProfile(0.28, 5, 10, 5, 12, 22, 20, 14, 17, 0.16),
    "Serpentine Canyon": NaturalDetailProfile(0.40, 0, 22, 5, 14, 12, 26, 18, 20, 0.18),
    "Natural Badlands": NaturalDetailProfile(0.40, 4, 26, 9, 22, 24, 34, 24, 26, 0.32),
}


def _component_state(a: np.ndarray, max_slope_deg: float = 15.0):
    passable = _slope_degrees(a) <= float(max_slope_deg)
    labels, n = ndimage.label(passable, structure=np.ones((3, 3), dtype=np.uint8))
    counts = np.bincount(labels.ravel())[1:] if n else np.asarray([], dtype=np.int64)
    return passable, labels, counts


def _nearest_component_pair(
    a: np.ndarray,
    labels: np.ndarray,
    counts: np.ndarray,
    rng: np.random.Generator,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if counts.size < 2:
        return None
    main_label = int(np.argmax(counts)) + 1
    main_mask = labels == main_label
    distance, nearest = ndimage.distance_transform_edt(~main_mask, return_indices=True)
    min_component = max(96, int(a.size * 0.003))
    choices = [i + 1 for i in np.argsort(counts)[::-1] if i + 1 != main_label and int(counts[i]) >= min_component]
    best = None
    for label_id in choices[:10]:
        ys, xs = np.nonzero(labels == label_id)
        if ys.size == 0:
            continue
        if ys.size > 10000:
            idx = rng.choice(ys.size, size=10000, replace=False)
            ys, xs = ys[idx], xs[idx]
        ny = nearest[0, ys, xs]
        nx = nearest[1, ys, xs]
        spatial = distance[ys, xs]
        height = np.abs(a[ys, xs] - a[ny, nx])
        # Prefer nearby barriers and modest elevation gaps, but do not force
        # a connection just because a component is large.
        score = spatial + height / 18.0
        j = int(np.argmin(score))
        candidate = (float(score[j]), (int(ys[j]), int(xs[j])), (int(ny[j]), int(nx[j])))
        if best is None or candidate[0] < best[0]:
            best = candidate
    return None if best is None else (best[1], best[2])


def _stamp_organic_saddle(
    a: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    rng: np.random.Generator,
    max_grade_deg: float = 10.0,
) -> np.ndarray:
    y0, x0 = start
    y1, x1 = end
    h0 = float(np.median(a[max(0, y0 - 3) : y0 + 4, max(0, x0 - 3) : x0 + 4]))
    h1 = float(np.median(a[max(0, y1 - 3) : y1 + 4, max(0, x1 - 3) : x1 + 4]))
    dy, dx = float(y1 - y0), float(x1 - x0)
    direct = max(math.hypot(dy, dx), 1.0)
    uy, ux = dy / direct, dx / direct
    py, px = -ux, uy

    max_step = math.tan(math.radians(max_grade_deg)) * HORIZONTAL_SAMPLE_SPACING / VERTICAL_UNIT_SCALE
    needed = abs(h1 - h0) / max(max_step, 1e-3)
    length = max(direct + 30.0, needed + 22.0)
    length = min(length, min(a.shape) * 0.38)
    half = length * 0.5
    cy = (y0 + y1) * 0.5
    cx = (x0 + x1) * 0.5
    core_width = float(rng.uniform(7.0, 12.0))
    bank_width = float(rng.uniform(14.0, 24.0))

    margin = int(math.ceil(half + bank_width + 8.0))
    y_min = max(0, int(math.floor(cy - margin)))
    y_max = min(a.shape[0], int(math.ceil(cy + margin + 1)))
    x_min = max(0, int(math.floor(cx - margin)))
    x_max = min(a.shape[1], int(math.ceil(cx + margin + 1)))
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max].astype(np.float32)
    rel_y, rel_x = yy - cy, xx - cx
    u = rel_y * uy + rel_x * ux
    v = rel_y * py + rel_x * px

    # Warp the banks, not the longitudinal grade. This reads as a broad eroded
    # saddle/pass rather than a perfectly straight painted ramp.
    warp_shape = u.shape
    warp_rng = np.random.default_rng(int(rng.integers(1, 2**31 - 1)))
    bank_warp = fbm(warp_shape, max(18.0, bank_width * 1.7), warp_rng, octaves=3, persistence=0.52) * 3.0
    v_eff = np.abs(v + bank_warp)

    lateral = np.ones_like(v_eff, dtype=np.float32)
    outer = v_eff > core_width
    lateral[outer] = 1.0 - smoothstep01((v_eff[outer] - core_width) / max(bank_width, 1.0))
    lateral[v_eff >= core_width + bank_width] = 0.0

    end_feather = 14.0
    longitudinal = np.ones_like(u, dtype=np.float32)
    end_zone = np.abs(u) > (half - end_feather)
    longitudinal[end_zone] = 1.0 - smoothstep01((np.abs(u[end_zone]) - (half - end_feather)) / end_feather)
    longitudinal[np.abs(u) >= half] = 0.0
    weight = lateral * longitudinal

    t = np.clip((u + half) / max(length, 1.0), 0.0, 1.0)
    target = h0 + (h1 - h0) * t
    area = a[y_min:y_max, x_min:x_max]
    area[:] = area * (1.0 - weight) + target * weight

    protected = np.zeros_like(a, dtype=bool)
    protected[y_min:y_max, x_min:x_max] = weight >= 0.70
    return protected


def repair_with_saddles(
    heightmap: np.ndarray,
    rng: np.random.Generator,
    target_map_fraction: float,
    max_saddles: int,
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(heightmap, dtype=np.float32).copy()
    protected = np.zeros_like(a, dtype=bool)
    for _ in range(max(0, int(max_saddles))):
        passable, labels, counts = _component_state(a, 15.0)
        if counts.size == 0:
            break
        largest_map_fraction = float(np.max(counts)) / float(a.size)
        if largest_map_fraction >= float(target_map_fraction):
            break
        pair = _nearest_component_pair(a, labels, counts, rng)
        if pair is None:
            break
        new_protected = _stamp_organic_saddle(a, pair[0], pair[1], rng)
        protected |= new_protected
    return a, protected


def _eligible_centers(a: np.ndarray, protected: np.ndarray) -> np.ndarray:
    slope = _slope_degrees(a)
    eligible = slope <= 14.0
    border = max(10, int(min(a.shape) * 0.012))
    eligible[:border] = False
    eligible[-border:] = False
    eligible[:, :border] = False
    eligible[:, -border:] = False
    if np.any(protected):
        eligible &= ~ndimage.binary_dilation(protected, iterations=4)
    return eligible


def _random_centers(mask: np.ndarray, count: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0 or count <= 0:
        return []
    count = min(int(count), int(ys.size))
    indices = rng.choice(ys.size, size=count, replace=False)
    return [(int(ys[i]), int(xs[i])) for i in np.atleast_1d(indices)]


def _clustered_micro_features(
    a: np.ndarray,
    protected: np.ndarray,
    profile: NaturalDetailProfile,
    settings: GeneratorSettings,
    rng: np.random.Generator,
) -> None:
    eligible = _eligible_centers(a, protected)
    area_scale = max(0.08, float(a.size) / float(REFERENCE_AREA))
    density = 0.65 + 0.85 * float(np.clip(settings.feature_density, 0.0, 1.0))
    natural = 0.75 + 0.40 * float(np.clip(settings.naturalization, 0.0, 1.0))

    major_craters = int(round(profile.crater_count * area_scale * density))
    for cy, cx in _random_centers(eligible, major_craters, rng):
        radius = float(rng.uniform(3.5, 12.0) * natural)
        _add_crater(
            a,
            cy,
            cx,
            radius,
            float(rng.uniform(14.0, 48.0)),
            float(rng.uniform(5.0, 22.0)),
            float(rng.uniform(0.65, 1.45)),
        )

    clusters = int(round(profile.cluster_count * area_scale * density))
    anchors = _random_centers(eligible, clusters, rng)
    for ay, ax in anchors:
        members = int(rng.integers(6, 15))
        spread = float(rng.uniform(18.0, 48.0))
        for _ in range(members):
            cy = int(round(ay + rng.normal(0.0, spread * 0.42)))
            cx = int(round(ax + rng.normal(0.0, spread * 0.42)))
            if cy < 2 or cx < 2 or cy >= a.shape[0] - 2 or cx >= a.shape[1] - 2 or not eligible[cy, cx]:
                continue
            choice = float(rng.random())
            if choice < 0.36:
                radius = float(rng.uniform(2.2, 6.8))
                _add_crater(a, cy, cx, radius, float(rng.uniform(7.0, 26.0)), float(rng.uniform(2.0, 10.0)), float(rng.uniform(0.72, 1.36)))
            elif choice < 0.70:
                _add_gaussian_feature(a, cy, cx, float(rng.uniform(2.5, 7.5)), float(rng.uniform(7.0, 30.0)), float(rng.uniform(0.65, 1.55)), float(rng.uniform(0.0, math.tau)))
            else:
                _add_gaussian_feature(a, cy, cx, float(rng.uniform(3.0, 8.0)), -float(rng.uniform(6.0, 24.0)), float(rng.uniform(0.70, 1.45)), float(rng.uniform(0.0, math.tau)))

    ledges = int(round(profile.ledge_count * area_scale * density))
    for cy, cx in _random_centers(eligible, ledges, rng):
        _add_ledge(
            a,
            cy,
            cx,
            float(rng.uniform(4.5, 12.0)),
            float(rng.uniform(2.5, 7.0)),
            float(rng.choice([-1.0, 1.0]) * rng.uniform(10.0, 42.0)),
            float(rng.uniform(0.0, math.tau)),
        )

    ramps = int(round(profile.ramp_count * area_scale * density))
    for cy, cx in _random_centers(eligible, ramps, rng):
        _add_small_ramp(
            a,
            cy,
            cx,
            float(rng.uniform(9.0, 24.0)),
            float(rng.uniform(3.0, 7.5)),
            float(rng.uniform(9.0, 34.0)),
            float(rng.uniform(0.0, math.tau)),
        )


def _layered_surface_detail(
    a: np.ndarray,
    protected: np.ndarray,
    profile: NaturalDetailProfile,
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

    modulation = fbm(a.shape, 80.0, rng, octaves=3, persistence=0.52)
    # Only a style-specific portion of authored flats receives small-scale
    # terrain events; large staging surfaces remain truly exact elsewhere.
    threshold = float(np.quantile(modulation[flat_core], 1.0 - profile.flat_detail_fraction)) if np.any(flat_core) else 1.0
    active_flat = flat_core & (modulation >= threshold)
    active = (slope <= 25.0) & ~protected & (~flat_core | active_flat)

    meso = fbm(a.shape, 18.0, rng, octaves=4, persistence=0.52)
    fine = fbm(a.shape, 5.0, rng, octaves=3, persistence=0.47)
    patch_weight = np.clip((modulation + 0.75) / 1.35, 0.18, 1.0)
    detail = (
        meso * float(profile.meso_amplitude) * detail_strength
        + fine * float(profile.fine_amplitude) * detail_strength
    ) * patch_weight
    detail[~active] = 0.0
    a += detail

    # Hand-authored maps frequently carry broken cliff lips and short local
    # escarpment texture. Keep this on slopes so it raises local curvature
    # without roughening the broad traversable flats.
    slope_after = _slope_degrees(a)
    edge_mask = (slope_after >= 16.0) & (slope_after <= 58.0) & ~protected
    edge_noise = fbm(a.shape, 7.0, rng, octaves=3, persistence=0.48)
    edge_noise += 0.45 * fbm(a.shape, 3.5, rng, octaves=2, persistence=0.45)
    edge_strength = np.clip((slope_after - 16.0) / 22.0, 0.0, 1.0)
    a += edge_noise * float(profile.edge_amplitude) * detail_strength * edge_strength * edge_mask


def enhance_stock_terrain(terrain: HG2Map, settings: GeneratorSettings, style: str) -> HG2Map:
    profile = PROFILES.get(style)
    if profile is None:
        return terrain

    seed = (int(settings.seed) ^ (zlib.crc32(style.encode("utf-8")) & 0xFFFFFFFF) ^ 0x37A1D5E9) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    a, protected = repair_with_saddles(
        np.asarray(terrain.heights, dtype=np.float32),
        rng,
        target_map_fraction=profile.target_map_fraction,
        max_saddles=profile.max_saddles,
    )
    _clustered_micro_features(a, protected, profile, settings, rng)
    _layered_surface_detail(a, protected, profile, settings, rng)

    # One final broad saddle is allowed on the non-canyon styles if the detail
    # pass fragmented a formerly useful route. Deliberate ravines/canyons are
    # not bridged merely to improve a global statistic.
    if style not in {"Serpentine Canyon", "Ravine Network"}:
        passable, labels, counts = _component_state(a, 15.0)
        largest = float(np.max(counts)) / float(a.size) if counts.size else 0.0
        if largest < profile.target_map_fraction * 0.82:
            a, final_protected = repair_with_saddles(a, rng, profile.target_map_fraction * 0.82, 1)
            protected |= final_protected

    heights = np.clip(np.rint(a), 0, HG2_SAFE_MAX_HEIGHT).astype(np.uint16)
    return HG2Map(
        heights,
        terrain.zones_x,
        terrain.zones_z,
        terrain.zone_bits,
        terrain.structure_version,
        terrain.map_version,
    )


__all__ = ["PROFILES", "STOCK_DETAIL_STYLES", "enhance_stock_terrain", "repair_with_saddles"]
