from __future__ import annotations

from dataclasses import dataclass
import math
import zlib

import numpy as np
from scipy import ndimage

from .hg2 import HG2Map, HG2_SAFE_MAX_HEIGHT
from .settings import GeneratorSettings

HORIZONTAL_SAMPLE_SPACING = 5.0
VERTICAL_UNIT_SCALE = 0.1
REFERENCE_AREA = 768 * 768


@dataclass(frozen=True)
class PlanetaryDetailProfile:
    craterlets: int
    medium_craters: int
    hummocks: int
    ejecta_lobes: int
    rough_coverage: float
    rough_amplitude: float


PLANETARY_DETAIL_PROFILES: dict[str, PlanetaryDetailProfile] = {
    # Native HG2 scale: 1 sample = 5 world units. These profiles deliberately
    # concentrate 10-80 m features into otherwise broad, traversable plains.
    "Lunar Catena": PlanetaryDetailProfile(150, 24, 54, 30, 0.24, 13.0),
    "Callisto Craterlands": PlanetaryDetailProfile(180, 34, 44, 42, 0.28, 15.0),
    "Walled Crater Basin": PlanetaryDetailProfile(120, 24, 48, 28, 0.20, 12.0),
}

PLANETARY_DETAIL_STYLES = frozenset(PLANETARY_DETAIL_PROFILES)


def _slope_degrees(a: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(
        np.asarray(a, dtype=np.float32) * VERTICAL_UNIT_SCALE,
        HORIZONTAL_SAMPLE_SPACING,
        HORIZONTAL_SAMPLE_SPACING,
    )
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


def _rng_for(settings: GeneratorSettings, style: str) -> np.random.Generator:
    style_hash = zlib.crc32(style.encode("utf-8")) & 0xFFFFFFFF
    seed = (int(settings.seed) ^ style_hash ^ 0xC0FFEE21) & 0xFFFFFFFF
    return np.random.default_rng(seed)


def _scaled_count(base: int, area: int, feature_density: float) -> int:
    area_scale = float(area) / float(REFERENCE_AREA)
    density_scale = 0.65 + 0.70 * float(np.clip(feature_density, 0.0, 1.0))
    return max(1, int(round(float(base) * area_scale * density_scale)))


def _candidate_centers(a: np.ndarray, margin: int, max_slope: float = 12.0) -> np.ndarray:
    slope = _slope_degrees(ndimage.gaussian_filter(a.astype(np.float32), 1.2))
    mask = slope <= float(max_slope)
    if margin > 0:
        mask[:margin, :] = False
        mask[-margin:, :] = False
        mask[:, :margin] = False
        mask[:, -margin:] = False
    points = np.argwhere(mask)
    if len(points):
        return points
    yy, xx = np.mgrid[margin : max(margin + 1, a.shape[0] - margin), margin : max(margin + 1, a.shape[1] - margin)]
    return np.column_stack([yy.ravel(), xx.ravel()])


def _stamp_crater(
    a: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    depth: float,
    rim_height: float,
    rng: np.random.Generator,
) -> None:
    aspect = float(rng.uniform(0.84, 1.18))
    angle = float(rng.uniform(0.0, math.tau))
    support = max(radius * 1.75, 3.0)
    x0 = max(0, int(math.floor(cx - support)))
    x1 = min(a.shape[1] - 1, int(math.ceil(cx + support)))
    y0 = max(0, int(math.floor(cy - support)))
    y1 = min(a.shape[0] - 1, int(math.ceil(cy + support)))
    yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)
    dx, dy = xx - float(cx), yy - float(cy)
    ca, sa = math.cos(angle), math.sin(angle)
    rx = (dx * ca + dy * sa) / max(radius * aspect, 1.0)
    ry = (-dx * sa + dy * ca) / max(radius / aspect, 1.0)
    rr = np.sqrt(rx * rx + ry * ry)
    bowl = -float(depth) * np.exp(-0.5 * (rr / 0.56) ** 4)
    rim = float(rim_height) * np.exp(-0.5 * ((rr - 1.0) / 0.17) ** 2)
    taper = np.clip((1.72 - rr) / 0.30, 0.0, 1.0)
    delta = (bowl + rim) * taper
    a[y0 : y1 + 1, x0 : x1 + 1] += delta.astype(np.float32)


def _stamp_hummock(
    a: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    amplitude: float,
    rng: np.random.Generator,
) -> None:
    aspect = float(rng.uniform(0.65, 1.45))
    angle = float(rng.uniform(0.0, math.tau))
    support = max(radius * 2.6, 3.0)
    x0 = max(0, int(math.floor(cx - support)))
    x1 = min(a.shape[1] - 1, int(math.ceil(cx + support)))
    y0 = max(0, int(math.floor(cy - support)))
    y1 = min(a.shape[0] - 1, int(math.ceil(cy + support)))
    yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)
    dx, dy = xx - float(cx), yy - float(cy)
    ca, sa = math.cos(angle), math.sin(angle)
    rx = (dx * ca + dy * sa) / max(radius * aspect, 1.0)
    ry = (-dx * sa + dy * ca) / max(radius / aspect, 1.0)
    rr2 = rx * rx + ry * ry
    delta = float(amplitude) * np.exp(-0.5 * rr2)
    a[y0 : y1 + 1, x0 : x1 + 1] += delta.astype(np.float32)


def _stamp_ejecta_lobe(
    a: np.ndarray,
    cx: float,
    cy: float,
    crater_radius: float,
    amplitude: float,
    rng: np.random.Generator,
) -> None:
    angle = float(rng.uniform(0.0, math.tau))
    distance = crater_radius * float(rng.uniform(1.20, 2.05))
    ex = cx + math.cos(angle) * distance
    ey = cy + math.sin(angle) * distance
    long_r = max(crater_radius * float(rng.uniform(1.0, 2.1)), 2.0)
    short_r = max(crater_radius * float(rng.uniform(0.22, 0.48)), 1.0)
    support = long_r * 2.2
    x0 = max(0, int(math.floor(ex - support)))
    x1 = min(a.shape[1] - 1, int(math.ceil(ex + support)))
    y0 = max(0, int(math.floor(ey - support)))
    y1 = min(a.shape[0] - 1, int(math.ceil(ey + support)))
    if x0 > x1 or y0 > y1:
        return
    yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)
    dx, dy = xx - ex, yy - ey
    ca, sa = math.cos(angle), math.sin(angle)
    along = (dx * ca + dy * sa) / long_r
    across = (-dx * sa + dy * ca) / short_r
    delta = float(amplitude) * np.exp(-0.5 * (along * along + across * across))
    a[y0 : y1 + 1, x0 : x1 + 1] += delta.astype(np.float32)


def _add_patchy_surface(
    a: np.ndarray,
    rng: np.random.Generator,
    coverage: float,
    amplitude: float,
) -> None:
    # A 1-3 sample high-pass surface is below the broad fBm scale used by the
    # recipes, but remains physically representable by the native HG2 mesh.
    white = rng.normal(0.0, 1.0, a.shape).astype(np.float32)
    fine = ndimage.gaussian_filter(white, 0.70, mode="reflect")
    fine -= ndimage.gaussian_filter(fine, 3.0, mode="reflect")
    std = max(float(np.std(fine)), 1e-5)
    fine /= std

    m = min(a.shape)
    field = ndimage.gaussian_filter(
        rng.normal(0.0, 1.0, a.shape).astype(np.float32),
        max(m * 0.035, 4.0),
        mode="reflect",
    )
    threshold = float(np.quantile(field, np.clip(1.0 - coverage, 0.05, 0.95)))
    patch = ndimage.gaussian_filter((field >= threshold).astype(np.float32), 3.0, mode="reflect")

    gentle = (_slope_degrees(ndimage.gaussian_filter(a, 1.5)) <= 14.0).astype(np.float32)
    gentle = ndimage.gaussian_filter(gentle, 2.5, mode="reflect")
    mask = np.clip(patch * gentle, 0.0, 1.0)
    a += fine * mask * float(amplitude)


def enhance_planetary_terrain(terrain: HG2Map, settings: GeneratorSettings, style: str) -> HG2Map:
    """Add native-scale craterland detail without replacing a recipe's macro morphology."""
    profile = PLANETARY_DETAIL_PROFILES.get(style)
    if profile is None:
        return terrain

    rng = _rng_for(settings, style)
    a = np.asarray(terrain.heights, dtype=np.float32).copy()
    m = min(a.shape)
    area = int(a.size)
    detail_scale = 0.45 + float(np.clip(settings.detail, 0.0, 1.0))

    centers = _candidate_centers(a, margin=max(8, int(m * 0.025)), max_slope=12.0)
    if len(centers) == 0:
        return terrain

    medium_sites: list[tuple[float, float, float]] = []
    for _ in range(_scaled_count(profile.medium_craters, area, settings.feature_density)):
        cy, cx = centers[int(rng.integers(0, len(centers)))]
        radius = float(rng.uniform(6.0, 17.0))
        depth = radius * float(rng.uniform(2.0, 3.9)) * detail_scale
        rim = radius * float(rng.uniform(0.8, 1.7)) * detail_scale
        _stamp_crater(a, float(cx), float(cy), radius, depth, rim, rng)
        medium_sites.append((float(cx), float(cy), radius))

    for _ in range(_scaled_count(profile.craterlets, area, settings.feature_density)):
        cy, cx = centers[int(rng.integers(0, len(centers)))]
        radius = float(rng.uniform(1.6, 5.2))
        depth = float(rng.uniform(5.0, 18.0)) * detail_scale
        rim = float(rng.uniform(3.0, 11.0)) * detail_scale
        _stamp_crater(a, float(cx), float(cy), radius, depth, rim, rng)

    for _ in range(_scaled_count(profile.hummocks, area, settings.feature_density)):
        cy, cx = centers[int(rng.integers(0, len(centers)))]
        radius = float(rng.uniform(2.0, 9.0))
        amplitude = float(rng.uniform(5.0, 24.0)) * detail_scale
        if rng.random() < 0.28:
            amplitude *= -0.70
        _stamp_hummock(a, float(cx), float(cy), radius, amplitude, rng)

    if medium_sites:
        for _ in range(_scaled_count(profile.ejecta_lobes, area, settings.feature_density)):
            cx, cy, radius = medium_sites[int(rng.integers(0, len(medium_sites)))]
            _stamp_ejecta_lobe(
                a,
                cx,
                cy,
                radius,
                float(rng.uniform(3.0, 12.0)) * detail_scale,
                rng,
            )

    _add_patchy_surface(
        a,
        rng,
        profile.rough_coverage,
        profile.rough_amplitude * detail_scale,
    )

    out = np.clip(np.rint(a), 0, HG2_SAFE_MAX_HEIGHT).astype(np.uint16)
    return HG2Map(
        out,
        terrain.zones_x,
        terrain.zones_z,
        terrain.zone_bits,
        terrain.structure_version,
        terrain.map_version,
    )
