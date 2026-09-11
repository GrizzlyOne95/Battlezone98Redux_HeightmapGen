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
class PlanetarySurfaceProfile:
    hummocks: int
    bowls: int
    lineaments: int
    lobes: int
    rough_coverage: float
    rough_amplitude: float
    mode: str


PLANETARY_SURFACE_PROFILES: dict[str, PlanetarySurfaceProfile] = {
    # These deliberately avoid applying craterland grammar to every planet.
    # Counts are calibrated for a default 3x3 / 768x768 HG2 map and scale by area.
    "Pluto Basin": PlanetarySurfaceProfile(100, 45, 10, 45, 0.15, 4.0, "basin"),
    "Venus Shield": PlanetarySurfaceProfile(130, 20, 22, 85, 0.16, 4.0, "volcanic"),
    "Titan Basin Network": PlanetarySurfaceProfile(150, 85, 16, 75, 0.16, 4.0, "basin"),
    "Europa Fracture Plains": PlanetarySurfaceProfile(55, 15, 65, 25, 0.10, 3.0, "fracture"),
}

PLANETARY_SURFACE_STYLES = frozenset(PLANETARY_SURFACE_PROFILES)


def _slope_degrees(a: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(
        np.asarray(a, dtype=np.float32) * VERTICAL_UNIT_SCALE,
        HORIZONTAL_SAMPLE_SPACING,
        HORIZONTAL_SAMPLE_SPACING,
    )
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


def _rng_for(settings: GeneratorSettings, style: str) -> np.random.Generator:
    style_hash = zlib.crc32(style.encode("utf-8")) & 0xFFFFFFFF
    seed = (int(settings.seed) ^ style_hash ^ 0x51A7FACE) & 0xFFFFFFFF
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
    yy, xx = np.mgrid[
        margin : max(margin + 1, a.shape[0] - margin),
        margin : max(margin + 1, a.shape[1] - margin),
    ]
    return np.column_stack([yy.ravel(), xx.ravel()])


def _stamp_blob(
    a: np.ndarray,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    amplitude: float,
    angle: float,
) -> None:
    support = max(float(rx), float(ry)) * 2.5
    x0 = max(0, int(math.floor(cx - support)))
    x1 = min(a.shape[1] - 1, int(math.ceil(cx + support)))
    y0 = max(0, int(math.floor(cy - support)))
    y1 = min(a.shape[0] - 1, int(math.ceil(cy + support)))
    if x0 > x1 or y0 > y1:
        return
    yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)
    dx, dy = xx - float(cx), yy - float(cy)
    ca, sa = math.cos(float(angle)), math.sin(float(angle))
    u = (dx * ca + dy * sa) / max(float(rx), 1.0)
    v = (-dx * sa + dy * ca) / max(float(ry), 1.0)
    delta = float(amplitude) * np.exp(-0.5 * (u * u + v * v))
    a[y0 : y1 + 1, x0 : x1 + 1] += delta.astype(np.float32)


def _stamp_lineament(
    a: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
    amplitude: float,
    width: float,
    mode: str,
) -> None:
    x0, y0 = p0
    x1, y1 = p1
    vx, vy = x1 - x0, y1 - y0
    length = math.hypot(vx, vy)
    if length < 2.0:
        return

    margin = max(float(width) * 5.0, 6.0)
    minx = max(0, int(math.floor(min(x0, x1) - margin)))
    maxx = min(a.shape[1] - 1, int(math.ceil(max(x0, x1) + margin)))
    miny = max(0, int(math.floor(min(y0, y1) - margin)))
    maxy = min(a.shape[0] - 1, int(math.ceil(max(y0, y1) + margin)))
    yy, xx = np.mgrid[miny : maxy + 1, minx : maxx + 1].astype(np.float32)

    t = np.clip(((xx - x0) * vx + (yy - y0) * vy) / (length * length), 0.0, 1.0)
    nx, ny = x0 + t * vx, y0 + t * vy
    signed_distance = (xx - nx) * (-vy / length) + (yy - ny) * (vx / length)
    taper = np.power(np.clip(np.sin(np.pi * t), 0.0, 1.0), 0.45)

    if mode == "fracture":
        offset = float(width) * 2.2
        core = -float(amplitude) * np.exp(-0.5 * (signed_distance / max(float(width), 0.6)) ** 2)
        side_width = max(float(width) * 0.8, 0.5)
        ridges = float(amplitude) * 0.55 * (
            np.exp(-0.5 * ((signed_distance - offset) / side_width) ** 2)
            + np.exp(-0.5 * ((signed_distance + offset) / side_width) ** 2)
        )
        profile = core + ridges
    elif mode == "volcanic":
        core_width = max(float(width) * 1.2, 0.8)
        berm_width = max(float(width) * 1.5, 1.0)
        profile = (
            -float(amplitude) * 0.60 * np.exp(-0.5 * (signed_distance / core_width) ** 2)
            + float(amplitude)
            * 0.35
            * np.exp(-0.5 * ((signed_distance - float(width) * 2.3) / berm_width) ** 2)
        )
    else:
        swale_width = max(float(width) * 2.2, 1.2)
        berm_width = max(float(width) * 2.0, 1.2)
        profile = (
            -float(amplitude) * 0.45 * np.exp(-0.5 * (signed_distance / swale_width) ** 2)
            + float(amplitude)
            * 0.25
            * np.exp(-0.5 * ((signed_distance - float(width) * 3.0) / berm_width) ** 2)
        )

    a[miny : maxy + 1, minx : maxx + 1] += (profile * taper).astype(np.float32)


def _add_broad_patch_surface(
    a: np.ndarray,
    rng: np.random.Generator,
    coverage: float,
    amplitude: float,
) -> None:
    # 2-sample smoothing keeps this in the 10m+ terrain domain. It is not
    # intended as visual noise; it supplies broad low hummocky patches that
    # remain visible at native HG2/Blender mesh scale.
    white = rng.normal(0.0, 1.0, a.shape).astype(np.float32)
    surface = ndimage.gaussian_filter(white, 2.0, mode="reflect")
    surface -= ndimage.gaussian_filter(surface, 7.0, mode="reflect")
    surface /= max(float(np.std(surface)), 1e-5)

    m = min(a.shape)
    field = ndimage.gaussian_filter(
        rng.normal(0.0, 1.0, a.shape).astype(np.float32),
        max(m * 0.050, 7.0),
        mode="reflect",
    )
    threshold = float(np.quantile(field, np.clip(1.0 - coverage, 0.05, 0.95)))
    patch = ndimage.gaussian_filter((field >= threshold).astype(np.float32), 8.0, mode="reflect")
    gentle = ndimage.gaussian_filter(
        (_slope_degrees(ndimage.gaussian_filter(a, 1.5)) <= 14.0).astype(np.float32),
        4.0,
        mode="reflect",
    )
    a += surface * np.clip(patch * gentle, 0.0, 1.0) * float(amplitude)


def enhance_planetary_surface(terrain: HG2Map, settings: GeneratorSettings, style: str) -> HG2Map:
    """Add morphology-specific native-scale detail while preserving planetary macro form."""
    profile = PLANETARY_SURFACE_PROFILES.get(style)
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

    for _ in range(_scaled_count(profile.hummocks, area, settings.feature_density)):
        cy, cx = centers[int(rng.integers(0, len(centers)))]
        radius = float(rng.uniform(2.5, 10.0))
        _stamp_blob(
            a,
            float(cx),
            float(cy),
            radius * float(rng.uniform(0.7, 1.5)),
            radius * float(rng.uniform(0.7, 1.4)),
            float(rng.uniform(4.0, 18.0)) * detail_scale,
            float(rng.uniform(0.0, math.tau)),
        )

    for _ in range(_scaled_count(profile.bowls, area, settings.feature_density)):
        cy, cx = centers[int(rng.integers(0, len(centers)))]
        radius = float(rng.uniform(3.5, 13.0))
        _stamp_blob(
            a,
            float(cx),
            float(cy),
            radius * float(rng.uniform(0.8, 1.5)),
            radius * float(rng.uniform(0.8, 1.5)),
            -float(rng.uniform(4.0, 16.0)) * detail_scale,
            float(rng.uniform(0.0, math.tau)),
        )

    for _ in range(_scaled_count(profile.lobes, area, settings.feature_density)):
        cy, cx = centers[int(rng.integers(0, len(centers)))]
        radius = float(rng.uniform(5.0, 18.0))
        amplitude = float(rng.uniform(3.0, 10.0)) * detail_scale
        if rng.random() < 0.25:
            amplitude *= -0.65
        _stamp_blob(
            a,
            float(cx),
            float(cy),
            radius * float(rng.uniform(1.3, 2.8)),
            radius * float(rng.uniform(0.45, 1.0)),
            amplitude,
            float(rng.uniform(0.0, math.tau)),
        )

    if profile.mode == "volcanic":
        smoothed = ndimage.gaussian_filter(a, 12.0, mode="reflect")
        peak_y, peak_x = np.unravel_index(int(np.argmax(smoothed)), smoothed.shape)
        for _ in range(_scaled_count(profile.lineaments, area, settings.feature_density)):
            angle = float(rng.uniform(0.0, math.tau))
            start_radius = float(rng.uniform(m * 0.03, m * 0.12))
            length = float(rng.uniform(m * 0.05, m * 0.16))
            p0 = (
                float(peak_x) + math.cos(angle) * start_radius,
                float(peak_y) + math.sin(angle) * start_radius,
            )
            bend = angle + float(rng.uniform(-0.22, 0.22))
            p1 = (p0[0] + math.cos(bend) * length, p0[1] + math.sin(bend) * length)
            _stamp_lineament(
                a,
                p0,
                p1,
                float(rng.uniform(4.0, 10.0)) * detail_scale,
                float(rng.uniform(1.0, 2.0)),
                "volcanic",
            )
    else:
        max_length = 0.12 if profile.mode == "fracture" else 0.08
        for _ in range(_scaled_count(profile.lineaments, area, settings.feature_density)):
            cy, cx = centers[int(rng.integers(0, len(centers)))]
            angle = float(rng.uniform(0.0, math.tau))
            length = float(rng.uniform(m * 0.025, m * max_length))
            p0 = (
                float(cx) - math.cos(angle) * length * 0.5,
                float(cy) - math.sin(angle) * length * 0.5,
            )
            p1 = (
                float(cx) + math.cos(angle) * length * 0.5,
                float(cy) + math.sin(angle) * length * 0.5,
            )
            _stamp_lineament(
                a,
                p0,
                p1,
                float(rng.uniform(4.0, 12.0)) * detail_scale,
                float(rng.uniform(0.9, 2.0)),
                profile.mode,
            )

    _add_broad_patch_surface(
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
