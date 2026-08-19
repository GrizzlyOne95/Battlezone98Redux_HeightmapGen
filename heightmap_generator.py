from __future__ import annotations

import argparse
import math
import os
import random
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageTk
from scipy import ndimage
from scipy.interpolate import CubicSpline

HG2_HEIGHT_MASK = 0x1FFF
HG2_MAX_HEIGHT = HG2_HEIGHT_MASK
HG2_STRUCTURE_VERSION = 1
HG2_MAP_VERSION = 10
DEFAULT_ZONE_BITS = 8
BZ_ZONE_WORLD_SIZE = 1280.0


def _smoothstep01(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _normalize01(a: np.ndarray) -> np.ndarray:
    lo = float(np.min(a))
    hi = float(np.max(a))
    if hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - lo) / (hi - lo)).astype(np.float32)


def _value_noise(shape: Tuple[int, int], feature_px: float, rng: np.random.Generator) -> np.ndarray:
    h, w = shape
    feature_px = max(2.0, float(feature_px))
    gh = max(3, int(math.ceil(h / feature_px)) + 3)
    gw = max(3, int(math.ceil(w / feature_px)) + 3)
    coarse = rng.random((gh, gw), dtype=np.float32)
    zy = h / coarse.shape[0]
    zx = w / coarse.shape[1]
    out = ndimage.zoom(coarse, (zy, zx), order=3, mode="reflect", prefilter=True)
    if out.shape[0] < h or out.shape[1] < w:
        out = np.pad(out, ((0, max(0, h - out.shape[0])), (0, max(0, w - out.shape[1]))), mode="edge")
    return _normalize01(out[:h, :w])


def _fbm(
    shape: Tuple[int, int],
    feature_px: float,
    rng: np.random.Generator,
    octaves: int = 5,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
) -> np.ndarray:
    acc = np.zeros(shape, dtype=np.float32)
    amp = 1.0
    amp_sum = 0.0
    scale = float(feature_px)
    for _ in range(max(1, int(octaves))):
        acc += (_value_noise(shape, scale, rng) * 2.0 - 1.0) * amp
        amp_sum += amp
        amp *= persistence
        scale /= lacunarity
        if scale < 2.0:
            break
    acc /= max(amp_sum, 1e-6)
    return acc


def _ridged_fbm(shape: Tuple[int, int], feature_px: float, rng: np.random.Generator, octaves: int = 5) -> np.ndarray:
    n = _fbm(shape, feature_px, rng, octaves=octaves, persistence=0.55)
    r = 1.0 - np.abs(n)
    r = r * r
    return _normalize01(r) * 2.0 - 1.0


def _warp_field(field: np.ndarray, amount_px: float, feature_px: float, rng: np.random.Generator) -> np.ndarray:
    if amount_px <= 0:
        return field
    h, w = field.shape
    wx = _fbm((h, w), feature_px, rng, octaves=3, persistence=0.55)
    wy = _fbm((h, w), feature_px, rng, octaves=3, persistence=0.55)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    coords = np.array([yy + wy * amount_px, xx + wx * amount_px], dtype=np.float32)
    return ndimage.map_coordinates(field, coords, order=1, mode="reflect").astype(np.float32)


def _polyline_distance(shape: Tuple[int, int], points: Sequence[Tuple[float, float]]) -> np.ndarray:
    h, w = shape
    img = Image.new("1", (w, h), 1)
    draw = ImageDraw.Draw(img)
    xy = [(float(x), float(y)) for x, y in points]
    if len(xy) == 1:
        draw.point(xy[0], fill=0)
    else:
        draw.line(xy, fill=0, width=1, joint="curve")
    mask = np.asarray(img, dtype=bool)
    return ndimage.distance_transform_edt(mask).astype(np.float32)


def _meander_path(
    start: Tuple[float, float],
    end: Tuple[float, float],
    count: int,
    jitter_px: float,
    rng: np.random.Generator,
) -> list[Tuple[float, float]]:
    count = max(3, int(count))
    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    length = max(math.hypot(dx, dy), 1.0)
    nx, ny = -dy / length, dx / length
    raw = rng.normal(0.0, 1.0, count).astype(np.float32)
    raw[0] = raw[-1] = 0.0
    raw = ndimage.gaussian_filter1d(raw, sigma=max(0.8, count / 12.0), mode="nearest")
    peak = float(np.max(np.abs(raw)))
    if peak > 1e-6:
        raw /= peak
    points = []
    for i, off in enumerate(raw):
        t = i / (count - 1)
        taper = math.sin(math.pi * t)
        x = sx + dx * t + nx * float(off) * jitter_px * taper
        y = sy + dy * t + ny * float(off) * jitter_px * taper
        points.append((x, y))

    # Densify with a natural cubic so canyons/ridges read as authored curves,
    # not as a chain of straight line segments.
    tt = np.arange(count, dtype=np.float32)
    dense_t = np.linspace(0.0, count - 1.0, count * 8, dtype=np.float32)
    xs = CubicSpline(tt, np.asarray([p[0] for p in points]), bc_type="natural")(dense_t)
    ys = CubicSpline(tt, np.asarray([p[1] for p in points]), bc_type="natural")(dense_t)
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def _signed_blob_field(shape: Tuple[int, int], feature_px: float, rng: np.random.Generator, warp: float) -> np.ndarray:
    field = _fbm(shape, feature_px, rng, octaves=3, persistence=0.55)
    field += 0.35 * _fbm(shape, feature_px * 0.45, rng, octaves=2, persistence=0.5)
    return _warp_field(field, warp, feature_px * 0.8, rng)


@dataclass
class HG2Map:
    heights: np.ndarray
    zones_x: int
    zones_z: int
    zone_bits: int = DEFAULT_ZONE_BITS
    structure_version: int = HG2_STRUCTURE_VERSION
    map_version: int = HG2_MAP_VERSION

    @property
    def zone_size(self) -> int:
        return 1 << self.zone_bits

    @property
    def shape(self) -> Tuple[int, int]:
        return self.zones_z * self.zone_size, self.zones_x * self.zone_size

    @property
    def world_size(self) -> Tuple[float, float]:
        return self.zones_x * BZ_ZONE_WORLD_SIZE, self.zones_z * BZ_ZONE_WORLD_SIZE

    def validate(self) -> None:
        if self.heights.shape != self.shape:
            raise ValueError(f"Height shape {self.heights.shape} does not match HG2 dimensions {self.shape}")
        if np.min(self.heights) < 0 or np.max(self.heights) > HG2_MAX_HEIGHT:
            raise ValueError(f"HG2 height samples must be 0..{HG2_MAX_HEIGHT}")

    @classmethod
    def read(cls, path: os.PathLike | str) -> "HG2Map":
        with open(path, "rb") as f:
            header = f.read(12)
            if len(header) != 12:
                raise ValueError("HG2 header is truncated")
            structure_version, zone_bits, zones_x, zones_z, map_version = struct.unpack("<HHHHI", header)
            zone_size = 1 << zone_bits
            count = zones_x * zones_z * zone_size * zone_size
            payload = f.read()
        raw = np.frombuffer(payload, dtype="<u2")
        if raw.size != count:
            raise ValueError(f"HG2 sample count mismatch: expected {count}, found {raw.size}")
        raw = raw & HG2_HEIGHT_MASK
        full = np.empty((zones_z * zone_size, zones_x * zone_size), dtype=np.uint16)
        cursor = 0
        for zz in range(zones_z):
            for zx in range(zones_x):
                zone = raw[cursor : cursor + zone_size * zone_size].reshape(zone_size, zone_size)
                full[zz * zone_size : (zz + 1) * zone_size, zx * zone_size : (zx + 1) * zone_size] = zone
                cursor += zone_size * zone_size
        return cls(full, zones_x, zones_z, zone_bits, structure_version, map_version)

    def write(self, path: os.PathLike | str) -> None:
        self.validate()
        a = np.clip(np.rint(self.heights), 0, HG2_MAX_HEIGHT).astype("<u2") & HG2_HEIGHT_MASK
        with open(path, "wb") as f:
            f.write(struct.pack("<HHHHI", self.structure_version, self.zone_bits, self.zones_x, self.zones_z, self.map_version))
            zs = self.zone_size
            for zz in range(self.zones_z):
                for zx in range(self.zones_x):
                    zone = a[zz * zs : (zz + 1) * zs, zx * zs : (zx + 1) * zs]
                    f.write(zone.astype("<u2", copy=False).tobytes(order="C"))

    def write_png16(self, path: os.PathLike | str) -> None:
        scaled = (np.clip(self.heights, 0, HG2_MAX_HEIGHT).astype(np.uint32) * 8).astype(np.uint16)
        Image.fromarray(scaled, mode="I;16").save(path)


@dataclass
class GeneratorSettings:
    zones_x: int = 3
    zones_z: int = 3
    seed: int = 1
    relief: float = 1.0
    naturalization: float = 0.65
    detail: float = 0.55
    plateau_bias: float = 0.5
    feature_density: float = 0.5
    symmetry: str = "None"
    synthetic_pads: int = 0

    @property
    def shape(self) -> Tuple[int, int]:
        zs = 1 << DEFAULT_ZONE_BITS
        return self.zones_z * zs, self.zones_x * zs


class TerrainBuilder:
    def __init__(self, settings: GeneratorSettings):
        self.settings = settings
        self.rng = np.random.default_rng(int(settings.seed) & 0xFFFFFFFF)
        self.h, self.w = settings.shape
        self.a = np.zeros((self.h, self.w), dtype=np.float32)

    def set_level(self, value: float) -> "TerrainBuilder":
        self.a.fill(float(value))
        return self

    def add_fbm(self, amplitude: float, feature_px: float, ridged: bool = False, octaves: int = 5) -> "TerrainBuilder":
        n = _ridged_fbm(self.a.shape, feature_px, self.rng, octaves) if ridged else _fbm(self.a.shape, feature_px, self.rng, octaves=octaves)
        self.a += n * float(amplitude)
        return self

    def add_terraced_blobs(
        self,
        levels: Sequence[float],
        feature_px: float,
        edge_px: float,
        threshold_bias: float = 0.0,
        warp_px: float = 0.0,
    ) -> "TerrainBuilder":
        if len(levels) < 2:
            return self
        field = _signed_blob_field(self.a.shape, feature_px, self.rng, warp_px)
        bias = np.clip(float(threshold_bias), -0.65, 0.65)
        # Quantile thresholds keep the map occupied by broad shelves instead of
        # producing sparse noise islands. Bias shifts area toward lower/higher shelves.
        probs = np.linspace(0.0, 1.0, len(levels) + 1, dtype=np.float32)[1:-1]
        probs = np.clip(probs + bias * 0.18, 0.08, 0.92)
        thresholds = np.quantile(field, probs)
        idx = np.digitize(field, thresholds)
        terraced = np.take(np.asarray(levels, dtype=np.float32), idx)
        if edge_px > 0:
            terraced = ndimage.gaussian_filter(terraced, sigma=float(edge_px), mode="reflect")
        if self.settings.naturalization > 0:
            gy, gx = np.gradient(terraced)
            edge = np.hypot(gx, gy)
            edge /= max(float(np.percentile(edge, 99)), 1e-5)
            edge = np.clip(edge, 0.0, 1.0)
            rough = _fbm(self.a.shape, max(10.0, feature_px * 0.18), self.rng, octaves=3)
            terraced += rough * edge * 22.0 * self.settings.naturalization * self.settings.relief
        self.a += terraced
        return self

    def carve_path(
        self,
        points: Sequence[Tuple[float, float]],
        depth: float,
        half_width: float,
        bank: float,
        rim: float = 0.0,
    ) -> "TerrainBuilder":
        dist = _polyline_distance(self.a.shape, points)
        inner = np.clip(dist / max(half_width, 1e-3), 0.0, 1.0)
        floor_profile = _smoothstep01(inner)
        shoulder = np.clip((dist - half_width) / max(bank, 1e-3), 0.0, 1.0)
        shoulder = _smoothstep01(shoulder)
        influence = np.where(dist <= half_width, 1.0 - 0.18 * floor_profile, 1.0 - shoulder)
        influence[dist >= half_width + bank] = 0.0
        self.a -= float(depth) * influence.astype(np.float32)
        if rim:
            sigma = max(bank * 0.35, 1.0)
            rim_profile = np.exp(-0.5 * ((dist - (half_width + bank * 0.35)) / sigma) ** 2)
            rim_profile[dist > half_width + bank * 1.5] = 0.0
            self.a += float(rim) * rim_profile.astype(np.float32)
        return self

    def add_ridge_path(
        self,
        points: Sequence[Tuple[float, float]],
        height: float,
        half_width: float,
        falloff: float,
    ) -> "TerrainBuilder":
        dist = _polyline_distance(self.a.shape, points)
        x = np.clip((dist - half_width) / max(falloff, 1e-3), 0.0, 1.0)
        profile = 1.0 - _smoothstep01(x)
        profile[dist > half_width + falloff] = 0.0
        self.a += float(height) * profile.astype(np.float32)
        return self

    def crater(
        self,
        cx: float,
        cy: float,
        radius: float,
        depth: float,
        rim_height: float,
        ellipse: float = 1.0,
    ) -> "TerrainBuilder":
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        ex = max(radius * ellipse, 1.0)
        ey = max(radius / max(ellipse, 1e-3), 1.0)
        r = np.sqrt(((xx - cx) / ex) ** 2 + ((yy - cy) / ey) ** 2)
        bowl = np.clip(1.0 - r, 0.0, 1.0)
        bowl = bowl * bowl * (3.0 - 2.0 * bowl)
        rim_sigma = 0.12
        rim = np.exp(-0.5 * ((r - 1.02) / rim_sigma) ** 2)
        rim[r > 1.45] = 0.0
        self.a -= float(depth) * bowl
        self.a += float(rim_height) * rim
        return self

    def flatten_pad(
        self,
        cx: float,
        cy: float,
        radius_x: float,
        radius_y: float,
        target: Optional[float] = None,
        feather: float = 14.0,
        rectangular: bool = False,
    ) -> "TerrainBuilder":
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        if rectangular:
            dx = np.abs(xx - cx) - radius_x
            dy = np.abs(yy - cy) - radius_y
            outside = np.hypot(np.maximum(dx, 0), np.maximum(dy, 0))
            inside = np.minimum(np.maximum(dx, dy), 0)
            sd = outside + inside
        else:
            sd = (np.sqrt(((xx - cx) / max(radius_x, 1.0)) ** 2 + ((yy - cy) / max(radius_y, 1.0)) ** 2) - 1.0) * min(radius_x, radius_y)
        weight = 1.0 - _smoothstep01((sd + feather) / max(feather * 2.0, 1e-3))
        if target is None:
            core = weight > 0.95
            target = float(np.median(self.a[core])) if np.any(core) else float(np.median(self.a))
        self.a = self.a * (1.0 - weight) + float(target) * weight
        return self

    def add_random_craters(self, count: int, radius_range: Tuple[float, float], depth_scale: float, rim_scale: float) -> "TerrainBuilder":
        margin = radius_range[1] * 1.3
        for _ in range(max(0, int(count))):
            r = float(self.rng.uniform(*radius_range))
            cx = float(self.rng.uniform(margin, max(margin + 1, self.w - margin)))
            cy = float(self.rng.uniform(margin, max(margin + 1, self.h - margin)))
            e = float(self.rng.uniform(0.8, 1.25))
            self.crater(cx, cy, r, r * depth_scale, r * rim_scale, ellipse=e)
        return self

    def add_detail(self, amplitude: float, feature_px: float) -> "TerrainBuilder":
        if amplitude <= 0:
            return self
        self.a += _fbm(self.a.shape, feature_px, self.rng, octaves=4, persistence=0.48) * float(amplitude)
        return self

    def smooth(self, sigma: float, amount: float = 1.0) -> "TerrainBuilder":
        if sigma <= 0 or amount <= 0:
            return self
        blurred = ndimage.gaussian_filter(self.a, sigma=float(sigma), mode="reflect")
        self.a = self.a * (1.0 - amount) + blurred * amount
        return self

    def apply_symmetry(self, mode: str) -> "TerrainBuilder":
        mode = (mode or "None").lower()
        if mode == "mirror x":
            self.a = (self.a + np.fliplr(self.a)) * 0.5
        elif mode == "mirror z":
            self.a = (self.a + np.flipud(self.a)) * 0.5
        elif mode == "2-way rotational":
            self.a = (self.a + np.flipud(np.fliplr(self.a))) * 0.5
        elif mode == "4-way":
            if self.a.shape[0] == self.a.shape[1]:
                self.a = (self.a + np.rot90(self.a, 1) + np.rot90(self.a, 2) + np.rot90(self.a, 3)) * 0.25
            else:
                self.a = (self.a + np.flipud(self.a) + np.fliplr(self.a) + np.flipud(np.fliplr(self.a))) * 0.25
        return self

    def finalize(self, center_height: float = 1800.0, preserve_flats: bool = True) -> HG2Map:
        relief = max(0.05, float(self.settings.relief))
        med = float(np.median(self.a))
        self.a = center_height + (self.a - med) * relief
        if self.settings.synthetic_pads > 0:
            for i in range(self.settings.synthetic_pads):
                angle = (i / max(self.settings.synthetic_pads, 1)) * math.tau + 0.35
                r = min(self.h, self.w) * 0.24
                cx = self.w * 0.5 + math.cos(angle) * r
                cy = self.h * 0.5 + math.sin(angle) * r
                self.flatten_pad(cx, cy, 28, 22, feather=9, rectangular=True)
        self.apply_symmetry(self.settings.symmetry)
        if not preserve_flats:
            self.smooth(0.55, 0.35)
        self.a = np.clip(np.rint(self.a), 0, HG2_MAX_HEIGHT).astype(np.uint16)
        return HG2Map(self.a, self.settings.zones_x, self.settings.zones_z)


def _edge_point(builder: TerrainBuilder, side: str, t: float) -> Tuple[float, float]:
    if side == "left":
        return 0.0, t * (builder.h - 1)
    if side == "right":
        return builder.w - 1.0, t * (builder.h - 1)
    if side == "top":
        return t * (builder.w - 1), 0.0
    return t * (builder.w - 1), builder.h - 1.0


def _recipe_terraced_labyrinth(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(0)
    m = min(b.h, b.w)
    plateau = np.clip(s.plateau_bias, 0, 1)
    levels = [0.0, 230.0, 500.0, 780.0] if plateau > 0.55 else [0.0, 260.0, 590.0]
    b.add_terraced_blobs(levels, m * (0.060 + 0.035 * (1 - s.feature_density)), 1.5 + 1.8 * s.naturalization,
                          threshold_bias=(plateau - 0.5) * -0.35, warp_px=m * 0.020 * s.naturalization)
    b.add_detail(5.0 * s.detail, 22.0)
    return b.finalize(center_height=850.0, preserve_flats=True)


def _recipe_cratered_divide(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(1150)
    m = min(b.h, b.w)
    b.add_fbm(420, m * 0.38, ridged=False, octaves=4)
    start = _edge_point(b, "left", 0.18)
    end = _edge_point(b, "right", 0.80)
    path = _meander_path(start, end, 12, m * 0.11 * s.naturalization, b.rng)
    b.carve_path(path, depth=900, half_width=m * 0.035, bank=m * 0.10, rim=180)
    count = int(8 + 22 * s.feature_density)
    b.add_random_craters(count, (m * 0.025, m * 0.075), 4.0, 1.0)
    b.add_detail(65 * s.detail, m * 0.035)
    return b.finalize(center_height=1900.0, preserve_flats=False)


def _recipe_ravine_network(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(2200)
    m = min(b.h, b.w)
    b.add_fbm(210, m * 0.22, ridged=True, octaves=4)
    trunk = _meander_path(_edge_point(b, "top", 0.42), _edge_point(b, "bottom", 0.58), 14,
                           m * 0.10 * s.naturalization, b.rng)
    b.carve_path(trunk, depth=1300, half_width=m * 0.028, bank=m * 0.065, rim=110)
    branches = 2 + int(3 * s.feature_density)
    for i in range(branches):
        side = "left" if i % 2 == 0 else "right"
        start = _edge_point(b, side, float(b.rng.uniform(0.15, 0.85)))
        target = trunk[int(b.rng.integers(3, len(trunk) - 3))]
        p = _meander_path(start, target, 9, m * 0.07 * s.naturalization, b.rng)
        b.carve_path(p, depth=950, half_width=m * 0.020, bank=m * 0.050, rim=80)
    b.add_detail(45 * s.detail, m * 0.025)
    return b.finalize(center_height=2600.0, preserve_flats=True)


def _recipe_mountain_basin(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(700)
    m = min(b.h, b.w)
    b.add_fbm(850, m * 0.22, ridged=True, octaves=5)
    yy, xx = np.mgrid[0:b.h, 0:b.w].astype(np.float32)
    cx, cy = b.w * 0.50, b.h * 0.52
    rr = np.sqrt(((xx - cx) / (m * 0.44)) ** 2 + ((yy - cy) / (m * 0.40)) ** 2)
    ring = np.exp(-0.5 * ((rr - 0.77) / 0.18) ** 2)
    basin = np.exp(-0.5 * (rr / 0.46) ** 2)
    b.a += ring * 720 - basin * 520
    b.add_random_craters(int(3 + 7 * s.feature_density), (m * 0.018, m * 0.05), 2.6, 0.7)
    b.add_detail(80 * s.detail, m * 0.025)
    if s.synthetic_pads == 0:
        b.flatten_pad(cx + m * 0.13, cy + m * 0.13, m * 0.055, m * 0.045, feather=m * 0.012, rectangular=True)
    return b.finalize(center_height=1700.0, preserve_flats=False)


def _recipe_radial_badlands(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(1400)
    m = min(b.h, b.w)
    b.add_fbm(330, m * 0.28, ridged=True, octaves=4)
    cx, cy = b.w * 0.5, b.h * 0.5
    arms = 5 + int(5 * s.feature_density)
    for i in range(arms):
        angle = i * math.tau / arms + float(b.rng.uniform(-0.18, 0.18))
        ex = cx + math.cos(angle) * m * 0.68
        ey = cy + math.sin(angle) * m * 0.68
        p = _meander_path((cx, cy), (ex, ey), 9, m * 0.045 * s.naturalization, b.rng)
        if i % 2:
            b.carve_path(p, depth=420, half_width=m * 0.012, bank=m * 0.045, rim=80)
        else:
            b.add_ridge_path(p, height=430, half_width=m * 0.012, falloff=m * 0.050)
    b.crater(cx, cy, m * 0.075, 380, 150)
    b.add_random_craters(int(4 + 10 * s.feature_density), (m * 0.015, m * 0.038), 2.2, 0.65)
    b.add_detail(65 * s.detail, m * 0.028)
    return b.finalize(center_height=1900.0, preserve_flats=False)


def _recipe_ridged_wastes(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(700)
    m = min(b.h, b.w)
    b.add_fbm(520, m * 0.075, ridged=True, octaves=5)
    b.add_fbm(260, m * 0.30, ridged=False, octaves=3)
    channels = 2 + int(3 * s.feature_density)
    for i in range(channels):
        if i % 2:
            st, en = _edge_point(b, "left", float(b.rng.uniform(.15,.85))), _edge_point(b, "right", float(b.rng.uniform(.15,.85)))
        else:
            st, en = _edge_point(b, "top", float(b.rng.uniform(.15,.85))), _edge_point(b, "bottom", float(b.rng.uniform(.15,.85)))
        p = _meander_path(st, en, 13, m * 0.09 * s.naturalization, b.rng)
        b.carve_path(p, depth=330, half_width=m * 0.010, bank=m * 0.025, rim=40)
    b.add_detail(35 * s.detail, m * 0.018)
    return b.finalize(center_height=1150.0, preserve_flats=False)


def _recipe_serpentine_canyon(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(2350)
    m = min(b.h, b.w)
    start = _edge_point(b, "top", 0.18)
    end = _edge_point(b, "bottom", 0.83)
    p = _meander_path(start, end, 18, m * (0.15 + 0.04 * s.naturalization), b.rng)
    b.carve_path(p, depth=1900, half_width=m * 0.026, bank=m * 0.055, rim=85)
    count = int(6 + 14 * s.feature_density)
    b.add_random_craters(count, (m * 0.012, m * 0.035), 1.8, 0.5)
    b.add_terraced_blobs([0, 90, 160], m * 0.11, 4.5, threshold_bias=0.25, warp_px=m * 0.015 * s.naturalization)
    b.add_detail(16 * s.detail, m * 0.024)
    return b.finalize(center_height=2700.0, preserve_flats=True)


def _recipe_natural_badlands(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(850)
    m = min(b.h, b.w)
    b.add_fbm(620, m * 0.16, ridged=True, octaves=5)
    b.add_fbm(300, m * 0.42, ridged=False, octaves=4)
    count = int(2 + 6 * s.feature_density)
    for _ in range(count):
        st = _edge_point(b, str(b.rng.choice(["left", "top"])), float(b.rng.uniform(.1,.9)))
        en = _edge_point(b, str(b.rng.choice(["right", "bottom"])), float(b.rng.uniform(.1,.9)))
        p = _meander_path(st, en, 11, m * 0.10 * s.naturalization, b.rng)
        b.carve_path(p, depth=250, half_width=m * 0.012, bank=m * 0.035, rim=30)
    b.add_detail(70 * s.detail, m * 0.020)
    return b.finalize(center_height=1500.0, preserve_flats=False)


RECIPES: Dict[str, Callable[[GeneratorSettings], HG2Map]] = {
    "Terraced Labyrinth": _recipe_terraced_labyrinth,
    "Cratered Divide": _recipe_cratered_divide,
    "Ravine Network": _recipe_ravine_network,
    "Mountain Basin": _recipe_mountain_basin,
    "Radial Badlands": _recipe_radial_badlands,
    "Ridged Wastes": _recipe_ridged_wastes,
    "Serpentine Canyon": _recipe_serpentine_canyon,
    "Natural Badlands": _recipe_natural_badlands,
}


def generate(style: str, settings: GeneratorSettings) -> HG2Map:
    try:
        fn = RECIPES[style]
    except KeyError as exc:
        raise ValueError(f"Unknown terrain style {style!r}. Choices: {', '.join(RECIPES)}") from exc
    return fn(settings)


def terrain_metrics(heightmap: np.ndarray) -> dict[str, float]:
    a = heightmap.astype(np.float32)
    gy, gx = np.gradient(a)
    g = np.hypot(gx, gy)
    c = a[1:-1, 1:-1]
    exact_flat = (
        (c == a[:-2, 1:-1])
        & (c == a[2:, 1:-1])
        & (c == a[1:-1, :-2])
        & (c == a[1:-1, 2:])
    )
    _, counts = np.unique(heightmap, return_counts=True)
    dominant = float(np.max(counts)) / heightmap.size if counts.size else 0.0
    return {
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "range": float(np.ptp(a)),
        "median_gradient": float(np.median(g)),
        "p95_gradient": float(np.percentile(g, 95)),
        "exact_flat_pct": float(np.mean(exact_flat) * 100.0),
        "dominant_level_pct": dominant * 100.0,
    }


def make_preview(heightmap: np.ndarray, max_size: Tuple[int, int] = (960, 760)) -> Image.Image:
    a = heightmap.astype(np.float32)
    lo, hi = np.percentile(a, [1.0, 99.5])
    norm = np.clip((a - lo) / max(hi - lo, 1.0), 0.0, 1.0)
    gy, gx = np.gradient(a)
    slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy) / 10.0)
    aspect = np.arctan2(-gx, gy)
    az = math.radians(315.0)
    alt = math.radians(45.0)
    shade = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    shade = _normalize01(shade)
    img = np.clip((0.62 * norm + 0.38 * shade) * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img, mode="L").convert("RGB")
    pil.thumbnail(max_size, Image.Resampling.LANCZOS)
    return pil


def _run_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("BZR Heightmap Generator")
    root.geometry("1320x850")
    root.configure(bg="#0a0a0a")

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    main = ttk.Frame(root, padding=10)
    main.pack(fill="both", expand=True)
    left = ttk.Frame(main, width=330)
    left.pack(side="left", fill="y", padx=(0, 10))
    left.pack_propagate(False)
    right = ttk.Frame(main)
    right.pack(side="right", fill="both", expand=True)

    vars_ = {
        "style": tk.StringVar(value="Terraced Labyrinth"),
        "zones_x": tk.IntVar(value=3),
        "zones_z": tk.IntVar(value=3),
        "seed": tk.IntVar(value=1),
        "relief": tk.DoubleVar(value=1.0),
        "naturalization": tk.DoubleVar(value=0.65),
        "detail": tk.DoubleVar(value=0.55),
        "plateau_bias": tk.DoubleVar(value=0.5),
        "feature_density": tk.DoubleVar(value=0.5),
        "symmetry": tk.StringVar(value="None"),
        "pads": tk.IntVar(value=0),
    }
    current: dict[str, Optional[HG2Map]] = {"map": None}
    preview_ref = {"image": None}

    ttk.Label(left, text="HEIGHTMAP GENERATOR", font=("Consolas", 13, "bold")).pack(anchor="w", pady=(0, 8))
    ttk.Label(left, text="Stock/custom-inspired terrain grammar with direct HG2 export.", wraplength=315).pack(anchor="w", pady=(0, 12))

    ttk.Label(left, text="Terrain Style").pack(anchor="w")
    ttk.Combobox(left, textvariable=vars_["style"], values=list(RECIPES), state="readonly").pack(fill="x", pady=(2, 8))

    dim = ttk.Frame(left)
    dim.pack(fill="x", pady=2)
    ttk.Label(dim, text="Zones X").pack(side="left")
    ttk.Spinbox(dim, textvariable=vars_["zones_x"], from_=1, to=8, width=5).pack(side="left", padx=(5, 12))
    ttk.Label(dim, text="Zones Z").pack(side="left")
    ttk.Spinbox(dim, textvariable=vars_["zones_z"], from_=1, to=8, width=5).pack(side="left", padx=5)

    seedrow = ttk.Frame(left)
    seedrow.pack(fill="x", pady=(5, 10))
    ttk.Label(seedrow, text="Seed").pack(side="left")
    ttk.Entry(seedrow, textvariable=vars_["seed"], width=12).pack(side="left", padx=6)
    def new_seed():
        vars_["seed"].set(random.SystemRandom().randint(1, 2_147_483_647))
        do_generate()
    ttk.Button(seedrow, text="Randomize", command=new_seed).pack(side="right")

    def slider(label: str, key: str, lo: float, hi: float, step: float):
        ttk.Label(left, text=label).pack(anchor="w")
        tk.Scale(left, variable=vars_[key], from_=lo, to=hi, resolution=step, orient="horizontal", showvalue=True,
                 bg="#0a0a0a", fg="#d4d4d4", highlightthickness=0, troughcolor="#222222").pack(fill="x")

    slider("Relief", "relief", 0.25, 2.25, 0.05)
    slider("Naturalization / edge warp", "naturalization", 0.0, 1.0, 0.05)
    slider("Fine detail", "detail", 0.0, 1.0, 0.05)
    slider("Plateau bias", "plateau_bias", 0.0, 1.0, 0.05)
    slider("Feature density", "feature_density", 0.0, 1.0, 0.05)

    ttk.Label(left, text="Synthetic Symmetry").pack(anchor="w", pady=(6, 0))
    ttk.Combobox(left, textvariable=vars_["symmetry"], values=["None", "Mirror X", "Mirror Z", "2-way rotational", "4-way"], state="readonly").pack(fill="x", pady=(2, 6))
    padrow = ttk.Frame(left)
    padrow.pack(fill="x", pady=3)
    ttk.Label(padrow, text="Objective pads").pack(side="left")
    ttk.Spinbox(padrow, textvariable=vars_["pads"], from_=0, to=8, width=5).pack(side="right")

    info = ttk.Label(right, text="Generate a terrain to preview it.")
    info.pack(anchor="w", pady=(0, 6))
    canvas = tk.Canvas(right, bg="#050505", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    def settings_from_ui() -> GeneratorSettings:
        return GeneratorSettings(
            zones_x=max(1, vars_["zones_x"].get()), zones_z=max(1, vars_["zones_z"].get()), seed=vars_["seed"].get(),
            relief=vars_["relief"].get(), naturalization=vars_["naturalization"].get(), detail=vars_["detail"].get(),
            plateau_bias=vars_["plateau_bias"].get(), feature_density=vars_["feature_density"].get(),
            symmetry=vars_["symmetry"].get(), synthetic_pads=max(0, vars_["pads"].get()),
        )

    def redraw():
        m = current["map"]
        if m is None:
            return
        maxw = max(200, canvas.winfo_width() - 16)
        maxh = max(200, canvas.winfo_height() - 16)
        pil = make_preview(m.heights, (maxw, maxh))
        tkimg = ImageTk.PhotoImage(pil)
        preview_ref["image"] = tkimg
        canvas.delete("all")
        canvas.create_image(canvas.winfo_width() // 2, canvas.winfo_height() // 2, image=tkimg, anchor="center")
        met = terrain_metrics(m.heights)
        wx, wz = m.world_size
        info.configure(text=(f"{m.heights.shape[1]}x{m.heights.shape[0]} samples | {wx:.0f}x{wz:.0f} world units | "
                             f"height {met['min']:.0f}..{met['max']:.0f} | flat {met['exact_flat_pct']:.1f}% | "
                             f"median/p95 gradient {met['median_gradient']:.1f}/{met['p95_gradient']:.1f}"))

    def do_generate():
        try:
            root.config(cursor="watch")
            root.update_idletasks()
            current["map"] = generate(vars_["style"].get(), settings_from_ui())
            redraw()
        except Exception as exc:
            messagebox.showerror("Generation failed", str(exc))
        finally:
            root.config(cursor="")

    def export_hg2():
        if current["map"] is None:
            do_generate()
        path = filedialog.asksaveasfilename(defaultextension=".hg2", filetypes=[("Battlezone HG2", "*.hg2")])
        if path:
            current["map"].write(path)

    def export_png():
        if current["map"] is None:
            do_generate()
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("16-bit PNG", "*.png")])
        if path:
            current["map"].write_png16(path)

    buttons = ttk.Frame(left)
    buttons.pack(fill="x", pady=(12, 0))
    ttk.Button(buttons, text="GENERATE", command=do_generate).pack(fill="x", pady=2)
    ttk.Button(buttons, text="Export HG2...", command=export_hg2).pack(fill="x", pady=2)
    ttk.Button(buttons, text="Export 16-bit PNG...", command=export_png).pack(fill="x", pady=2)
    canvas.bind("<Configure>", lambda _e: redraw())

    do_generate()
    root.mainloop()


def _cli() -> int:
    p = argparse.ArgumentParser(description="Stock/custom-inspired Battlezone 98 Redux HG2 heightmap generator")
    p.add_argument("--gui", action="store_true", help="open the Tkinter editor")
    p.add_argument("--style", choices=list(RECIPES), default="Terraced Labyrinth")
    p.add_argument("--zones", default="3x3", help="zone dimensions, e.g. 3x3 or 4x5")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--relief", type=float, default=1.0)
    p.add_argument("--naturalization", type=float, default=0.65)
    p.add_argument("--detail", type=float, default=0.55)
    p.add_argument("--plateau-bias", type=float, default=0.5)
    p.add_argument("--feature-density", type=float, default=0.5)
    p.add_argument("--symmetry", choices=["None", "Mirror X", "Mirror Z", "2-way rotational", "4-way"], default="None")
    p.add_argument("--pads", type=int, default=0)
    p.add_argument("--output", type=Path, help="output .hg2 path")
    p.add_argument("--png", type=Path, help="optional 16-bit PNG output")
    p.add_argument("--preview", type=Path, help="optional hillshade JPEG/PNG preview")
    args = p.parse_args()
    if args.gui:
        _run_gui()
        return 0
    try:
        zx, zz = (int(x) for x in args.zones.lower().split("x", 1))
    except Exception as exc:
        raise SystemExit("--zones must be formatted like 3x3") from exc
    s = GeneratorSettings(zx, zz, args.seed, args.relief, args.naturalization, args.detail,
                          args.plateau_bias, args.feature_density, args.symmetry, args.pads)
    m = generate(args.style, s)
    if args.output:
        m.write(args.output)
    if args.png:
        m.write_png16(args.png)
    if args.preview:
        make_preview(m.heights).save(args.preview)
    met = terrain_metrics(m.heights)
    print(f"style={args.style!r} seed={args.seed} zones={zx}x{zz} samples={m.heights.shape[1]}x{m.heights.shape[0]}")
    print(" ".join(f"{k}={v:.2f}" for k, v in met.items()))
    if not (args.output or args.png or args.preview):
        print("No output requested; use --output map.hg2, --png height.png, --preview preview.png, or --gui.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
