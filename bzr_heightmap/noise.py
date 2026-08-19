from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.interpolate import CubicSpline


def smoothstep01(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def normalize01(a: np.ndarray) -> np.ndarray:
    lo = float(np.min(a))
    hi = float(np.max(a))
    if hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - lo) / (hi - lo)).astype(np.float32)


def value_noise(shape: Tuple[int, int], feature_px: float, rng: np.random.Generator) -> np.ndarray:
    h, w = shape
    feature_px = max(2.0, float(feature_px))
    gh = max(3, int(math.ceil(h / feature_px)) + 3)
    gw = max(3, int(math.ceil(w / feature_px)) + 3)
    coarse = rng.random((gh, gw), dtype=np.float32)
    out = ndimage.zoom(coarse, (h / gh, w / gw), order=3, mode="reflect", prefilter=True)
    if out.shape[0] < h or out.shape[1] < w:
        out = np.pad(out, ((0, max(0, h - out.shape[0])), (0, max(0, w - out.shape[1]))), mode="edge")
    return normalize01(out[:h, :w])


def fbm(
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
        acc += (value_noise(shape, scale, rng) * 2.0 - 1.0) * amp
        amp_sum += amp
        amp *= persistence
        scale /= lacunarity
        if scale < 2.0:
            break
    return acc / max(amp_sum, 1e-6)


def ridged_fbm(shape: Tuple[int, int], feature_px: float, rng: np.random.Generator, octaves: int = 5) -> np.ndarray:
    n = fbm(shape, feature_px, rng, octaves=octaves, persistence=0.55)
    ridged = (1.0 - np.abs(n)) ** 2
    return normalize01(ridged) * 2.0 - 1.0


def warp_field(field: np.ndarray, amount_px: float, feature_px: float, rng: np.random.Generator) -> np.ndarray:
    if amount_px <= 0:
        return field
    h, w = field.shape
    wx = fbm((h, w), feature_px, rng, octaves=3, persistence=0.55)
    wy = fbm((h, w), feature_px, rng, octaves=3, persistence=0.55)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    coords = np.array([yy + wy * amount_px, xx + wx * amount_px], dtype=np.float32)
    return ndimage.map_coordinates(field, coords, order=1, mode="reflect").astype(np.float32)


def signed_blob_field(shape: Tuple[int, int], feature_px: float, rng: np.random.Generator, warp: float) -> np.ndarray:
    field = fbm(shape, feature_px, rng, octaves=3, persistence=0.55)
    field += 0.35 * fbm(shape, feature_px * 0.45, rng, octaves=2, persistence=0.5)
    return warp_field(field, warp, feature_px * 0.8, rng)


def polyline_distance(shape: Tuple[int, int], points: Sequence[Tuple[float, float]]) -> np.ndarray:
    h, w = shape
    img = Image.new("1", (w, h), 1)
    draw = ImageDraw.Draw(img)
    xy = [(float(x), float(y)) for x, y in points]
    if len(xy) == 1:
        draw.point(xy[0], fill=0)
    else:
        draw.line(xy, fill=0, width=1, joint="curve")
    return ndimage.distance_transform_edt(np.asarray(img, dtype=bool)).astype(np.float32)


def variable_corridor_mask(
    shape: Tuple[int, int],
    points: Sequence[Tuple[float, float]],
    half_widths: Sequence[float] | float,
) -> np.ndarray:
    h, w = shape
    if not points:
        return np.zeros(shape, dtype=bool)
    if isinstance(half_widths, (int, float)):
        widths = np.full(len(points), float(half_widths), dtype=np.float32)
    else:
        widths = np.asarray(list(half_widths), dtype=np.float32)
        if widths.size != len(points):
            raise ValueError("half_widths must be scalar or match points")
    widths = np.maximum(widths, 1.0)

    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    xy = [(float(x), float(y)) for x, y in points]
    if len(xy) == 1:
        x, y = xy[0]
        r = float(widths[0])
        draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
    else:
        for i in range(len(xy) - 1):
            width = max(1, int(round(float(widths[i]) + float(widths[i + 1]))))
            draw.line([xy[i], xy[i + 1]], fill=255, width=width, joint="curve")
        stride = max(1, len(xy) // 96)
        for i in range(0, len(xy), stride):
            x, y = xy[i]
            r = float(widths[i])
            draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
    return np.asarray(img, dtype=np.uint8) > 0


def vary_widths(
    count: int,
    base_width: float,
    variation: float,
    rng: np.random.Generator,
    cycles: float = 5.0,
) -> np.ndarray:
    count = max(1, int(count))
    if count == 1 or variation <= 0:
        return np.full(count, float(base_width), dtype=np.float32)
    raw = rng.normal(0.0, 1.0, count).astype(np.float32)
    raw = ndimage.gaussian_filter1d(raw, sigma=max(1.0, count / max(float(cycles), 1.0)), mode="nearest")
    raw /= max(float(np.max(np.abs(raw))), 1e-6)
    return np.maximum(float(base_width) * (1.0 + raw * float(variation)), 1.0).astype(np.float32)


def meander_path(
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
        points.append((sx + dx * t + nx * float(off) * jitter_px * taper, sy + dy * t + ny * float(off) * jitter_px * taper))

    tt = np.arange(count, dtype=np.float32)
    dense_t = np.linspace(0.0, count - 1.0, count * 8, dtype=np.float32)
    xs = CubicSpline(tt, np.asarray([p[0] for p in points]), bc_type="natural")(dense_t)
    ys = CubicSpline(tt, np.asarray([p[1] for p in points]), bc_type="natural")(dense_t)
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def organic_loop(
    center: Tuple[float, float],
    radius_x: float,
    radius_y: float,
    count: int,
    jitter: float,
    rng: np.random.Generator,
    rotation: float = 0.0,
) -> list[Tuple[float, float]]:
    count = max(6, int(count))
    angles = np.linspace(0.0, math.tau, count, endpoint=False, dtype=np.float32)
    radial = rng.normal(0.0, 1.0, count).astype(np.float32)
    radial = ndimage.gaussian_filter1d(radial, sigma=max(0.8, count / 10.0), mode="wrap")
    radial = radial / max(float(np.max(np.abs(radial))), 1e-6) * float(jitter)
    c, s = math.cos(rotation), math.sin(rotation)
    cx, cy = center
    xs, ys = [], []
    for angle, rj in zip(angles, radial):
        ex = math.cos(float(angle)) * (radius_x + rj)
        ey = math.sin(float(angle)) * (radius_y + rj * 0.65)
        xs.append(cx + ex * c - ey * s)
        ys.append(cy + ex * s + ey * c)
    t = np.arange(count + 1, dtype=np.float32)
    dense_t = np.linspace(0.0, count, count * 16 + 1, dtype=np.float32)
    xs2 = np.asarray(xs + [xs[0]], dtype=np.float32)
    ys2 = np.asarray(ys + [ys[0]], dtype=np.float32)
    xsd = CubicSpline(t, xs2, bc_type="periodic")(dense_t)
    ysd = CubicSpline(t, ys2, bc_type="periodic")(dense_t)
    return [(float(x), float(y)) for x, y in zip(xsd, ysd)]
