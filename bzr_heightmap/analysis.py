from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

from .noise import normalize01


HORIZONTAL_SAMPLE_SPACING = 5.0
VERTICAL_UNIT_SCALE = 0.1



def traversability_metrics(
    heightmap: np.ndarray,
    max_slope_deg: float = 40.0,
    min_component_fraction: float = 0.002,
) -> dict[str, float]:
    """Estimate broad terrain connectivity using a slope-limited driveability mask.

    This is deliberately a terrain-level heuristic rather than a claim about the
    exact Battlezone vehicle/AI slope limit.  It is useful for spotting generator
    failures where large flat-looking shelves are separated by continuous cliff
    bands with no ramps or saddles.
    """
    a = np.asarray(heightmap, dtype=np.float32)
    gy, gx = np.gradient(a * VERTICAL_UNIT_SCALE, HORIZONTAL_SAMPLE_SPACING, HORIZONTAL_SAMPLE_SPACING)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    passable = slope_deg <= float(max_slope_deg)
    labels, _ = ndimage.label(passable, structure=np.ones((3, 3), dtype=np.uint8))
    counts = np.bincount(labels.ravel())[1:]
    passable_count = int(np.count_nonzero(passable))
    if counts.size == 0 or passable_count == 0:
        return {
            "passable_pct": 0.0,
            "largest_passable_component_pct": 0.0,
            "major_passable_components": 0.0,
        }
    min_size = max(1, int(a.size * float(min_component_fraction)))
    major = counts[counts >= min_size]
    return {
        "passable_pct": float(passable_count) * 100.0 / float(a.size),
        "largest_passable_component_pct": float(np.max(counts)) * 100.0 / float(passable_count),
        "major_passable_components": float(major.size),
    }

def terrain_metrics(heightmap: np.ndarray) -> dict[str, float]:
    a = heightmap.astype(np.float32)
    gy, gx = np.gradient(a * VERTICAL_UNIT_SCALE, HORIZONTAL_SAMPLE_SPACING, HORIZONTAL_SAMPLE_SPACING)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    center = a[1:-1, 1:-1]
    exact_flat = (
        (center == a[:-2, 1:-1])
        & (center == a[2:, 1:-1])
        & (center == a[1:-1, :-2])
        & (center == a[1:-1, 2:])
    )
    _, counts = np.unique(heightmap, return_counts=True)
    dominant = float(np.max(counts)) / heightmap.size if counts.size else 0.0
    report = {
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "range": float(np.ptp(a)),
        "median_slope_deg": float(np.median(slope_deg)),
        "p95_slope_deg": float(np.percentile(slope_deg, 95)),
        "exact_flat_pct": float(np.mean(exact_flat) * 100.0),
        "dominant_level_pct": dominant * 100.0,
    }
    connectivity = traversability_metrics(a, max_slope_deg=40.0)
    report.update({f"terrain40_{key}": value for key, value in connectivity.items()})
    return report


def describe_heightmap(heightmap: np.ndarray, top_levels: int = 6) -> dict[str, object]:
    a = np.asarray(heightmap, dtype=np.uint16)
    report: dict[str, object] = dict(terrain_metrics(a))
    values, counts = np.unique(a, return_counts=True)
    order = np.argsort(counts)[::-1][: max(1, int(top_levels))]
    total = max(int(a.size), 1)
    report["top_levels"] = [
        {"height": int(values[i]), "percent": float(counts[i]) * 100.0 / total}
        for i in order
    ]
    return report


def make_preview(heightmap: np.ndarray, max_size: Tuple[int, int] = (960, 760)) -> Image.Image:
    a = heightmap.astype(np.float32)
    lo, hi = np.percentile(a, [1.0, 99.5])
    normalized = np.clip((a - lo) / max(hi - lo, 1.0), 0.0, 1.0)
    gy, gx = np.gradient(a * VERTICAL_UNIT_SCALE, HORIZONTAL_SAMPLE_SPACING, HORIZONTAL_SAMPLE_SPACING)
    slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    azimuth = math.radians(315.0)
    altitude = math.radians(45.0)
    shade = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
    shade = normalize01(shade)
    image = np.clip((0.62 * normalized + 0.38 * shade) * 255.0, 0, 255).astype(np.uint8)
    preview = Image.fromarray(image, mode="L").convert("RGB")
    preview.thumbnail(max_size, Image.Resampling.LANCZOS)
    return preview
