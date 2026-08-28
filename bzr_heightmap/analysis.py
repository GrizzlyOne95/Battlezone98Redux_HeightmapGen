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
    exact Battlezone vehicle/AI slope limit. It is useful for spotting generator
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


def _shelf_metrics(heightmap: np.ndarray) -> dict[str, float]:
    """Elevation grammar: shelves, plateaus, histogram shape."""
    a = np.asarray(heightmap, dtype=np.uint16)
    vals, cnts = np.unique(a, return_counts=True)
    total = float(a.size)
    order = np.argsort(cnts)[::-1]
    sorted_vals = vals[order]
    sorted_cnts = cnts[order]
    # Dominant shelves: levels covering >2% individually
    shelf_mask = sorted_cnts / total > 0.02
    shelf_count = int(np.count_nonzero(shelf_mask))
    shelf_area_pct = float(np.sum(sorted_cnts[shelf_mask]) / total * 100.0) if shelf_count else 0.0
    # Close to common authored levels: within +-2 of any top shelf
    top_shell_heights = sorted_vals[: max(1, shelf_count)]
    if top_shell_heights.size:
        close = np.zeros_like(a, dtype=bool)
        for h in top_shell_heights:
            close |= np.abs(a.astype(np.int32) - int(h)) <= 2
        close_pct = float(np.mean(close) * 100.0)
    else:
        close_pct = 0.0
    # Distance between major shelves (median gap)
    if shelf_count >= 2:
        gaps = np.diff(np.sort(top_shell_heights))
        shelf_gap_median = float(np.median(gaps))
        shelf_gap_max = float(np.max(gaps))
    else:
        shelf_gap_median = 0.0
        shelf_gap_max = 0.0
    # Large contiguous flat regions (exact-flat mask connected components)
    center = a[1:-1, 1:-1]
    exact_flat_core = (
        (center == a[:-2, 1:-1]) & (center == a[2:, 1:-1]) & (center == a[1:-1, :-2]) & (center == a[1:-1, 2:])
    )
    flat_mask = np.zeros_like(a, dtype=bool)
    flat_mask[1:-1, 1:-1] = exact_flat_core
    labels, ncomp = ndimage.label(flat_mask, structure=np.ones((3, 3), dtype=np.uint8))
    if ncomp:
        counts = np.bincount(labels.ravel())[1:]
        largest_flat = int(np.max(counts))
        largest_flat_pct = float(largest_flat / total * 100.0)
        # area of flats larger than 0.5% of map
        big = counts[counts >= total * 0.005]
        big_flat_area_pct = float(np.sum(big) / total * 100.0) if big.size else 0.0
        flat_component_count = int(ncomp)
    else:
        largest_flat_pct = 0.0
        big_flat_area_pct = 0.0
        flat_component_count = 0
    # Elevation histogram shape: lowland/highland balance (below vs above median)
    median_h = float(np.median(a))
    lowland_pct = float(np.mean(a < median_h) * 100.0)
    highland_pct = 100.0 - lowland_pct
    # Distinct heights needed for 80% coverage
    cum = np.cumsum(sorted_cnts) / total
    n80 = int(np.searchsorted(cum, 0.8) + 1) if cum.size else 0
    return {
        "shelf_count_gt2pct": float(shelf_count),
        "shelf_area_pct": shelf_area_pct,
        "close_to_shelf_pct": close_pct,
        "shelf_gap_median": shelf_gap_median,
        "shelf_gap_max": shelf_gap_max,
        "largest_flat_component_pct": largest_flat_pct,
        "big_flat_area_pct": big_flat_area_pct,
        "flat_component_count": float(flat_component_count),
        "lowland_pct": lowland_pct,
        "highland_pct": highland_pct,
        "distinct_heights": float(vals.size),
        "levels_for_80pct": float(n80),
    }


def _slope_grammar(heightmap: np.ndarray) -> dict[str, float]:
    a = np.asarray(heightmap, dtype=np.float32)
    gy, gx = np.gradient(a * VERTICAL_UNIT_SCALE, HORIZONTAL_SAMPLE_SPACING, HORIZONTAL_SAMPLE_SPACING)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    hist, edges = np.histogram(slope_deg, bins=[0, 5, 15, 30, 45, 60, 90])
    total = float(max(int(slope_deg.size), 1))
    gentle = float(hist[0] / total * 100.0)  # 0-5
    moderate = float(hist[1] / total * 100.0)  # 5-15
    steep = float(hist[2] / total * 100.0)  # 15-30
    very_steep = float(hist[3] / total * 100.0)  # 30-45
    cliff = float(hist[4] / total * 100.0 + hist[5] / total * 100.0)  # 45-90
    # Transition band width: estimate via gradient of slope near escarpments
    # Use Laplacian of height as proxy for abrupt steps
    lap = ndimage.laplace(a)
    abrupt = float(np.mean(np.abs(lap) > 50.0) * 100.0)
    return {
        "slope_gentle_0_5_pct": gentle,
        "slope_moderate_5_15_pct": moderate,
        "slope_steep_15_30_pct": steep,
        "slope_very_steep_30_45_pct": very_steep,
        "slope_cliff_45_90_pct": cliff,
        "abrupt_step_pct": abrupt,
        "slope_std": float(np.std(slope_deg)),
    }


def _spatial_scale_metrics(heightmap: np.ndarray) -> dict[str, float]:
    a = np.asarray(heightmap, dtype=np.float32)
    # Low vs high frequency energy via Gaussian blurs at multiple scales
    sigma_small, sigma_large = 4.0, 32.0
    small = ndimage.gaussian_filter(a, sigma=sigma_small, mode="reflect")
    large = ndimage.gaussian_filter(a, sigma=sigma_large, mode="reflect")
    hf = a - small
    lf = large
    hf_energy = float(np.var(hf))
    lf_energy = float(np.var(lf))
    total_var = float(np.var(a))
    lf_ratio = lf_energy / max(total_var, 1e-6) * 100.0
    hf_ratio = hf_energy / max(total_var, 1e-6) * 100.0
    # Local variance at medium scale as roughness proxy (fast uniform-filter path)
    # Use uniform filters to avoid generic_filter per-pixel Python callback overhead.
    mean9 = ndimage.uniform_filter(a, size=9, mode="reflect")
    mean9_sq = ndimage.uniform_filter(a * a, size=9, mode="reflect")
    local_var = mean9_sq - mean9 * mean9
    local_var = np.maximum(local_var, 0.0)
    roughness = float(np.mean(local_var))
    # Autocorrelation length approx via distance where autocorr drops to 0.5
    # Cheap estimate: compare variance of downsampled vs original
    # Use 0.1 quantile of gradient magnitude spacing as characteristic feature size
    gy, gx = np.gradient(a)
    grad_mag = np.hypot(gx, gy)
    char_scale = float(np.percentile(grad_mag, 50))  # median gradient
    return {
        "lf_energy_pct": lf_ratio,
        "hf_energy_pct": hf_ratio,
        "roughness_var9": roughness,
        "char_gradient_median": char_scale,
    }


def _gameplay_structure_metrics(heightmap: np.ndarray) -> dict[str, float]:
    # Reuse traversability but also add corridor / choke proxies
    trav = traversability_metrics(heightmap, max_slope_deg=40.0)
    a = np.asarray(heightmap, dtype=np.float32)
    gy, gx = np.gradient(a * VERTICAL_UNIT_SCALE, HORIZONTAL_SAMPLE_SPACING, HORIZONTAL_SAMPLE_SPACING)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    passable = slope_deg <= 40.0
    # Distance transform of impassable gives corridor width estimate for passable cells
    if np.any(passable):
        inv = ~passable
        dist = ndimage.distance_transform_edt(passable).astype(np.float32)
        # Corridor width = 2 * distance to nearest wall (in samples * spacing)
        widths_m = dist[passable] * HORIZONTAL_SAMPLE_SPACING * 2.0
        median_corridor_m = float(np.median(widths_m)) if widths_m.size else 0.0
        p10_corridor_m = float(np.percentile(widths_m, 10)) if widths_m.size else 0.0
        open_field_pct = float(np.mean(widths_m > 80.0) * 100.0)  # broad >80m
        choke_pct = float(np.mean(widths_m < 20.0) * 100.0)  # narrow <20m
    else:
        median_corridor_m = 0.0
        p10_corridor_m = 0.0
        open_field_pct = 0.0
        choke_pct = 0.0
    # Isolated basins: count small passable components disconnected from main
    labels, n = ndimage.label(passable, structure=np.ones((3, 3), dtype=np.uint8))
    counts = np.bincount(labels.ravel())[1:] if n else np.array([], dtype=int)
    isolated_basins = int(np.sum(counts < max(1, int(a.size * 0.002)))) if counts.size else 0
    return {
        "corridor_median_width_m": median_corridor_m,
        "corridor_p10_width_m": p10_corridor_m,
        "open_field_pct": open_field_pct,
        "choke_pct": choke_pct,
        "isolated_basins": float(isolated_basins),
        **{f"gameplay_{k}": v for k, v in trav.items()},
    }


def describe_heightmap(heightmap: np.ndarray, top_levels: int = 6) -> dict[str, object]:
    a = np.asarray(heightmap, dtype=np.uint16)
    report: dict[str, object] = dict(terrain_metrics(a))
    # Extend with new grammar metrics (retain old keys for compatibility)
    report.update(_shelf_metrics(a))
    report.update(_slope_grammar(a))
    report.update(_spatial_scale_metrics(a))
    report.update(_gameplay_structure_metrics(a))
    values, counts = np.unique(a, return_counts=True)
    order = np.argsort(counts)[::-1][: max(1, int(top_levels))]
    total = max(int(a.size), 1)
    report["top_levels"] = [
        {"height": int(values[i]), "percent": float(counts[i]) * 100.0 / total}
        for i in order
    ]
    return report


def make_shaded_image(heightmap: np.ndarray) -> Image.Image:
    """Render the full-resolution combined elevation and hillshade image."""
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
    return Image.fromarray(image, mode="L").convert("RGB")


def make_preview(heightmap: np.ndarray, max_size: Tuple[int, int] = (960, 760)) -> Image.Image:
    preview = make_shaded_image(heightmap)
    preview.thumbnail(max_size, Image.Resampling.LANCZOS)
    return preview
