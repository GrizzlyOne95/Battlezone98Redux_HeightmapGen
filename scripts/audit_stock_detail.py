from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

import bzr_heightmap as hmg

STYLES = [
    "Terraced Labyrinth",
    "Cratered Divide",
    "Ravine Network",
    "Mountain Basin",
    "Radial Badlands",
    "Ridged Wastes",
    "Serpentine Canyon",
    "Natural Badlands",
]

REFERENCE = {
    "height_range": 2131.0,
    "flat_cell_pct": 40.57,
    "slope_median": 4.04,
    "slope_p95": 49.63,
    "trav15_pct": 72.68,
    "largest_trav15_pct": 51.34,
    "lap_p95": 28.0,
    "resid2_rms": 23.09,
    "band2_8_rms": 71.07,
    "localstd9_p90": 101.87,
}

BASELINE_STYLE = {
    "Cratered Divide": {"trav15_pct": 81.66, "largest_trav15_pct": 17.01, "flat_cell_pct": 34.20, "slope_median": 2.56, "slope_p95": 44.30, "resid2_rms": 8.29, "lap_p95": 12.0},
    "Mountain Basin": {"trav15_pct": 57.69, "largest_trav15_pct": 8.01, "flat_cell_pct": 24.06, "slope_median": 12.41, "slope_p95": 45.57, "resid2_rms": 8.09, "lap_p95": 12.0},
    "Natural Badlands": {"trav15_pct": 61.49, "largest_trav15_pct": 25.13, "flat_cell_pct": 17.84, "slope_median": 11.64, "slope_p95": 35.83, "resid2_rms": 6.61, "lap_p95": 10.0},
    "Radial Badlands": {"trav15_pct": 77.36, "largest_trav15_pct": 23.23, "flat_cell_pct": 12.06, "slope_median": 7.97, "slope_p95": 35.92, "resid2_rms": 5.40, "lap_p95": 8.0},
    "Ravine Network": {"trav15_pct": 70.72, "largest_trav15_pct": 21.08, "flat_cell_pct": 2.12, "slope_median": 6.65, "slope_p95": 34.91, "resid2_rms": 11.34, "lap_p95": 5.0},
    "Ridged Wastes": {"trav15_pct": 47.09, "largest_trav15_pct": 5.67, "flat_cell_pct": 16.52, "slope_median": 15.98, "slope_p95": 37.60, "resid2_rms": 9.94, "lap_p95": 12.0},
    "Serpentine Canyon": {"trav15_pct": 81.55, "largest_trav15_pct": 41.41, "flat_cell_pct": 17.94, "slope_median": 3.84, "slope_p95": 49.82, "resid2_rms": 14.59, "lap_p95": 6.0},
    "Terraced Labyrinth": {"trav15_pct": 61.55, "largest_trav15_pct": 13.40, "flat_cell_pct": 40.93, "slope_median": 2.92, "slope_p95": 49.44, "resid2_rms": 11.09, "lap_p95": 15.0},
}


def metrics(heightmap: np.ndarray) -> dict[str, float]:
    a = np.asarray(heightmap, dtype=np.float32)
    gy, gx = np.gradient(a * 0.1, 5.0, 5.0)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    passable = slope <= 15.0
    labels, n = ndimage.label(passable, structure=np.ones((3, 3), dtype=np.uint8))
    counts = np.bincount(labels.ravel())[1:] if n else np.asarray([], dtype=np.int64)
    largest = int(np.max(counts)) if counts.size else 0

    center = a[1:-1, 1:-1]
    exact_flat = (
        (center == a[:-2, 1:-1])
        & (center == a[2:, 1:-1])
        & (center == a[1:-1, :-2])
        & (center == a[1:-1, 2:])
    )
    lap = np.abs(ndimage.laplace(a))
    blur2 = ndimage.gaussian_filter(a, 2.0, mode="reflect")
    blur8 = ndimage.gaussian_filter(a, 8.0, mode="reflect")
    resid2 = a - blur2
    band2_8 = blur2 - blur8
    mean9 = ndimage.uniform_filter(a, size=9, mode="reflect")
    mean9sq = ndimage.uniform_filter(a * a, size=9, mode="reflect")
    std9 = np.sqrt(np.maximum(mean9sq - mean9 * mean9, 0.0))

    return {
        "height_range": float(np.ptp(a)),
        "flat_cell_pct": float(np.mean(exact_flat) * 100.0),
        "slope_median": float(np.median(slope)),
        "slope_p95": float(np.percentile(slope, 95)),
        "trav15_pct": float(np.mean(passable) * 100.0),
        "largest_trav15_pct": float(largest) * 100.0 / float(a.size),
        "lap_p95": float(np.percentile(lap, 95)),
        "resid2_rms": float(np.sqrt(np.mean(resid2 * resid2))),
        "band2_8_rms": float(np.sqrt(np.mean(band2_8 * band2_8))),
        "localstd9_p90": float(np.percentile(std9, 90)),
    }


def preview_image(heightmap: np.ndarray, size: int = 300) -> Image.Image:
    a = np.asarray(heightmap, dtype=np.float32)
    lo, hi = np.percentile(a, [1, 99])
    height_norm = np.clip((a - lo) / max(float(hi - lo), 1.0), 0.0, 1.0)
    gy, gx = np.gradient(a)
    nx = -gx
    ny = -gy
    nz = np.full_like(a, 42.0)
    mag = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / mag, ny / mag, nz / mag
    az = math.radians(315.0)
    el = math.radians(45.0)
    lx = math.cos(el) * math.cos(az)
    ly = math.cos(el) * math.sin(az)
    lz = math.sin(el)
    shade = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
    combined = np.clip(height_norm * 0.44 + shade * 0.56, 0.0, 1.0)
    img = Image.fromarray(np.rint(combined * 255.0).astype(np.uint8), mode="L").convert("RGB")
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    return img


def median_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    keys = list(REFERENCE)
    for style in STYLES:
        subset = [row for row in rows if row["style"] == style]
        entry: dict[str, object] = {"style": style, "n": len(subset)}
        for key in keys:
            entry[key] = float(np.median([float(row[key]) for row in subset]))
        result.append(entry)
    return result


def main() -> None:
    out = Path("stock_detail_audit")
    hg2_dir = out / "hg2"
    hg2_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    cards: list[tuple[str, int, Image.Image]] = []
    for style in STYLES:
        for _ in range(3):
            seed = hmg.random_seed()
            settings = hmg.GeneratorSettings(zones_x=3, zones_z=3, seed=seed)
            terrain = hmg.generate(style, settings)
            terrain.validate()
            filename = style.lower().replace(" ", "_") + f"_{seed}.hg2"
            terrain.write(hg2_dir / filename)
            row: dict[str, object] = {"style": style, "seed": seed}
            row.update(metrics(terrain.heights))
            rows.append(row)
            cards.append((style, seed, preview_image(terrain.heights)))

    medians = median_rows(rows)
    baseline_connectivity = float(np.median([v["largest_trav15_pct"] for v in BASELINE_STYLE.values()]))
    baseline_resid = float(np.median([v["resid2_rms"] for v in BASELINE_STYLE.values()]))
    baseline_lap = float(np.median([v["lap_p95"] for v in BASELINE_STYLE.values()]))
    new_connectivity = float(np.median([float(v["largest_trav15_pct"]) for v in medians]))
    new_resid = float(np.median([float(v["resid2_rms"]) for v in medians]))
    new_lap = float(np.median([float(v["lap_p95"]) for v in medians]))
    new_trav = float(np.median([float(v["trav15_pct"]) for v in medians]))

    summary = {
        "reference_medians": REFERENCE,
        "baseline_style_medians": BASELINE_STYLE,
        "fresh_style_medians": medians,
        "aggregate": {
            "baseline_largest_trav15_pct": baseline_connectivity,
            "fresh_largest_trav15_pct": new_connectivity,
            "reference_largest_trav15_pct": REFERENCE["largest_trav15_pct"],
            "baseline_resid2_rms": baseline_resid,
            "fresh_resid2_rms": new_resid,
            "reference_resid2_rms": REFERENCE["resid2_rms"],
            "baseline_lap_p95": baseline_lap,
            "fresh_lap_p95": new_lap,
            "reference_lap_p95": REFERENCE["lap_p95"],
            "fresh_trav15_pct": new_trav,
        },
        "acceptance": {
            "connectivity_improved_50pct": new_connectivity >= baseline_connectivity * 1.50,
            "micro_residual_improved_35pct": new_resid >= baseline_resid * 1.35,
            "laplacian_improved_20pct": new_lap >= baseline_lap * 1.20,
            "median_traversability_at_least_55pct": new_trav >= 55.0,
        },
    }

    with open(out / "fresh_metrics.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out / "summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    cols = 4
    card_w, card_h = 320, 345
    rows_n = math.ceil(len(cards) / cols)
    sheet = Image.new("RGB", (cols * card_w, rows_n * card_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for i, (style, seed, img) in enumerate(cards):
        x = (i % cols) * card_w
        y = (i // cols) * card_h
        sheet.paste(img, (x + 10, y + 10))
        draw.text((x + 10, y + 314), style, fill="black", font=font)
        draw.text((x + 10, y + 328), f"seed {seed}", fill="black", font=font)
    sheet.save(out / "fresh_contact_sheet.png")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
