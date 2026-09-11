from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import bzr_heightmap as hmg
from audit_stock_detail import metrics, preview_image

CORE_STYLES = [
    "Terraced Labyrinth",
    "Cratered Divide",
    "Ravine Network",
    "Mountain Basin",
    "Radial Badlands",
    "Ridged Wastes",
    "Serpentine Canyon",
    "Natural Badlands",
    "Campaign Canyon Network",
    "Compartmented Plateau",
    "Sparse Mission Field",
    "Walled Crater Basin",
    "Escarpment Stronghold",
]

PLANETARY_STYLES = [
    "Pluto Basin",
    "Venus Shield",
    "Lunar Catena",
    "Mars Rift",
    "Callisto Craterlands",
    "Titan Basin Network",
    "Europa Fracture Plains",
]

STYLES = CORE_STYLES + PLANETARY_STYLES
SEEDS = [1729, 8675309, 314159265]


def style_group(style: str) -> str:
    return "planetary" if style in PLANETARY_STYLES else "core"


def detail_profile(style: str) -> str:
    if style in hmg.STOCK_DETAIL_STYLES:
        return "stock_v5"
    if style in hmg.PLANETARY_DETAIL_STYLES:
        return "craterland_v1"
    if style in hmg.PLANETARY_SURFACE_STYLES:
        return "planetary_surface_v1"
    return "raw"


def median_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    numeric = [
        "min_height",
        "max_height",
        "height_range",
        "flat_cell_pct",
        "slope_median",
        "slope_p95",
        "trav15_pct",
        "largest_trav15_pct",
        "lap_p95",
        "resid2_rms",
        "band2_8_rms",
        "localstd9_p90",
    ]
    result: list[dict[str, object]] = []
    for style in STYLES:
        subset = [row for row in rows if row["style"] == style]
        entry: dict[str, object] = {
            "style": style,
            "group": style_group(style),
            "detail_profile": detail_profile(style),
            "n": len(subset),
        }
        for key in numeric:
            entry[key] = float(np.median([float(row[key]) for row in subset]))
        result.append(entry)
    return result


def main() -> None:
    out = Path("all_natural_audit")
    hg2_dir = out / "hg2"
    hg2_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    cards: list[tuple[str, int, Image.Image, str]] = []

    for style in STYLES:
        if style not in hmg.RECIPES:
            raise RuntimeError(f"Natural audit style is not registered: {style}")
        for seed in SEEDS:
            settings = hmg.GeneratorSettings(zones_x=3, zones_z=3, seed=seed)
            terrain = hmg.generate(style, settings)
            terrain.validate()
            heights = np.asarray(terrain.heights)
            filename = style.lower().replace(" ", "_") + f"_{seed}.hg2"
            terrain.write(hg2_dir / filename)

            row: dict[str, object] = {
                "style": style,
                "group": style_group(style),
                "detail_profile": detail_profile(style),
                "seed": seed,
                "min_height": int(np.min(heights)),
                "max_height": int(np.max(heights)),
                "authoring_floor_hits": int(np.count_nonzero(heights == 0)),
                "authoring_ceiling_hits": int(np.count_nonzero(heights == hmg.HG2_SAFE_MAX_HEIGHT)),
            }
            row.update(metrics(heights))
            rows.append(row)
            cards.append((style, seed, preview_image(heights), detail_profile(style)))

    medians = median_rows(rows)
    groups: dict[str, dict[str, float]] = {}
    for group in ("core", "planetary"):
        subset = [row for row in medians if row["group"] == group]
        groups[group] = {
            "styles": float(len(subset)),
            "median_trav15_pct": float(np.median([float(r["trav15_pct"]) for r in subset])),
            "median_largest_trav15_pct": float(np.median([float(r["largest_trav15_pct"]) for r in subset])),
            "median_flat_cell_pct": float(np.median([float(r["flat_cell_pct"]) for r in subset])),
            "median_lap_p95": float(np.median([float(r["lap_p95"]) for r in subset])),
            "median_resid2_rms": float(np.median([float(r["resid2_rms"]) for r in subset])),
            "max_observed_height": float(max(float(r["max_height"]) for r in subset)),
        }

    summary = {
        "sample_spacing_world_units": 5.0,
        "vertical_quantum_world_units": 0.1,
        "safe_authoring_range_hg2": [0, hmg.HG2_SAFE_MAX_HEIGHT],
        "styles": len(STYLES),
        "seeds_per_style": len(SEEDS),
        "total_maps": len(rows),
        "stock_v5_styles": sorted(hmg.STOCK_DETAIL_STYLES),
        "craterland_v1_styles": sorted(hmg.PLANETARY_DETAIL_STYLES),
        "planetary_surface_v1_styles": sorted(hmg.PLANETARY_SURFACE_STYLES),
        "raw_styles": [style for style in STYLES if detail_profile(style) == "raw"],
        "groups": groups,
        "style_medians": medians,
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
    for i, (style, seed, img, profile) in enumerate(cards):
        x = (i % cols) * card_w
        y = (i // cols) * card_h
        sheet.paste(img, (x + 10, y + 10))
        draw.text((x + 10, y + 314), f"{style} [{profile}]", fill="black", font=font)
        draw.text((x + 10, y + 328), f"seed {seed}", fill="black", font=font)
    sheet.save(out / "fresh_contact_sheet.png")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
