from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from bzr_heightmap import (
    GeneratorSettings,
    PLANETARY_RECIPES,
    RANDOM_SEED_MAX,
    generate,
    make_preview,
    random_seed,
    terrain_metrics,
    traversability_metrics,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a planetary-inspired HG2 review corpus")
    p.add_argument("--output", type=Path, default=Path("planetary_samples"))
    p.add_argument("--variants", type=int, default=2, help="fresh-seed variants per style")
    p.add_argument("--zones", default="3x3", help="default zone size, e.g. 3x3")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    zx, zz = (int(v) for v in args.zones.lower().split("x", 1))
    out = args.output
    for sub in ("hg2", "height_png", "previews"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for style in PLANETARY_RECIPES:
        for variant in range(max(1, args.variants)):
            seed = random_seed()
            zones_x = zx + (variant % 2)
            zones_z = zz
            settings = GeneratorSettings(
                zones_x=zones_x,
                zones_z=zones_z,
                seed=seed,
                relief=1.0,
                naturalization=0.65,
                detail=0.4,
                plateau_bias=0.45,
                feature_density=0.62,
            )
            terrain = generate(style, settings)
            label = f"{style.lower().replace(' ', '_')}_{variant + 1:02d}"
            terrain.write(out / "hg2" / f"{label}.hg2")
            terrain.write_png16(out / "height_png" / f"{label}.png")
            make_preview(terrain.heights, (480, 480)).save(out / "previews" / f"{label}.png")
            row: dict[str, object] = {
                "label": label,
                "style": style,
                "variant": variant + 1,
                "zones_x": zones_x,
                "zones_z": zones_z,
                "seed": seed,
            }
            row.update(terrain_metrics(terrain.heights))
            row.update({f"terrain40_{k}": v for k, v in traversability_metrics(terrain.heights, 40.0).items()})
            rows.append(row)

    fields = list(rows[0]) if rows else []
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    font = ImageFont.load_default()
    cols = max(1, min(4, args.variants))
    thumb, pad, label_h = 240, 16, 44
    sheet_rows = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (pad + cols * (thumb + pad), 34 + pad + sheet_rows * (thumb + label_h + pad)), (18, 18, 22))
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 10), "Planetary-inspired Battlezone terrain", fill=(245, 245, 245), font=font)
    for i, row in enumerate(rows):
        rr, cc = divmod(i, cols)
        x, y = pad + cc * (thumb + pad), 34 + pad + rr * (thumb + label_h + pad)
        img = Image.open(out / "previews" / f"{row['label']}.png").convert("RGB")
        img.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb, thumb), (45, 45, 50))
        canvas.paste(img, ((thumb - img.width) // 2, (thumb - img.height) // 2))
        sheet.paste(canvas, (x, y))
        draw.text((x, y + thumb + 3), str(row["label"]), fill=(235, 235, 235), font=font)
        draw.text((x, y + thumb + 19), f"seed {row['seed']} | {row['zones_x']}x{row['zones_z']}", fill=(170, 180, 190), font=font)
    sheet.save(out / "review_contact_sheet.png")
    print(f"Generated {len(rows)} samples in {out} (seed max {RANDOM_SEED_MAX}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
