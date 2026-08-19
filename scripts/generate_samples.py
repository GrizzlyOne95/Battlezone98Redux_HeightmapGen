from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "heightmap_generator.py"
OUT = ROOT / "samples"

# style, label, zones, seed, relief, naturalization, detail,
# plateau_bias, feature_density, symmetry, synthetic_pads
SAMPLES = [
    ("Terraced Labyrinth", "terraced_baseline", "3x3", 101, 1.00, 0.55, 0.45, 0.70, 0.55, "None", 0),
    ("Terraced Labyrinth", "terraced_symmetric", "4x4", 111, 1.15, 0.50, 0.35, 0.82, 0.60, "4-way", 4),
    ("Cratered Divide", "cratered_divide", "3x3", 202, 1.10, 0.55, 0.50, 0.45, 0.60, "None", 0),
    ("Cratered Divide", "cratered_arena", "2x2", 212, 0.95, 0.45, 0.40, 0.55, 0.50, "2-way rotational", 2),
    ("Ravine Network", "ravine_basin", "4x4", 303, 1.00, 0.70, 0.55, 0.42, 0.65, "None", 0),
    ("Ravine Network", "ravine_mirror", "3x4", 313, 1.05, 0.62, 0.50, 0.40, 0.72, "Mirror X", 2),
    ("Mountain Basin", "mountain_basin", "4x4", 404, 1.20, 0.70, 0.60, 0.35, 0.50, "None", 0),
    ("Mountain Basin", "mountain_fortress", "3x3", 414, 1.10, 0.62, 0.48, 0.50, 0.48, "4-way", 4),
    ("Radial Badlands", "radial_badlands", "3x3", 505, 1.05, 0.60, 0.55, 0.48, 0.62, "None", 0),
    ("Radial Badlands", "radial_spokes", "4x4", 515, 1.10, 0.58, 0.52, 0.50, 0.70, "2-way rotational", 2),
    ("Ridged Wastes", "ridged_wastes", "4x5", 606, 1.18, 0.65, 0.62, 0.30, 0.68, "None", 0),
    ("Ridged Wastes", "ridged_corridors", "3x4", 616, 1.08, 0.58, 0.58, 0.36, 0.60, "Mirror Z", 2),
    ("Serpentine Canyon", "serpentine_canyon", "4x4", 707, 1.00, 0.60, 0.46, 0.55, 0.55, "None", 0),
    ("Serpentine Canyon", "serpentine_arena", "2x2", 717, 0.90, 0.50, 0.38, 0.60, 0.48, "4-way", 4),
    ("Natural Badlands", "natural_badlands", "4x4", 808, 1.12, 0.72, 0.68, 0.28, 0.62, "None", 0),
    ("Natural Badlands", "natural_badlands_mirror", "3x3", 818, 1.00, 0.66, 0.60, 0.32, 0.54, "Mirror X", 2),
    ("Campaign Canyon Network", "campaign_canyon", "4x4", 909, 0.98, 0.58, 0.48, 0.62, 0.66, "None", 0),
    ("Campaign Canyon Network", "campaign_canyon_4way", "3x3", 919, 0.92, 0.52, 0.42, 0.66, 0.60, "4-way", 4),
    ("Compartmented Plateau", "compartmented_plateau", "4x4", 1001, 1.00, 0.56, 0.44, 0.78, 0.64, "None", 0),
    ("Compartmented Plateau", "compartmented_plateau_mirror", "3x4", 1011, 1.05, 0.52, 0.40, 0.80, 0.58, "Mirror Z", 2),
    ("Sparse Mission Field", "sparse_mission_field", "4x5", 1101, 0.90, 0.45, 0.35, 0.72, 0.30, "None", 0),
    ("Sparse Mission Field", "sparse_mission_arena", "3x3", 1111, 0.88, 0.42, 0.32, 0.74, 0.36, "2-way rotational", 4),
    ("Walled Crater Basin", "walled_crater_basin", "2x2", 1201, 0.95, 0.46, 0.38, 0.62, 0.44, "4-way", 4),
    ("Walled Crater Basin", "walled_crater_large", "3x3", 1211, 1.02, 0.50, 0.42, 0.58, 0.48, "2-way rotational", 2),
    ("Escarpment Stronghold", "escarpment_stronghold", "3x3", 1301, 1.08, 0.54, 0.42, 0.82, 0.56, "None", 2),
    ("Escarpment Stronghold", "escarpment_stronghold_4way", "4x4", 1311, 1.12, 0.50, 0.38, 0.84, 0.62, "4-way", 4),
]


def generate() -> list[dict[str, object]]:
    for subdir in ("hg2", "previews", "height_png"):
        (OUT / subdir).mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    for rec in SAMPLES:
        style, label, zones, seed, relief, naturalization, detail, plateau_bias, feature_density, symmetry, pads = rec
        hg2 = OUT / "hg2" / f"{label}.hg2"
        height_png = OUT / "height_png" / f"{label}.png"
        preview = OUT / "previews" / f"{label}_preview.png"

        cmd = [
            sys.executable,
            str(ENTRYPOINT),
            "--style", style,
            "--zones", zones,
            "--seed", str(seed),
            "--relief", str(relief),
            "--naturalization", str(naturalization),
            "--detail", str(detail),
            "--plateau-bias", str(plateau_bias),
            "--feature-density", str(feature_density),
            "--symmetry", symmetry,
            "--pads", str(pads),
            "--output", str(hg2),
            "--png", str(height_png),
            "--preview", str(preview),
        ]
        print(f"Generating {label} ({style}, {zones}, seed={seed})", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)

        manifest.append(
            {
                "label": label,
                "style": style,
                "zones": zones,
                "seed": seed,
                "relief": relief,
                "naturalization": naturalization,
                "detail": detail,
                "plateau_bias": plateau_bias,
                "feature_density": feature_density,
                "symmetry": symmetry,
                "pads": pads,
                "hg2": f"hg2/{label}.hg2",
                "height_png": f"height_png/{label}.png",
                "preview": f"previews/{label}_preview.png",
            }
        )
    return manifest


def write_manifest(manifest: list[dict[str, object]]) -> None:
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)


def write_contact_sheet(manifest: list[dict[str, object]]) -> None:
    thumb_w, thumb_h = 300, 220
    label_h = 46
    pad = 16
    cols = 3
    rows = (len(manifest) + cols - 1) // cols
    sheet = Image.new(
        "RGB",
        (cols * (thumb_w + pad) + pad, rows * (thumb_h + label_h + pad) + pad),
        (20, 20, 24),
    )
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(sheet)

    for i, item in enumerate(manifest):
        image = Image.open(OUT / str(item["preview"])).convert("RGB")
        image.thumbnail((thumb_w, thumb_h))
        cell = Image.new("RGB", (thumb_w, thumb_h), (40, 40, 46))
        cell.paste(image, ((thumb_w - image.width) // 2, (thumb_h - image.height) // 2))

        row, col = divmod(i, cols)
        ox = pad + col * (thumb_w + pad)
        oy = pad + row * (thumb_h + label_h + pad)
        sheet.paste(cell, (ox, oy))
        draw.text((ox, oy + thumb_h + 4), str(item["label"]), fill=(235, 235, 235), font=font)
        draw.text(
            (ox, oy + thumb_h + 20),
            f'{item["style"]} | {item["zones"]} | seed {item["seed"]}',
            fill=(170, 170, 180),
            font=font,
        )

    sheet.save(OUT / "preview_contact_sheet.png")


def write_readme() -> None:
    (OUT / "README.md").write_text(
        """# Generated Terrain Samples

This folder contains a reproducible review corpus generated by `scripts/generate_samples.py`.

- `hg2/` contains Battlezone 98 Redux HG2 terrain files ready for in-game testing.
- `previews/` contains hillshade PNG previews for rapid visual review.
- `height_png/` contains lossless 16-bit PNG height exports.
- `preview_contact_sheet.png` provides a single overview of all generated examples.
- `manifest.json` and `manifest.csv` record the exact style, seed, dimensions, and generator controls for each sample.

The sample pack intentionally spans natural, terraced, canyon, basin, radial, ridged, sparse mission, symmetric arena, compartmented plateau, and synthetic stronghold forms. It is intended for qualitative review and in-game iteration, not as a claim that every preset is final-quality terrain.

Regenerate the entire folder from the repository root with:

```bash
python scripts/generate_samples.py
```
""",
        encoding="utf-8",
    )


def main() -> int:
    manifest = generate()
    write_manifest(manifest)
    write_contact_sheet(manifest)
    write_readme()
    print(f"Generated {len(manifest)} samples under {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
