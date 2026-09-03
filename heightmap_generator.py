from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from bzr_heightmap import (
    GeneratorSettings,
    HG2Map,
    RECIPES,
    compute_lgt_lightmap,
    describe_heightmap,
    generate,
    make_hg2_height_image,
    make_lgt_preview_image,
    make_preview,
    resolve_seed,
    terrain_metrics,
    write_lgt,
)

APP_VERSION = "1.0.0"


def cli() -> int:
    # A packaged desktop build should behave like a desktop application when
    # launched normally. Source/CLI invocations keep their existing behavior.
    if getattr(sys, "frozen", False) and len(sys.argv) == 1:
        sys.argv.append("--gui")

    parser = argparse.ArgumentParser(description="Stock/custom-inspired Battlezone 98 Redux HG2 heightmap generator")
    parser.add_argument("--version", action="version", version=f"Battlezone98Redux Heightmap Generator {APP_VERSION}")
    parser.add_argument("--gui", action="store_true", help="open the Tkinter editor")
    parser.add_argument("--style", choices=list(RECIPES), default="Terraced Labyrinth")
    parser.add_argument("--zones", default="3x3", help="zone dimensions, e.g. 3x3 or 4x5")
    parser.add_argument("--seed", default="random", help="integer seed for reproducibility, or 'random' (default)")
    parser.add_argument("--relief", type=float, default=1.0)
    parser.add_argument("--vertical-scale", type=float, default=1.0, help="post-generation vertical contrast multiplier; 0.75 keeps 75%% of height differences")
    parser.add_argument("--naturalization", type=float, default=0.65)
    parser.add_argument("--detail", type=float, default=0.55)
    parser.add_argument("--plateau-bias", type=float, default=0.5)
    parser.add_argument("--feature-density", type=float, default=0.5)
    parser.add_argument("--symmetry", choices=["None", "Mirror X", "Mirror Z", "2-way rotational", "4-way"], default="None")
    parser.add_argument("--pads", type=int, default=0)
    parser.add_argument("--output", type=Path, help="output .hg2 path")
    parser.add_argument("--png", type=Path, help="optional lossless 16-bit PNG output")
    parser.add_argument("--preview", type=Path, help="optional hillshade JPEG/PNG preview")
    parser.add_argument("--hg2-preview", type=Path, help="optional fixed-range HG2 height preview PNG")
    parser.add_argument("--lgt-preview", type=Path, help="optional LGT-style lighting preview PNG")
    parser.add_argument("--lgt-output", type=Path, help="optional experimental bordered LGT output")
    parser.add_argument("--lgt-zone-size", type=int, choices=(128, 256), default=256, help="LGT samples per zone (default: 256 for Redux)")
    parser.add_argument("--analyze-hg2", type=Path, help="inspect an existing HG2 and print terrain metrics")
    args = parser.parse_args()

    if args.analyze_hg2:
        terrain = HG2Map.read(args.analyze_hg2)
        report = describe_heightmap(terrain.heights)
        report.update({"zones_x": terrain.zones_x, "zones_z": terrain.zones_z, "zone_bits": terrain.zone_bits, "world_size": terrain.world_size})
        print(json.dumps(report, indent=2))
        return 0

    if args.gui:
        from bzr_heightmap.gui import run_gui
        run_gui()
        return 0

    try:
        zones_x, zones_z = (int(value) for value in args.zones.lower().split("x", 1))
    except Exception as exc:
        raise SystemExit("--zones must be formatted like 3x3") from exc

    try:
        resolved_seed = resolve_seed(args.seed)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if not (0.1 <= args.vertical_scale <= 2.5):
        raise SystemExit("--vertical-scale must be between 0.1 and 2.5")

    settings = GeneratorSettings(
        zones_x=zones_x, zones_z=zones_z, seed=resolved_seed,
        relief=args.relief, vertical_scale=args.vertical_scale,
        naturalization=args.naturalization, detail=args.detail,
        plateau_bias=args.plateau_bias, feature_density=args.feature_density,
        symmetry=args.symmetry, synthetic_pads=args.pads,
    )
    terrain = generate(args.style, settings)
    if args.output:
        terrain.write(args.output)
    if args.png:
        terrain.write_png16(args.png)
    if args.preview:
        make_preview(terrain.heights).save(args.preview)
    if args.hg2_preview:
        make_hg2_height_image(terrain.heights).save(args.hg2_preview)
    if args.lgt_preview:
        make_lgt_preview_image(terrain.heights, terrain.zones_x, terrain.zones_z, lgt_zone_size=args.lgt_zone_size).save(args.lgt_preview)
    if args.lgt_output:
        lightmap = compute_lgt_lightmap(terrain.heights, terrain.zones_x, terrain.zones_z, lgt_zone_size=args.lgt_zone_size)
        write_lgt(args.lgt_output, lightmap, terrain.zones_x, terrain.zones_z)

    metrics = terrain_metrics(terrain.heights)
    print(f"style={args.style!r} seed={resolved_seed} zones={zones_x}x{zones_z} vertical_scale={args.vertical_scale:.2f} samples={terrain.heights.shape[1]}x{terrain.heights.shape[0]}")
    print(" ".join(f"{key}={value:.2f}" for key, value in metrics.items()))
    if not (args.output or args.png or args.preview or args.hg2_preview or args.lgt_preview or args.lgt_output):
        print("No output requested; use --output, --png, --preview, --hg2-preview, --lgt-preview, --lgt-output, or --gui.")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
