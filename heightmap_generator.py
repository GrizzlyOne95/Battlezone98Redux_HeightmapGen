from __future__ import annotations

import argparse
import json
from pathlib import Path

from bzr_heightmap import GeneratorSettings, HG2Map, RECIPES, describe_heightmap, generate, make_preview, resolve_seed, terrain_metrics


def cli() -> int:
    parser = argparse.ArgumentParser(description="Stock/custom-inspired Battlezone 98 Redux HG2 heightmap generator")
    parser.add_argument("--gui", action="store_true", help="open the Tkinter editor")
    parser.add_argument("--style", choices=list(RECIPES), default="Terraced Labyrinth")
    parser.add_argument("--zones", default="3x3", help="zone dimensions, e.g. 3x3 or 4x5")
    parser.add_argument("--seed", default="random", help="integer seed for reproducibility, or 'random' (default)")
    parser.add_argument("--relief", type=float, default=1.0)
    parser.add_argument("--naturalization", type=float, default=0.65)
    parser.add_argument("--detail", type=float, default=0.55)
    parser.add_argument("--plateau-bias", type=float, default=0.5)
    parser.add_argument("--feature-density", type=float, default=0.5)
    parser.add_argument("--symmetry", choices=["None", "Mirror X", "Mirror Z", "2-way rotational", "4-way"], default="None")
    parser.add_argument("--pads", type=int, default=0)
    parser.add_argument("--output", type=Path, help="output .hg2 path")
    parser.add_argument("--png", type=Path, help="optional lossless 16-bit PNG output")
    parser.add_argument("--preview", type=Path, help="optional hillshade JPEG/PNG preview")
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

    settings = GeneratorSettings(
        zones_x=zones_x, zones_z=zones_z, seed=resolved_seed,
        relief=args.relief, naturalization=args.naturalization, detail=args.detail,
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

    metrics = terrain_metrics(terrain.heights)
    print(f"style={args.style!r} seed={resolved_seed} zones={zones_x}x{zones_z} samples={terrain.heights.shape[1]}x{terrain.heights.shape[0]}")
    print(" ".join(f"{key}={value:.2f}" for key, value in metrics.items()))
    if not (args.output or args.png or args.preview):
        print("No output requested; use --output map.hg2, --png height.png, --preview preview.png, or --gui.")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
