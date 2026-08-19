# Battlezone98Redux Heightmap Generator

Experimental Battlezone 98 Redux terrain generator built around the terrain grammar visible in good stock and hand-authored custom HG2 maps.

The goal is **not** to generate generic Perlin-noise terrain. Battlezone maps frequently depend on large exact-height shelves, readable corridors, ravines, crater rims, staging basins, escarpments, synthetic pads, and deliberately controlled transition bands. This tool treats those as first-class terrain primitives and then applies naturalization/detail selectively.

## Current terrain styles

- Terraced Labyrinth
- Cratered Divide
- Ravine Network
- Mountain Basin
- Radial Badlands
- Ridged Wastes
- Serpentine Canyon
- Natural Badlands
- Campaign Canyon Network
- Compartmented Plateau
- Sparse Mission Field
- Walled Crater Basin
- Escarpment Stronghold

Generation uses a **fresh random seed by default** so repeated runs/clicks quickly produce new terrain. Every resolved seed is shown and can be supplied again for exact reproducibility. Global controls adjust relief, naturalization, fine detail, plateau bias, feature density, optional symmetry, and synthetic objective pads.

## Generated sample corpus

The repository includes a checked-in `samples/` review corpus containing 26 generated terrains across the current style set and a range of dimensions, seeds, symmetry modes, relief/detail settings, and synthetic pad counts.

Each sample includes:

- a directly testable `.hg2` under `samples/hg2/`
- a hillshade review image under `samples/previews/`
- a lossless 16-bit height PNG under `samples/height_png/`
- exact generation parameters in `samples/manifest.json` and `samples/manifest.csv`

`samples/preview_contact_sheet.png` provides a single visual overview of the checked-in set. The corpus can be reproduced with:

```bash
python scripts/generate_samples.py
```

## HG2 handling

HG2 I/O follows the same layout/indexing used by `BZMapIO.py` and the Redux WorldBuilder tooling:

- 12-byte header: structure version, zone bits, map width/depth in zones, map version
- zone-major height payload
- normal Redux maps use `zone_bits = 8`, or 256x256 samples per 1280-unit zone
- the reader preserves the 13-bit storage mask (`0x1FFF`) for compatibility
- generated/exported terrain defaults to the stock authoring-safe `0..4095` height range (`0..409.5` world units)

The generator can also export a lossless 16-bit PNG representation.

Reference implementation for HG2/world tooling: [Battlezone98Redux_WorldBuilder](https://github.com/GrizzlyOne95/Battlezone98Redux_WorldBuilder)

## Install

```bash
python -m pip install -r requirements.txt
```

Python 3.10+ is recommended.

## GUI

```bash
python heightmap_generator.py --gui
```

The GUI provides terrain style, map dimensions, seed, relief/naturalization/detail controls, symmetry, objective pads, a hillshaded preview, and HG2/PNG export. **Fresh random seed each Generate** is enabled by default; disable it when you want to lock a seed while tuning parameters.

## CLI examples

Generate a 3x3 campaign-style canyon with a reproducible seed:

```bash
python heightmap_generator.py \
  --style "Campaign Canyon Network" \
  --zones 3x3 \
  --seed 42 \
  --output canyon.hg2 \
  --png canyon.png \
  --preview canyon_preview.png
```

Omit `--seed` (or pass `--seed random`) for a fresh seed. The resolved numeric seed is printed so a useful random result can always be reproduced later.

Generate a more synthetic symmetric arena:

```bash
python heightmap_generator.py \
  --style "Walled Crater Basin" \
  --zones 2x2 \
  --symmetry "4-way" \
  --pads 4 \
  --output arena.hg2
```

Inspect an existing HG2:

```bash
python heightmap_generator.py --analyze-hg2 map.hg2
```

The analysis reports dimensions, elevation range, physical slope statistics, exact-flat percentage, dominant authored elevation percentage, the most common exact elevation levels, and a slope-mask connectivity diagnostic. The connectivity value is a generator-quality heuristic, not a claim about Battlezone's exact vehicle or AI slope limit.

## Design principles

The generator intentionally separates three scales of terrain construction:

1. **Macro composition** — broad basins, highlands, lowlands, bounded arenas, radial or directional layouts.
2. **Authored gameplay forms** — exact-height shelves, flat-core canyons, variable-width approaches, loops/compartments, mesas, craters, ramps and objective pads.
3. **Naturalization** — domain variation, irregular banks, ridged/fBm detail, edge breakup and smoothing where it does not destroy authored gameplay geometry.

Protected gameplay flats are not modified by later detail/smoothing passes. Synthetic symmetry copies authored halves/quadrants rather than averaging them, so exact shelf and corridor heights remain exact.

## Validation

The HG2 reader/writer has been round-trip checked against the original 28-map stock/custom/campaign reference corpus. Terrain-shape analysis has since been expanded to **55 accessible authored HG2 references**, including ROTBD/RBD maps ranging from compact 1x1 arenas through large campaign terrain. Generated maps remain deterministic when a resolved numeric seed is supplied and are clamped to the stock-safe output range.

Run the unit tests with:

```bash
python -m unittest discover -s tests -v
```
