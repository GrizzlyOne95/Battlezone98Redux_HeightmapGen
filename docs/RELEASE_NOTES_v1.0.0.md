# Battlezone98Redux Heightmap Generator v1.0.0

## Summary

First formal packaged release of the Battlezone 98 Redux Heightmap Generator.

The generator focuses on Battlezone-style authored terrain grammar rather than generic noise: shelves, ravines, basins, crater structures, corridors, escarpments, objective pads, planetary-inspired archetypes, and urban terrain substrates. Generated terrain can be exported directly to HG2 along with several preview/reference formats.

## Highlights

- Core Battlezone-derived terrain recipes plus planetary and urban style families.
- Reproducible seeded generation with fresh random seeds by default.
- Live Tkinter GUI with HG2 height, LGT-style lighting, and shaded previews.
- Direct HG2 export plus lossless 16-bit PNG/reference image outputs.
- Experimental bordered LGT export and LGT-style preview generation.
- Existing-HG2 analysis and terrain-quality diagnostics.
- HGT/HG2 support and authoring-oriented terrain utilities.
- Deterministic generation for fixed settings/seeds.

## Packaged Windows build

The Windows release contains a standalone `BZR_Heightmap_Generator.exe` built with PyInstaller. Double-clicking the packaged executable opens the GUI automatically. Command-line/source workflows remain available from the repository.

## Validation

The release workflow runs the repository unit-test suite before packaging and performs a basic frozen-executable version smoke test.

## Installation

1. Download `BZR_Heightmap_Generator.exe` from this release.
2. Run the executable.
3. The GUI opens directly; no separate Python installation is required for the packaged Windows build.

For source/CLI use, install `requirements.txt` and run `python heightmap_generator.py --help`.
