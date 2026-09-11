# Battlezone98Redux Heightmap Generator v1.0.1

## Summary

This release focuses on HG2 correctness and a major upgrade to natural terrain finishing. The generator now uses stock-derived terrain detail grammar and morphology-specific finishing to produce terrain that better matches Battlezone's authored terrain character while preserving playability, symmetry, and safe authoring limits.

## Highlights

- Correct HG2 codec/interchange behavior:
  - Preserve the full 13-bit HG2 storage range for reading/writing.
  - Keep newly generated terrain constrained to the safe 0..4095 authoring range.
  - Use canonical `height × 8` scaling for lossless PNG16 interchange.
  - Add dedicated HG2 header, payload-order, round-trip, truncation, and PNG16 tests.
- Add stock-derived natural terrain finishing with localized shelves, ramps, craterlets, knolls/bowls, broken edges, and small-scale surface structure.
- Add morphology-specific planetary finishing for Pluto Basin, Venus Shield, Titan Basin Network, Europa Fracture Plains, and crater-focused planetary styles.
- Add localized finishing for Campaign Canyon Network, Compartmented Plateau, Escarpment Stronghold, and Mars Rift while preserving their macro topology.
- Preserve explicitly requested terrain symmetry through post-generation finishing.
- Add deterministic stock-detail and full-natural audit tooling plus expanded regression coverage.
- Add BZMapIO-derived authoring reference documentation for Redux terrain dimensions, sample spacing, vertical quantization, and authoring range.
- Remove superseded experimental terrain-detail modules after the validated finishing pipeline was selected.

## Validation

The merged stock-detail pass reported:

- 79/79 unit tests passing.
- 24/24 stock-detail validation samples generated successfully across eight core natural styles and three fixed seeds.
- 60/60 all-natural validation samples generated successfully across twenty styles and three fixed seeds.
- Generated HG2 terrain remained within the safe 0..4095 authoring range.

The release workflow reruns the repository unit-test suite, verifies that the executable reports version 1.0.1, builds the standalone Windows executable with PyInstaller, and smoke-tests the frozen executable before publishing.

## Installation

1. Download `BZR_Heightmap_Generator.exe` from this release.
2. Run the executable.
3. The GUI opens directly; no separate Python installation is required for the packaged Windows build.

For source/CLI use, install `requirements.txt` and run `python heightmap_generator.py --help`.
