# BZMapIO Blender authoring reference

Research note for the stock-detail / terrain-grammar work.

## Source

- File: `BZMapIO.py`
- Add-on name: **Battlezone Map IO**
- Add-on version: **1.5**
- Declared authors: **Business Lawyer**, HG2 converter by **DivisionByZero**
- Declared Blender target: **3.60.0**
- License header: **GPL v2 or later**
- Source snapshot SHA-256: `bd4fc144e9d732a8ccb144bf6633afe8fe72b14d5afcbde3cc8e0a25fe6d7e36`

The source is valuable because it exposes HG2 as an editable Blender mesh at native terrain-sample density, then writes the edited vertex heights back to HG2. It therefore provides an independent authoring reference for practical terrain scale, vertical quantization, orientation, and feature-size limits.

## HG2 format corroboration

The embedded HG2 reader uses:

- structure version `1`
- `zone_bits = 8`
- map version `10`
- little-endian 16-bit height samples
- `height & 0x1FFF` on read

This independently agrees with the codec/corpus work already in this repository.

## Native sculpt lattice

BZMapIO constructs each HG2 zone as:

- **1280 world units** wide/deep
- **256 × 256 HG2 samples** per zone

Therefore the native HG2 sample interval is:

`1280 / 256 = 5 world units per sample`

This is important for terrain-generation design. Features substantially smaller than the sample lattice cannot be represented faithfully in HG2. A procedural feature should generally span multiple samples if it is intended to read as an authored landform rather than a one-cell spike.

BZMapIO imports each HG2 height sample into Blender with:

`Blender Z = HG2 height / 10`

and exports Blender terrain with:

`HG2 height = round(Blender Z * 10)`

So one HG2 height unit corresponds to **0.1 Blender/world vertical unit** in this workflow.

## Practical vertical authoring envelope

Before HG2 export, BZMapIO clamps terrain vertices to:

- minimum: **0.0**
- maximum: **409.5** Blender/world units

The exported integer range is therefore:

- `0 .. 4095` (`0x0000 .. 0x0FFF`)

This strongly corroborates HeightmapGen's existing distinction between:

- **HG2 storage range:** `0 .. 8191` (`0x1FFF`)
- **safe/new-terrain authoring range:** `0 .. 4095` (`0x0FFF`)

The Blender source is evidence for the **authoring ceiling**, not proof that the engine cannot load stored values `4096 .. 8191`. Existing HG2 files may still contain the full 13-bit storage range and must remain losslessly readable/writable.

## Terrain-grammar implications

The generator should be evaluated at the actual HG2 vertex lattice rather than only as a smooth image.

Useful derived constraints:

- horizontal sample interval: **5 world units**
- vertical quantization: **0.1 world unit**
- practical new-terrain vertical span: **409.5 world units**
- one-zone native terrain grid: **256 × 256 samples over 1280 × 1280 world units**

For stock-style detail, this supports the current regional-feature direction:

1. Preserve large connected traversable surfaces.
2. Concentrate detail into localized crater, knoll, shelf, ridge, ramp, and ravine provinces.
3. Avoid relying on sub-sample/high-frequency noise to create perceived detail.
4. Give small cliffs/shelves approaches that occupy several HG2 samples rather than single-cell transitions.
5. Validate generated features after integer quantization, because Blender export ultimately rounds back to integer HG2 heights.

## Validation use

BZMapIO should be treated as a third reference alongside:

1. the stock/custom HG2 corpus,
2. the reconstructed HG2 binary codec,
3. engine/WorldBuilder behavior.

It is especially useful for answering a different question from corpus statistics: **how much geometrically meaningful terrain detail can be sculpted into the exact HG2 lattice while remaining within the normal Redux authoring envelope?**

A future validation pass should compare generated maps after native-lattice quantization and optionally round-trip representative maps through the same Blender coordinate mapping.
