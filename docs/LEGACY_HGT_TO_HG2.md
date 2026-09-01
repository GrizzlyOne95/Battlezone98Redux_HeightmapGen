# Legacy HGT to HG2 conversion

Battlezone 98 Redux can load Battlezone 1.x `.hgt` terrain, but it cooks it at
load time and finishes with a 3x3 box blur that rounds off authored geometry.
This document records the format and algorithm behind
`bzr_heightmap/hgt.py` and `scripts/convert_legacy_hgt.py`, which produce the
same terrain *without* that blur.

Everything here was transcribed from the shipped GOG
`battlezone98redux.exe` (2.2.301, image base `0x400000`) and verified against
files the game itself produced. No constant below is inferred from guesswork.

## Formats

### Legacy `.HGT`

No header. `zones * 0x8000` bytes of little-endian `uint16`, zone-major then
row-major within each zone, 128x128 samples per zone. `FUN_00785f50`:

    index = (z & 127) * 128 + (x & 127)
          + (x >> 7) * 0x4000
          + (z >> 7) * zones_x * 0x4000

Only the low 12 bits are height — every fetch is masked `& 0xFFF`. The upper
nibble is set in most stock files and is not height; the engine discards it.

Dimensions are **not** in the file. Redux reads `Width`/`Depth` from the
companion `.TRN` and computes `zones = (int)(v * 0.1) >> 7`, rejecting the
terrain unless `v == zones * 1280` (`FUN_00786340`). A zone is therefore 1280
world units across: 128 legacy samples at 10 units each.

Where a TRN carries more than one `[Size]` section, the first one wins.

### Redux `.HG2`

A 12-byte header then `zones * 0x20000` bytes of `uint16`, 256x256 per zone,
same tiling (`FUN_00785c00` returns `base + 0xC + zone * 0x20000`).

| Offset | Type | Field | Loader check |
|---|---|---|---|
| 0 | `u16` | structure version | must be `1` |
| 2 | `u16` | zone bits | must be `8` (256 samples/zone) |
| 4 | `u16` | zones X | must equal the TRN's |
| 6 | `u16` | zones Z | must equal the TRN's |
| 8 | `u32` | map version | must be `>= 10` |

An HG2 zone is 256 samples over the same 1280 world units, i.e. 5 units per
sample — exactly twice the legacy density. One raw height unit is 0.1 world
units (`FUN_007859d0` fills default terrain with `(short)(Height / 0.1)`).

> The loader checks `size >= 13` but never checks the payload length against
> the declared zone counts, so a truncated or mislabelled HG2 is accepted and
> then read out of bounds. `read_hg2_header` reports `size_consistent`
> separately for this reason.

## Loader precedence

`FUN_00786340` builds `<mission>.hg2` and tries it first. Only if the file is
missing *or* its header fails the table above does it fall back to
`<mission>.HGT`. After cooking an HGT it **writes the result to disk as
`.hg2`**, so a converted map simply takes the place of that cache.

Supplying a valid `.hg2` is therefore sufficient. No `.trn` edit, no mission
change, no renaming of the `.hgt`.

## The cook

`FUN_00786200` walks every output sample and reads the legacy grid at
`(out_x * 0.5, out_z * 0.5)` through `FUN_00785fe0` — a plain 2x upsample.
Then, unless the `-nohgtsmoothing` command-line switch was given
(`DAT_009454cc`, set in `FUN_007d5120`), it calls `FUN_00785c80`.

### Interpolation is piecewise-planar, not bilinear

`FUN_00785fe0` splits each legacy cell along its `(0,0)-(1,1)` diagonal and
evaluates the plane through the three vertices of the containing triangle
(`comiss fz, fx; jbe` selects the `(0,0)-(1,0)-(1,1)` triangle when `fz <= fx`).
That is the surface the 1998 engine rendered, so this step is faithful: it
reproduces authored vertices exactly and reconstructs the legacy surface
between them. It is a resolution change, not a loss.

The whole expression is single-precision SSE and ends in `cvttss2si`
(truncation toward zero). `rounding="engine"` reproduces that bit-for-bit;
`rounding="half-up"` evaluates the plane in exact integers and rounds halves
up. The two differ by at most one raw unit (0.1 world units) and only at
half-sample positions.

### The blur is the destructive step

`FUN_00785c80` averages each sample with its in-bounds 8-neighbourhood, out of
place, rounding with `(2 * sum + n) / (2 * n)`. This is the only part of the
pipeline that moves authored vertices, and it is what erases stair-steps,
mesa rims and ridge lines. `smoothing=False` omits it; nothing else changes.

## Verification

- The full pipeline (upsample + blur) reproduces `addon/ccafun01.hg2` — a file
  the game itself cooked from `ccafun01.hgt` — **bytewise**, 2,097,164 bytes,
  zero differing samples.
- Across 72 stock maps, comparing Redux's shipped `StockODFFiles/*.hg2`
  against both modes: 67 match the smoothed cook to within one raw unit,
  and `multdm29` (Great Pyramid) and `multdm17` match the *unsmoothed*
  conversion instead — Rebellion shipped those two the way this tool does.
  `misn02b` and `multdm77` match neither and are hand-modified terrain.
- Legacy vertices are reproduced exactly on all 72.
- Zone dimensions agree across three independent sources (the 1.5 `.trn` from
  `bzone.zfs`, Redux's extracted `.trn`, and the shipped `.hg2` headers) on
  every stock map.

## Usage

```
python scripts/convert_legacy_hgt.py convert misn01.hgt -o misn01.hg2
python scripts/convert_legacy_hgt.py batch <stockfiles> -o out/ --trn-dir <trns> --csv report.csv
python scripts/convert_legacy_hgt.py scan <root> --csv census.csv
python scripts/convert_legacy_hgt.py compare ours.hg2 shipped.hg2
```

`--smoothed` reproduces Redux's default cook instead, for parity checking.
`scan` never writes terrain.

## Known bad inputs

- `multst35.hgt` (stock 1.5) is 141312 bytes where its own TRN declares 4x4
  (524288). It is truncated; Redux ships no `multst35.hg2` either.
- `test.hgt` is 16 zones but `test.trn` declares 2x4, and the shipped
  `test.hg2` header declares 2x4 over a 4x4 payload. Broken in both engines.
- HGT files with no TRN are dimensionally ambiguous whenever the zone count is
  not a perfect square; `zone_count_candidates` lists the alternatives and the
  CLI reports which one it assumed.
