# HG2 corpus analysis — 2026-08-28

This report records the reproducible findings used to tune the generator and
implement the HG2/LGT previews. It deliberately separates measurements from
generator heuristics. No proprietary terrain files or machine-specific paths
are committed.

## Discovery and accounting

`scripts/analyze_local_corpus.py` searched the local Documents, Steam
`steamapps`, and GOG game trees case-insensitively for `.hg2`, parsed each
candidate without writing to it, and hashed the masked little-endian height
payload. The path classifier normalizes both slash styles into case-folded
components; it does not use a Windows-separator-sensitive substring test.

The final equations reconcile exactly:

- **509 paths = 507 valid + 2 invalid**
- **507 valid paths = 275 unique contents + 232 duplicate copies**
- **275 unique contents = 249 authored + 26 HeightmapGen samples + 0
  synthetic-only**

The 507 valid paths comprise 458 paths classified as authored, 26 under a
`HeightmapGen/samples/hg2` component sequence, and 23 synthetic/test paths.
The synthetic paths are recognized from the OpenShim `test_missions` tree or
the known fixture stems `bz14atk`, `lcbench`, `magnz0`, `pilot`, and `wmtest0`.
All 23 synthetic paths have one height hash, and that hash is also present at
two authored paths. Authored therefore wins for that shared unique-content
group; it is not counted again as synthetic-only.

There are 133 duplicate-content groups. The two invalid files are copies of
`test.hg2` whose headers declare a 2x4 map but whose payload size corresponds
to 4x4; neither was included in terrain metrics.

Valid path dimensions are:

| Zones | Paths |
|---|---:|
| 4x4 | 259 |
| 3x3 | 86 |
| 2x2 | 80 |
| 1x1 | 45 |
| 4x3 | 26 |
| 3x4 | 5 |
| 4x5 | 3 |
| 2x4 | 2 |
| 8x8 | 1 |

## Authored terrain measurements

The following are arithmetic means over the 249 disjoint unique authored
height contents. They come from `describe_heightmap`; they are corpus
measurements, not engine rules.

| Metric | Mean |
|---|---:|
| Exact-flat four-neighbour area | 39.49% |
| Largest exact height level | 32.53% |
| Median local slope | 7.75 degrees |
| 95th-percentile local slope | 44.17 degrees |
| Height range | 2419.92 height units |
| Exact shelves covering more than 2% | 1.76 |
| Area in those shelves | 39.23% |
| Largest connected exact-flat component | 27.05% |

The analyzer also reports a distance-transform clearance statistic. Its
authored mean is 613.38 world units, but this is an open-space proxy over a
slope mask, not a literal measurement of designed corridor width and is not
used as evidence for a specific Battlezone vehicle limit.

The strongest supported terrain finding is structural: authored maps contain
large piecewise-constant shelves while also spanning substantially more
vertical and upper-tail slope range than the generated baseline. That
supports the project model of **macro composition -> authored gameplay forms
-> selective naturalization**, rather than treating every map as a global
noise field.

## Generated baseline and tuned branch

The comparison used every 26 recipe styles at 3x3 zones and five fixed seeds
(1, 42, 123, 777, and 2024), for 130 samples at baseline commit `69fd27e` and
130 samples after this branch's recipe edits.

| Metric | Authored 249 | Baseline 130 | Tuned 130 |
|---|---:|---:|---:|
| Exact-flat area | 39.49% | 22.28% | 26.68% |
| Dominant exact level | 32.53% | 6.12% | 9.10% |
| Median slope | 7.75 deg | 4.78 deg | 3.20 deg |
| P95 slope | 44.17 deg | 27.19 deg | 28.81 deg |
| Height range | 2419.92 | 1107.14 | 1069.87 |

The changes moved exact-flat area by +4.40 percentage points, dominant-level
area by +2.98 points, and P95 slope by +1.62 degrees. They did **not** close
the relief gap: mean range fell by 37.27 height units and remains far below
the authored corpus. The vertical-contrast control lets an author adjust the
finished height field without claiming that recipe defaults now match the
corpus in every dimension.

The largest per-recipe flat-area changes were:

| Recipe | Baseline | Tuned | Other reproduced change |
|---|---:|---:|---|
| Terraced Labyrinth | 27.53% | 37.89% | P95 slope 42.14 -> 49.33 |
| Cratered Divide | 0.73% | 33.51% | range 1421 -> 1690 |
| Mountain Basin | 0.56% | 22.96% | range 2516 -> 1928 |
| Ridged Wastes | 0.01% | 15.08% | P95 slope 52.48 -> 38.59 |
| Natural Badlands | 0.00% | 16.41% | P95 slope 40.83 -> 35.09 |
| Radial Badlands | 0.09% | 8.60% | range 1632 -> 1708 |
| Pluto Basin | 9.29% | 13.24% | P95 slope 5.26 -> 14.82 |
| Titan Basin Network | 4.48% | 8.25% | P95 slope 4.27 -> 16.05 |

The implementation widens selected authored forms, stamps a small number of
protected staging shelves in formerly all-noise rugged recipes, reduces
detail that destroyed those flats, and retains recipe-specific macro forms.
It does not apply a universal shelf pass. Recipe identities remain distinct;
for example, reproduced flat area ranges from 1.44% for Ravine Network to
61.04% for Sparse Mission Field, and P95 slope ranges from 2.41 degrees for
Lunar Catena to 60.58 degrees for Compartmented Plateau.

Planetary changes are Battlezone authoring heuristics inspired by the same
corpus-wide shelf evidence. They are not claims that the local corpus contains
enough Pluto- or Titan-specific authored maps to establish a planetary rule.

## HG2/LGT relationship

All 507 valid HG2 files use `zone_bits=8`: 256x256 height samples per
1280-world-unit zone. The controlled scan found **447 HG2/LGT path pairs**,
covering **238 unique HG2 contents**, all of which fall in the authored
classification.

Companion-aware size inference found these path-level layouts:

- 433 bordered, 256 samples per zone
- 9 bordered, 128 samples per zone
- 5 unrecognized

By unique HG2 content, 230 have only a bordered-256 companion layout, five
have only bordered-128, one appears with bordered-256 and an unrecognized
copy, and two have only unrecognized companions. Those four disjoint groups
sum to 238.

Both HG2 and the readable LGT zone blocks use row-major southwest origin.
The project represents this as array row 0 = south, row increasing north,
column 0 = west. Z64Tools flips a conventional north-at-top PNG before
zoning; this project already holds south-first arrays, so it writes those
arrays directly. A second flip would invert lighting relative to height.
This interpretation was checked against local pairs: direct computed/file
correlation exceeded a north/south-flipped comparison for both a bordered-256
stock example and a bordered-128 legacy example. Correlation supports
orientation, not exact recovery of the game's lighting algorithm.

LGT byte values are contributions above the documented 25% ambient floor;
preview brightness maps byte 0 to 25% and byte 255 to 100%. The generator
computes slope-normal lighting with a configurable sun direction and supports
both 128 and 256 samples per zone. The live preview defaults to 128 for the
classic terrain-cell relationship; GUI binary export defaults to the dominant
Redux 256 layout.

The binary writer emits the observed border-plus-zone layout and round-trips
through this project's reader. It has **not** been boot-tested in Battlezone
98 Redux and remains explicitly experimental. The PNG/live LGT-style preview
is supported as an authoring visualization; it is not proof of an
engine-valid `.LGT` or an exact replica of game shadowing/occlusion.

## Live GUI behavior

The GUI exposes seed/randomize, terrain style, zone dimensions, vertical
contrast, and advanced recipe settings with a debounced live preview. It has
separate HG2 Height, LGT Lighting, and Shaded views. Full generation runs on
one background worker, but Tk variables, widgets, and `PhotoImage` creation
remain on the main thread.

Generation-affecting fields form the raw-terrain cache key. Vertical contrast
is deliberately excluded: a contrast-only change reuses the raw terrain and
calls `apply_vertical_scale`. A latest-job coordinator coalesces rapid input,
rejects stale results by revision, and starts the newest pending request after
the active job. Closing invalidates outstanding revisions and stops polling,
so a worker can only place a plain result in a queue and cannot touch a
destroyed Tk interpreter.

## Reproduction and limitations

- Analyzer: `python scripts/analyze_local_corpus.py --discover`
- Tests: `python -m unittest discover -s tests -v`
- The ignored JSON/CSV report contains local paths; only this aggregate report
  and the sanitized summary are committed.
- Terrain classes and provenance are explicit path heuristics. Content hashes,
  parse validity, dimensions, and metric calculations are data-derived.
- Slope/passability/clearance metrics are generator diagnostics, not verified
  engine physics or AI thresholds.
- Local layout evidence establishes LGT dimensions, block structure, and a
  defensible orientation. It does not establish full renderer semantics.

## References

- `scripts/analyze_local_corpus.py`
- `bzr_heightmap.analysis.describe_heightmap`
- `bzr_heightmap.lgt`
- Battlezone `format_lgt.html` documentation
- Z64Tools `tools/terrain_pack.py`
- Battlezone98Redux WorldBuilder HG2 tooling
