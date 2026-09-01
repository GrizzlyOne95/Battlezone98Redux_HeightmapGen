"""Convert legacy Battlezone 1.x ``.HGT`` terrain to Redux ``.HG2``.

By default this performs Redux's own 2x piecewise-planar upsample and skips its
3x3 box blur, so authored geometry (stair-steps, mesas, sharp rims) survives.
Supplying the resulting ``.hg2`` alongside the ``.hgt`` is enough: Redux loads
HG2 first and only falls back to the smoothing cook when no valid HG2 exists.

    python scripts/convert_legacy_hgt.py convert misn01.hgt -o misn01.hg2
    python scripts/convert_legacy_hgt.py batch "C:/.../stockfiles" -o out/
    python scripts/convert_legacy_hgt.py scan "C:/Program Files (x86)" --csv census.csv

See ``docs/LEGACY_HGT_TO_HG2.md`` for the format and algorithm provenance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bzr_heightmap.hgt import (  # noqa: E402
    HGTFormatError,
    HGTMap,
    LEGACY_ZONE_BYTES,
    ROUNDING_MODES,
    find_trn,
    read_hg2_header,
    read_trn_zone_counts,
    zone_count_candidates,
)


def _iter_hgt(root: Path):
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".hgt":
            yield path


def _load(path: Path, zones: tuple[int, int] | None, trn_dir: Path | None = None):
    if zones is not None:
        return HGTMap.read(path, *zones), "explicit"
    size = path.stat().st_size
    # Prefer a TRN beside the HGT, then one in an explicit lookup directory.
    # Stock 1.5 keeps its TRNs inside bzone.zfs, so the extracted Redux
    # StockODFFiles tree is usually the only authoritative dimension source --
    # and it matters: multst25.hgt is 12 zones, which infers as 3x4 but is
    # really 4x3.
    sources = []
    beside = find_trn(path)
    if beside is not None:
        sources.append(("trn", beside))
    if trn_dir is not None:
        for suffix in (".trn", ".TRN"):
            candidate = trn_dir / (path.stem + suffix)
            if candidate.exists():
                sources.append(("trn-dir", str(candidate)))
                break
    for label, trn in sources:
        counts = read_trn_zone_counts(trn)
        if counts is not None and counts[0] * counts[1] * LEGACY_ZONE_BYTES == size:
            return HGTMap.read(path, *counts), label
    return HGTMap.read_auto(path), "inferred"


def _trn_dir(args) -> Path | None:
    value = getattr(args, "trn_dir", None)
    return Path(value) if value else None


def _describe(hgt: HGTMap, source: str) -> dict:
    heights = hgt.heights
    return {
        "zones": f"{hgt.zones_x}x{hgt.zones_z}",
        "dim_source": source,
        "samples": int(heights.size),
        "min": int(heights.min()),
        "max": int(heights.max()),
        "levels": int(np.unique(heights).size),
        "flagged": int((hgt.flags != 0).sum()),
    }


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def cmd_convert(args) -> int:
    src = Path(args.input)
    out = Path(args.output) if args.output else src.with_suffix(".hg2")
    if out.exists() and not args.force:
        print(f"refusing to overwrite {out} (pass --force)", file=sys.stderr)
        return 1
    zones = tuple(int(v) for v in args.zones.lower().split("x")) if args.zones else None
    hgt, source = _load(src, zones, _trn_dir(args))
    hg2 = hgt.to_hg2(smoothing=args.smoothed, rounding=args.rounding,
                     map_version=args.map_version)
    out.parent.mkdir(parents=True, exist_ok=True)
    hg2.write(out)
    info = _describe(hgt, source)
    mode = "redux-smoothed" if args.smoothed else f"legacy-surface/{args.rounding}"
    print(f"{src.name} -> {out}  {info['zones']} zones ({source})  "
          f"heights {info['min']}..{info['max']}  {mode}")
    return 0


def cmd_batch(args) -> int:
    root = Path(args.root)
    out_dir = Path(args.output) if args.output else None
    rows = []
    ok = failed = skipped = 0
    for path in _iter_hgt(root):
        row = {"path": str(path), "name": path.name, "bytes": path.stat().st_size}
        try:
            hgt, source = _load(path, None, _trn_dir(args))
        except HGTFormatError as exc:
            row.update(status="unparsed", note=str(exc))
            rows.append(row)
            failed += 1
            continue
        row.update(_describe(hgt, source))
        candidates = zone_count_candidates(row["bytes"])
        row["ambiguous"] = int(source == "inferred" and len(candidates) > 1)
        if out_dir is None:
            row["status"] = "parsed"
            rows.append(row)
            ok += 1
            continue
        rel = path.relative_to(root) if root.is_dir() else Path(path.name)
        target = (out_dir / rel).with_suffix(".hg2")
        if target.exists() and not args.force:
            row.update(status="exists", note="target present; --force to overwrite")
            rows.append(row)
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        hgt.to_hg2(smoothing=args.smoothed, rounding=args.rounding,
                   map_version=args.map_version).write(target)
        row.update(status="converted", output=str(target), sha256_16=_sha(target))
        rows.append(row)
        ok += 1

    if args.csv:
        fields = ["path", "name", "bytes", "zones", "dim_source", "samples", "min", "max",
                  "levels", "flagged", "ambiguous", "status", "output", "sha256_16", "note"]
        with open(args.csv, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")

    verb = "converted" if out_dir is not None else "parsed"
    print(f"{verb}: {ok}   skipped: {skipped}   failed: {failed}   total: {len(rows)}")
    for row in rows:
        if row.get("status") in ("unparsed", "exists"):
            print(f"  [{row['status']}] {row['name']}: {row.get('note','')}")
        elif row.get("ambiguous"):
            print(f"  [ambiguous dims] {row['name']}: assumed {row['zones']} "
                  f"(no usable TRN; candidates {zone_count_candidates(row['bytes'])})")
    return 0 if failed == 0 else 2


def cmd_scan(args) -> int:
    """Census only: never writes terrain."""
    args.output = None
    return cmd_batch(args)


def cmd_compare(args) -> int:
    """Numerically compare a generated HG2 against a reference HG2."""
    from bzr_heightmap.hg2 import HG2Map

    def load(path):
        header = read_hg2_header(path)
        zone = 1 << header["zone_bits"]
        raw = np.fromfile(path, dtype="<u2", offset=12)
        zx, zz = header["zones_x"], header["zones_z"]
        tiled = raw.reshape(zz, zx, zone, zone)
        return header, tiled.transpose(0, 2, 1, 3).reshape(zz * zone, zx * zone).astype(np.int64)

    ha, a = load(args.a)
    hb, b = load(args.b)
    if a.shape != b.shape:
        print(f"shape mismatch: {a.shape} vs {b.shape}", file=sys.stderr)
        return 1
    d = a - b
    nz = int((d != 0).sum())
    print(f"A {args.a}  {ha['zones_x']}x{ha['zones_z']} mver={ha['map_version']}")
    print(f"B {args.b}  {hb['zones_x']}x{hb['zones_z']} mver={hb['map_version']}")
    print(f"differing samples : {nz} / {a.size} ({100.0 * nz / a.size:.4f}%)")
    print(f"max abs error     : {int(np.abs(d).max())} raw units")
    print(f"RMSE              : {float(np.sqrt((d.astype(np.float64) ** 2).mean())):.6f}")
    print(f"discrete levels   : A={np.unique(a).size}  B={np.unique(b).size}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add_conversion_flags(p):
        p.add_argument("--smoothed", action="store_true",
                       help="reproduce Redux's default HGT cook including its 3x3 box blur "
                            "(parity mode; NOT recommended for map preservation)")
        p.add_argument("--rounding", choices=ROUNDING_MODES, default="engine",
                       help="engine: bit-exact with the shipped interpolator (default). "
                            "half-up: exact integer plane, halves rounded up.")
        p.add_argument("--map-version", type=int, default=10,
                       help="HG2 map version field; Redux requires >= 10 (default: 10)")
        p.add_argument("--force", action="store_true", help="overwrite existing outputs")
        p.add_argument("--trn-dir",
                       help="directory of .trn files to resolve dimensions from when the "
                            "HGT has no TRN beside it (e.g. Redux StockODFFiles)")

    c = sub.add_parser("convert", help="convert a single HGT")
    c.add_argument("input")
    c.add_argument("-o", "--output")
    c.add_argument("--zones", help="override dimensions, e.g. 4x3")
    add_conversion_flags(c)
    c.set_defaults(func=cmd_convert)

    b = sub.add_parser("batch", help="convert every HGT under a directory")
    b.add_argument("root")
    b.add_argument("-o", "--output", required=True, help="output directory (mirrors structure)")
    b.add_argument("--csv", help="write a per-file conversion report")
    add_conversion_flags(b)
    b.set_defaults(func=cmd_batch)

    s = sub.add_parser("scan", help="census local HGT files without writing anything")
    s.add_argument("root")
    s.add_argument("--csv")
    add_conversion_flags(s)
    s.set_defaults(func=cmd_scan)

    p = sub.add_parser("compare", help="numerically compare two HG2 files")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
