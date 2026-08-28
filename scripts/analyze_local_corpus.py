#!/usr/bin/env python3
"""Reusable local HG2 corpus analyzer.

Discovers .hg2 files case-insensitively, parses them, deduplicates by content
hash, computes BZ terrain-shape metrics, and emits machine-readable JSON/CSV
plus a human-readable summary.

Never writes to discovered source HG2 files and never copies proprietary
terrain into the repository.  Aggregate statistics only.

Usage:
    python scripts/analyze_local_corpus.py --discover
    python scripts/analyze_local_corpus.py --roots "C:\\path" "D:\\other" --out output.json
    python scripts/analyze_local_corpus.py --hg2 path/to/map.hg2 --out report.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

# Ensure package importable when run as script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bzr_heightmap.analysis import describe_heightmap
from bzr_heightmap.hg2 import HG2Map

EXCLUDE_DIR_NAMES = {
    "windows",
    "programdata",
    "appdata",
    "$recycle.bin",
    "system volume information",
    "recovery",
    "perflogs",
    "__pycache__",
    "node_modules",
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "cache",
}

GENERATED_SAMPLE = "generated_sample"
SYNTHETIC_TEST = "synthetic_test"
AUTHORED = "authored"
SYNTHETIC_TEST_STEMS = {"bz14atk", "lcbench", "magnz0", "pilot", "wmtest0"}


def normalized_path_parts(path: Path) -> tuple[str, ...]:
    """Return case-folded path components independent of host separators.

    ``Path.parts`` alone cannot split a Windows path when tests run on a
    non-Windows host. Normalizing both separator styles first keeps corpus
    classification deterministic on Windows and in CI.
    """
    return tuple(part.casefold() for part in str(path).replace("\\", "/").split("/") if part)


def _contains_sequence(parts: tuple[str, ...], sequence: tuple[str, ...]) -> bool:
    width = len(sequence)
    return any(parts[index : index + width] == sequence for index in range(len(parts) - width + 1))


def discover_hg2_files(roots: Iterable[Path]) -> list[Path]:
    """Discover .hg2 files case-insensitively under roots.

    Skips network drives, recycle bins, system internals, python envs, etc.
    Handles access-denied cleanly.
    """
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root, topdown=True, onerror=lambda e: None):
            low = dirpath.lower()
            # Skip Windows system internals entirely
            if low.startswith("c:\\windows") or "system volume information" in low or "$recycle.bin" in low:
                dirnames[:] = []
                continue
            if "appdata" in low:
                dirnames[:] = []
                continue
            # Prune excluded dir names (case-insensitive)
            dirnames[:] = [d for d in dirnames if d.lower() not in EXCLUDE_DIR_NAMES and not d.startswith(".")]
            # Also prune obvious cache trees
            dirnames[:] = [d for d in dirnames if "cache" not in d.lower() or "battlezone" in low]
            for fname in filenames:
                if fname.lower().endswith(".hg2"):
                    try:
                        found.append(Path(dirpath) / fname)
                    except Exception:
                        continue
    return found


def discover_default_roots() -> list[Path]:
    """Discover sensible default roots per task spec."""
    roots: list[Path] = []
    # Fixed drives via Win32_LogicalDisk DriveType=3 (local fixed)
    try:
        import subprocess

        out = subprocess.check_output(
            'Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" | Select-Object -ExpandProperty DeviceID',
            shell=True,
            text=True,
        )
        drives = [d.strip() for d in out.splitlines() if d.strip().endswith(":")]
        for d in drives:
            p = Path(d + "\\")
            if p.exists():
                roots.append(p)
    except Exception:
        roots.append(Path("C:\\"))
    # Known Battlezone related roots (will be covered by drives but ensure)
    candidates = [
        Path.home() / "Documents" / "GIT",
        Path(r"C:\Program Files (x86)\Steam"),
        Path(r"C:\Program Files (x86)\GOG Galaxy\Games"),
        Path.home() / "Documents",
    ]
    for c in candidates:
        if c.exists() and c not in roots:
            roots.append(c)
    return roots


def find_companion(hg2_path: Path, extension: str) -> Path | None:
    """Find a same-stem companion file without assuming extension case."""
    stem = hg2_path.stem
    parent = hg2_path.parent
    try:
        for p in parent.iterdir():
            if p.is_file() and p.stem.casefold() == stem.casefold() and p.suffix.casefold() == extension.casefold():
                return p
    except OSError:
        return None
    return None


def infer_lgt_layout(byte_count: int, zones_x: int, zones_z: int) -> str:
    """Classify an LGT size using its companion HG2 zone dimensions."""
    zone_count = int(zones_x) * int(zones_z)
    for zone_size in (256, 128, 64):
        block = zone_size * zone_size
        if byte_count == (zone_count + 1) * block:
            return f"bordered_{zone_size}"
        if byte_count == zone_count * block:
            return f"unbordered_{zone_size}"
    return "unrecognized"


def companion_assets(hg2_path: Path, zones_x: int | None = None, zones_z: int | None = None) -> dict[str, bool | str | int | None]:
    """Look for nearby .TRN/.BZN/.LGT/.MAT and infer provenance."""
    companions = {ext: find_companion(hg2_path, f".{ext}") for ext in ("trn", "bzn", "lgt", "mat")}
    lgt_path = companions["lgt"]
    lgt_size = None
    lgt_layout = None
    if lgt_path is not None:
        try:
            lgt_size = lgt_path.stat().st_size
            if zones_x is not None and zones_z is not None:
                lgt_layout = infer_lgt_layout(lgt_size, zones_x, zones_z)
        except OSError:
            lgt_size = None

    prov = classify_provenance(hg2_path)
    return {
        "has_trn": companions["trn"] is not None,
        "has_bzn": companions["bzn"] is not None,
        "has_lgt": lgt_path is not None,
        "has_mat": companions["mat"] is not None,
        "lgt_size": lgt_size,
        "lgt_layout": lgt_layout,
        "provenance": prov,
        "terrain_class": classify_terrain_class(hg2_path),
    }


def classify_provenance(path: Path) -> str:
    parts = normalized_path_parts(path)
    joined = "/".join(parts)
    terrain_class = classify_terrain_class(path)
    if terrain_class == GENERATED_SAMPLE:
        return "generated_sample"
    if terrain_class == SYNTHETIC_TEST:
        return "test_synthetic"
    if "bzonezfsodf" in parts or "stockodffiles" in parts:
        return "stock"
    if "isdf chronicles" in parts:
        return "isdf_chronicles"
    if "rotbdv2" in joined or "rotbd-master" in joined:
        return "rotbd"
    if "campaignreimagined" in joined or "missions/misn" in joined:
        return "campaign"
    if "workshop" in parts:
        return "workshop"
    if "addon" in parts:
        return "addon"
    if any("bzp" in part for part in parts):
        return "bzp"
    return "unknown"


def classify_terrain_class(path: Path) -> str:
    """Classify corpus paths into generated, synthetic/test, or authored.

    Generated samples are recognized by normalized path components rather
    than a slash-sensitive substring. Synthetic fixtures are limited to the
    OpenShim test-mission tree and the known copied fixture stems found in the
    local corpus; broad words such as ``pilot`` elsewhere do not match.
    """
    parts = normalized_path_parts(path)
    for index, part in enumerate(parts):
        if part.endswith("heightmapgen") and _contains_sequence(parts[index + 1 :], ("samples", "hg2")):
            return GENERATED_SAMPLE
    stem = parts[-1].rsplit(".", 1)[0] if parts else ""
    if "test_missions" in parts or stem in SYNTHETIC_TEST_STEMS:
        return SYNTHETIC_TEST
    return AUTHORED


def analyze_file(path: Path, metric_cache: dict[str, dict[str, object]] | None = None) -> dict:
    # Discovery may receive a relative root (for example ``samples/hg2``).
    # Resolve before path-based classification so the same file has the same
    # provenance regardless of how the caller spelled its root.
    path = path.resolve()
    try:
        hg = HG2Map.read(path)
        h = hashlib.sha256(hg.heights.astype("<u2", copy=False).tobytes()).hexdigest()
        if metric_cache is not None and h in metric_cache:
            desc = metric_cache[h]
        else:
            desc = describe_heightmap(hg.heights)
            if metric_cache is not None:
                metric_cache[h] = desc
        info = {
            "path": str(path),
            "sha256": h,
            "valid": True,
            "zones_x": hg.zones_x,
            "zones_z": hg.zones_z,
            "zone_bits": hg.zone_bits,
            "shape": list(hg.heights.shape),
        }
        info.update(desc)  # includes all new metrics
        info.update(companion_assets(path, hg.zones_x, hg.zones_z))
        return info
    except Exception as exc:
        return {"path": str(path), "valid": False, "error": str(exc), **companion_assets(path)}


def aggregate_report(records: list[dict]) -> dict:
    valid = [r for r in records if r.get("valid")]
    invalid = [r for r in records if not r.get("valid")]
    # Deduplicate by sha256
    hash_to = {}
    for r in valid:
        h = r["sha256"]
        hash_to.setdefault(h, []).append(r)
    unique = len(hash_to)
    # duplicate groups
    dup_groups = sum(1 for v in hash_to.values() if len(v) > 1)
    duplicate_paths = sum(len(v) - 1 for v in hash_to.values() if len(v) > 1)
    # dimensions and path-level classifications
    dims = Counter((r["zones_x"], r["zones_z"]) for r in valid)
    prov = Counter(r.get("provenance", "unknown") for r in valid)
    path_classes = Counter(r.get("terrain_class", AUTHORED) for r in valid)

    # Classify each unique content group. Authored wins if the same content is
    # present in both an authored and a test/generated path; that content is
    # not unique to the generator/fixture corpus.
    unique_records_by_class: dict[str, list[dict]] = {AUTHORED: [], GENERATED_SAMPLE: [], SYNTHETIC_TEST: []}
    unique_groups_by_class: dict[str, list[list[dict]]] = {AUTHORED: [], GENERATED_SAMPLE: [], SYNTHETIC_TEST: []}
    for grouped in hash_to.values():
        classes = {r.get("terrain_class", AUTHORED) for r in grouped}
        if AUTHORED in classes:
            terrain_class = AUTHORED
        elif SYNTHETIC_TEST in classes:
            terrain_class = SYNTHETIC_TEST
        else:
            terrain_class = GENERATED_SAMPLE
        representative = next((r for r in grouped if r.get("terrain_class") == terrain_class), grouped[0])
        unique_records_by_class[terrain_class].append(representative)
        unique_groups_by_class[terrain_class].append(grouped)

    uniq_records = [r for grouped in unique_records_by_class.values() for r in grouped]

    def means(source: list[dict]) -> dict[str, float]:
        def mean(key: str) -> float:
            vals = [r[key] for r in source if key in r and isinstance(r[key], (int, float))]
            return float(np.mean(vals)) if vals else 0.0

        return {
            "exact_flat_pct": mean("exact_flat_pct"),
            "dominant_level_pct": mean("dominant_level_pct"),
            "median_slope_deg": mean("median_slope_deg"),
            "p95_slope_deg": mean("p95_slope_deg"),
            "range": mean("range"),
            "shelf_count_gt2pct": mean("shelf_count_gt2pct"),
            "shelf_area_pct": mean("shelf_area_pct"),
            "largest_flat_component_pct": mean("largest_flat_component_pct"),
            "corridor_median_width_m": mean("corridor_median_width_m"),
        }

    lgt_layouts = Counter(r.get("lgt_layout") or "uninspected" for r in valid if r.get("has_lgt"))
    unique_with_lgt = sum(1 for grouped in hash_to.values() if any(r.get("has_lgt") for r in grouped))
    unique_authored_with_lgt = sum(1 for grouped in unique_groups_by_class[AUTHORED] if any(r.get("has_lgt") for r in grouped))
    unique_lgt_layout_sets = Counter()
    for grouped in hash_to.values():
        layouts = sorted({str(r.get("lgt_layout") or "uninspected") for r in grouped if r.get("has_lgt")})
        if layouts:
            unique_lgt_layout_sets["+".join(layouts)] += 1
    unique_counts = {terrain_class: len(grouped) for terrain_class, grouped in unique_records_by_class.items()}
    if sum(unique_counts.values()) != unique:
        raise AssertionError("unique terrain classes do not reconcile")

    if len(valid) != unique + duplicate_paths:
        raise AssertionError("valid path, unique content, and duplicate counts do not reconcile")

    summary = {
        "total_paths": len(records),
        "valid": len(valid),
        "invalid": len(invalid),
        "unique_content_hashes": unique,
        "duplicate_groups": dup_groups,
        "duplicate_paths_extra": duplicate_paths,
        "path_classification": dict(path_classes),
        "unique_classification": unique_counts,
        "dimensions": {f"{k[0]}x{k[1]}": v for k, v in dims.items()},
        "provenance": dict(prov),
        "aggregate_metrics": means(uniq_records),
        "authored_aggregate_metrics": means(unique_records_by_class[AUTHORED]),
        "companion_lgt_pairs": sum(1 for r in valid if r.get("has_lgt")),
        "unique_content_with_lgt": unique_with_lgt,
        "unique_authored_with_lgt": unique_authored_with_lgt,
        "lgt_layouts": dict(lgt_layouts),
        "unique_lgt_layout_sets": dict(unique_lgt_layout_sets),
    }
    if summary["total_paths"] != summary["valid"] + summary["invalid"]:
        raise AssertionError("total path accounting does not reconcile")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze local HG2 corpus")
    parser.add_argument("--discover", action="store_true", help="discover via default roots (fixed drives)")
    parser.add_argument("--roots", nargs="*", type=Path, help="explicit roots to scan")
    parser.add_argument("--hg2", nargs="*", type=Path, help="explicit HG2 files")
    parser.add_argument("--out", type=Path, help="output JSON path")
    parser.add_argument("--csv", type=Path, help="output CSV path")
    parser.add_argument("--summary", type=Path, help="output human-readable summary markdown")
    args = parser.parse_args()

    paths: list[Path] = []
    if args.hg2:
        paths.extend([p for p in args.hg2 if p.exists()])
    if args.roots:
        for r in args.roots:
            paths.extend(discover_hg2_files([r]))
    if args.discover or (not args.roots and not args.hg2):
        # Controlled discovery: limit to sensible roots to avoid scanning entire C quickly
        # We scan Documents, GIT, Steam, GOG
        controlled = [
            Path.home() / "Documents",
            Path(r"C:\Program Files (x86)\Steam\steamapps"),
            Path(r"C:\Program Files (x86)\GOG Galaxy\Games"),
        ]
        controlled = [p for p in controlled if p.exists()]
        if controlled:
            for r in controlled:
                paths.extend(discover_hg2_files([r]))
        else:
            for r in discover_default_roots():
                paths.extend(discover_hg2_files([r]))

    # Deduplicate paths (resolved)
    resolved = {}
    for p in paths:
        try:
            rp = p.resolve()
            resolved[str(rp).casefold()] = p
        except Exception:
            resolved[str(p).casefold()] = p
    paths = sorted(resolved.values(), key=lambda path: str(path).casefold())

    metric_cache: dict[str, dict[str, object]] = {}
    records = [analyze_file(p, metric_cache) for p in paths]
    summary = aggregate_report(records)

    # Output JSON
    out_json = args.out or (ROOT / "output" / "hg2_corpus_report.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8")
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = out_json.with_suffix(".csv")
    # Write CSV for valid records
    valid = [r for r in records if r.get("valid")]
    if valid:
        keys = ["path", "sha256", "zones_x", "zones_z", "exact_flat_pct", "dominant_level_pct", "median_slope_deg", "p95_slope_deg", "range", "shelf_count_gt2pct", "terrain_class", "provenance", "has_lgt", "lgt_layout"]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in valid:
                w.writerow({k: r.get(k, "") for k in keys})

    summary_path = args.summary or (ROOT / "docs" / "HG2_CORPUS_SUMMARY.md")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    # Avoid committing absolute user paths: sanitize summary to not include full paths
    lines = [
        "# HG2 Corpus Summary (local discovery)",
        "",
        f"- total paths discovered: {summary['total_paths']}",
        f"- valid parses: {summary['valid']}",
        f"- invalid/corrupt: {summary['invalid']}",
        f"- unique content hashes: {summary['unique_content_hashes']}",
        f"- duplicate groups: {summary['duplicate_groups']} (extra duplicate paths {summary['duplicate_paths_extra']})",
        f"- path classification: {summary['path_classification']}",
        f"- unique classification: {summary['unique_classification']}",
        f"- companion HG2/LGT path pairs: {summary['companion_lgt_pairs']}",
        f"- unique contents with LGT: {summary['unique_content_with_lgt']}",
        f"- unique authored contents with LGT: {summary['unique_authored_with_lgt']}",
        f"- LGT layouts by valid path pair: {summary['lgt_layouts']}",
        f"- LGT layout sets by unique HG2 content: {summary['unique_lgt_layout_sets']}",
        f"- dimensions: {summary['dimensions']}",
        f"- provenance: {summary['provenance']}",
        "",
        "## Authored aggregate metrics (mean over unique authored contents)",
    ]
    for k, v in summary["authored_aggregate_metrics"].items():
        lines.append(f"- {k}: {v:.2f}")
    lines.append("")
    lines.append(f"Records written to {out_json.name} and {csv_path.name} (paths sanitized in summary).")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_json} and {csv_path} and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
