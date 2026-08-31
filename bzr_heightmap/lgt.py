"""Battlezone LGT (terrain lighting) handling and preview generation.

Classic BZ documentation (battlezone.videoventure.org/format_lgt.html) defines
LGT as::

    - no header
    - one block per terrain zone
    - each zone = 128 x 128 entries
    - row-major, southwest origin
    - one unsigned 8-bit light value per entry
    - 0 = minimum lighting (25% ambient brightness)
    - 255 = maximum / 100% brightness

Redux observation from local corpus (2026-08-28 scan of this PC):

    - HG2 uses ``zone_bits=8`` -> 256 x 256 height *vertices* per 1280-unit
      zone at 5 m spacing.  Verified on 507 valid HG2 files on this machine.

    - LGT files on this PC are overwhelmingly 256 x 256 *per zone* as well
      (433 of 447 valid HG2/LGT path pairs),
      stored as ``(zones+1) * zone_size * zone_size`` bytes where the first
      ``zone_size*zone_size`` block is a border chunk (filled with the
      southwest/file-origin corner value). This matches Z64Tools ``terrain_pack.py``
      behaviour: ``border + zoned grayscale lightmap chunks'' after an N/S
      flip.  For example 2 x 2 zones at 256 yields 5*65536 = 327680 bytes,
      observed for ``StockODFFiles/misn01.lgt`` and ``addon/ISDF Chronicles/*.lgt``.

    - Nine path pairs (five unique HG2 contents, including ``evolve_*.lgt``)
      use classic 128 x 128 per zone even though the companion HG2 is 256.
      Five additional path pairs have sizes that do not reconcile with their
      companion HG2 dimensions and remain explicitly unrecognized.

    - Therefore the authoritative Redux relationship is:

          HG2 vertices : 256 x 256 per zone at 5 m
          LGT cells    : 256 x 256 per zone at 5 m (high-res) OR
                         128 x 128 per zone at 10 m (classic)

      The preview code here defaults to the classic 10 m cell mapping
      (LGT = HG2 // 2) because the task explicitly warns *not* to naively
      treat a 256 HG2 vertex map as a 256 LGT, and because a 128-per-zone
      cell grid is the documented terrain-cell grid (10 m).  The high-res
      256 path is also implemented and is selectable when evidence shows the
      file on disk uses it.

Orientation notes:

    - Both HG2 and LGT zone blocks are row-major with southwest origin.
      ``bzr_heightmap.hg2.HG2Map`` stores the assembled height array with
      row 0 = south, row increasing = north, col 0 = west.  This matches the
      file zone order (zone_z outer, zone_x inner).
    - Z64Tools flips a conventional north-at-top PNG before zoning it.
      This module's arrays already use HG2's south-first row order, so writing
      them directly produces the same bytes. Applying another flip here would
      invert the LGT relative to its HG2. Preview and file I/O therefore keep
      a single south-first in-memory convention.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from bzr_heightmap.hg2 import BZ_ZONE_WORLD_SIZE, HG2Map

# Horizontal / vertical units for slope.
VERTICAL_UNIT_SCALE_M = 0.1  # 1 HG2 height unit = 0.1 world units

# LGT ambient floor: 0 in file = 25% brightness, 255 = 100%.
# Linear mapping: brightness = 0.25 + 0.75 * (value/255)
LGT_AMBIENT_FRACTION = 0.25
LGT_RANGE_FRACTION = 0.75

# Default sun for preview (matches analysis.make_preview hillshade)
DEFAULT_SUN_AZIMUTH_DEG = 315.0  # northwest, as used by make_preview
DEFAULT_SUN_ALTITUDE_DEG = 45.0


def _sun_direction(azimuth_deg: float, altitude_deg: float) -> tuple[float, float, float]:
    """Return the unit vector from terrain toward the sun (east, up, north)."""
    az = math.radians(float(azimuth_deg))
    alt = math.radians(float(altitude_deg))
    # World: +X east, +Z north, +Y up.
    # Azimuth 0 = north, 90 = east, increasing clockwise.
    # For hillshade compatibility we use the same convention as analysis.py:
    #   aspect = atan2(-gx, gy)  (slope facing)
    #   shade = sin(alt)*sin(slope) + cos(alt)*cos(slope)*cos(az - aspect)
    # This is the standard GIS hillshade formulation.
    # Convert to vector (x, y, z):
    #   sun_x = cos(alt) * sin(az)
    #   sun_z = cos(alt) * cos(az)
    #   sun_y = sin(alt)
    x = math.cos(alt) * math.sin(az)
    z = math.cos(alt) * math.cos(az)
    y = math.sin(alt)
    return x, y, z


def compute_lgt_lightmap(
    heights: np.ndarray,
    zones_x: int | None = None,
    zones_z: int | None = None,
    *,
    lgt_zone_size: int = 128,
    sun_azimuth_deg: float = DEFAULT_SUN_AZIMUTH_DEG,
    sun_altitude_deg: float = DEFAULT_SUN_ALTITUDE_DEG,
    ambient: float = LGT_AMBIENT_FRACTION,
) -> np.ndarray:
    """Compute a BZ LGT-style lighting field from HG2 heights.

    Parameters
    ----------
    heights:
        HG2 height array, shape ``(zones_z * hg_zone_size, zones_x * hg_zone_size)``.
        Values are 0..4095.  Orientation is HG2Map native (row 0 = south).
    zones_x, zones_z:
        Inferred from heights if not provided.  Required to verify zone_size.
    lgt_zone_size:
        LGT samples per zone side.  Classic 128 (10 m cells) or Redux 256
        (5 m vertices).  Default 128 as documented.
    sun_azimuth_deg, sun_altitude_deg:
        Sun direction for Lambert shading.
    ambient:
        Ambient floor fraction (0.25 per spec).

    Returns
    -------
    np.ndarray
        uint8 array shape ``(zones_z * lgt_zone_size, zones_x * lgt_zone_size)``,
        row 0 = south, values 0..255 where 0 = ambient-only.
    """
    a = np.asarray(heights, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError("heights must be 2-D")
    h, w = a.shape
    if not 0.0 <= float(ambient) < 1.0:
        raise ValueError("ambient must be in the range 0..1")
    if zones_x is None or zones_z is None:
        # Infer from shape assuming hg_zone_size = 256
        hg_zone_size = 1 << 8  # 256
        if h % hg_zone_size != 0 or w % hg_zone_size != 0:
            raise ValueError(f"Cannot infer zones from shape {a.shape}")
        zones_z, zones_x = h // hg_zone_size, w // hg_zone_size
    else:
        zones_x, zones_z = int(zones_x), int(zones_z)
    if zones_x <= 0 or zones_z <= 0:
        raise ValueError("zone counts must be positive")
    if h % zones_z != 0 or w % zones_x != 0:
        raise ValueError("height shape is not divisible by zone counts")
    hg_zone_size_z = h // zones_z
    hg_zone_size_x = w // zones_x
    if hg_zone_size_x != hg_zone_size_z:
        raise ValueError("Non-square zone layout")
    hg_zone_size = hg_zone_size_x
    if hg_zone_size not in (128, 256):
        raise ValueError(f"Unsupported HG2 zone size {hg_zone_size}; expected 128 or 256")
    lgt_zone_size = int(lgt_zone_size)
    if lgt_zone_size not in (128, 256) or lgt_zone_size > hg_zone_size or hg_zone_size % lgt_zone_size:
        raise ValueError("lgt_zone_size must be 128 or 256 and divide the HG2 zone size")

    # World +X is east and array rows increase north. The upward surface
    # normal is (-dh/dx, 1, -dh/dz). This direct dot product makes the
    # southwest/file-row convention explicit and avoids an aspect-sign trap.
    vertex_spacing = BZ_ZONE_WORLD_SIZE / float(hg_zone_size)
    grad_z, grad_x = np.gradient(a * VERTICAL_UNIT_SCALE_M, vertex_spacing, vertex_spacing)
    normal_length = np.sqrt(grad_x * grad_x + grad_z * grad_z + 1.0)
    sun_x, sun_y, sun_z = _sun_direction(sun_azimuth_deg, sun_altitude_deg)
    lambert = (-grad_x * sun_x + sun_y - grad_z * sun_z) / normal_length
    lambert = np.clip(lambert, 0.0, 1.0)

    # Downsample to LGT cell resolution if needed.
    # HG2 vertices 256 per zone vs LGT cells 128 per zone => factor 2.
    factor = hg_zone_size // lgt_zone_size
    if factor == 1:
        lambert_lgt = lambert
    elif factor > 1:
        # Block-average lambert over factor x factor vertex blocks.
        # This corresponds to one LGT cell covering factor x factor vertices.
        # For factor 2 (256->128) each cell is 10 m covering 2 vertex steps.
        # Use ndimage uniform filter style block reduction via reshape.
        # Ensure divisible
        if h % factor != 0 or w % factor != 0:
            raise ValueError("HG2 size not divisible by LGT factor")
        hh, ww = h // factor, w // factor
        # Reshape to (hh, factor, ww, factor) then mean
        # Need to ensure we reshape correctly: rows are contiguous
        lambert_lgt = lambert.reshape(hh, factor, ww, factor).mean(axis=(1, 3))
    else:
        raise ValueError("lgt_zone_size larger than hg_zone_size not supported for preview")

    # Map lambert 0..1 to LGT 0..255 with ambient floor.
    # Spec: 0 = 25% ambient, 255 = 100% . So LGT value encodes brightness linearly
    # between ambient and full. If we interpret lambert as 0=ambient only,
    # 1=full sun, then value = lambert * 255 . But also need to ensure
    # flat terrain under sun yields ~180 as observed. With sun altitude 45,
    # lambert for flat (slope=0) = sin(alt)=0.707. So flat => 180.
    # That matches observed mean ~190-203 for typical maps.
    lgt = np.clip(np.rint(lambert_lgt * 255.0), 0, 255).astype(np.uint8)
    # The byte encodes illumination above the ambient baseline. A byte value
    # of zero means ambient-only, not black; lgt_to_brightness applies that
    # documented 25% floor for display/validation.
    return lgt


def lgt_to_brightness(lgt: np.ndarray, ambient: float = LGT_AMBIENT_FRACTION) -> np.ndarray:
    """Convert LGT 0..255 values to linear brightness 0.25..1.0."""
    a = np.asarray(lgt, dtype=np.float32) / 255.0
    return ambient + (1.0 - ambient) * a


def write_lgt(path: str | Path, lightmap: np.ndarray, zones_x: int, zones_z: int) -> None:
    """Write a Redux-style LGT file (border chunk + zoned blocks).

    This replicates the layout observed on disk and written by Z64Tools
    terrain_pack.py: first zone_size*zone_size bytes are a border chunk,
    followed by row-major zone blocks starting at the southwest corner.
    Z64Tools flips a north-at-top PNG first; this API already accepts a
    south-first array and therefore writes it without another flip.

    Parameters
    ----------
    path:
        Output .lgt path.
    lightmap:
        2-D uint8 array shape ``(zones_z*zone_size, zones_x*zone_size)``,
        row 0 = south, same orientation as HG2Map.heights.
    zones_x, zones_z:
        Zone counts, must match lightmap shape.
    """
    lm = np.asarray(lightmap, dtype=np.uint8)
    if lm.ndim != 2:
        raise ValueError("lightmap must be 2-D")
    zones_x, zones_z = int(zones_x), int(zones_z)
    if zones_x <= 0 or zones_z <= 0:
        raise ValueError("zone counts must be positive")
    h, w = lm.shape
    if h % zones_z != 0 or w % zones_x != 0:
        raise ValueError("lightmap shape not divisible by zones")
    zone_size_x = w // zones_x
    zone_size_z = h // zones_z
    if zone_size_x != zone_size_z:
        raise ValueError("non-square LGT zones not supported")
    zone_size = int(zone_size_x)
    if zone_size not in (128, 256):
        raise ValueError("LGT zone size must be 128 or 256")
    if lm.shape != (zones_z * zone_size, zones_x * zone_size):
        raise ValueError("lightmap shape mismatch")

    border = int(lm[0, 0])
    out = bytearray(bytes([border]) * (zone_size * zone_size))
    for zy in range(zones_z):
        for zx in range(zones_x):
            zone = lm[zy * zone_size : (zy + 1) * zone_size, zx * zone_size : (zx + 1) * zone_size]
            out.extend(zone.tobytes())
    Path(path).write_bytes(out)


def _companion_hg2_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        for candidate in path.parent.iterdir():
            if candidate.is_file() and candidate.stem.casefold() == path.stem.casefold() and candidate.suffix.casefold() == ".hg2":
                hg2 = HG2Map.read(candidate)
                return hg2.zones_x, hg2.zones_z
    except (OSError, ValueError):
        return None
    return None


def read_lgt(
    path: str | Path,
    zones_x: int | None = None,
    zones_z: int | None = None,
    *,
    zone_size: int | None = None,
) -> tuple[np.ndarray, int, int, int]:
    """Read an LGT file, returning (lightmap, zones_x, zones_z, zone_size).

    Handles bordered Redux and unbordered legacy layouts. Zone dimensions are
    taken from explicit arguments or a same-stem companion HG2. File size
    alone cannot distinguish 1x4, 2x2, and 4x1 maps, so ambiguous standalone
    files raise instead of silently choosing the wrong layout. Returns a
    south-first lightmap matching ``HG2Map.heights``.
    """
    data = Path(path).read_bytes()
    n = len(data)
    arr = np.frombuffer(data, dtype=np.uint8)
    lgt_path = Path(path)
    if (zones_x is None) != (zones_z is None):
        raise ValueError("zones_x and zones_z must be provided together")
    if zones_x is None:
        companion = _companion_hg2_dimensions(lgt_path)
        if companion is not None:
            zones_x, zones_z = companion

    candidate_zone_sizes = (int(zone_size),) if zone_size is not None else (256, 128)
    if any(size not in (128, 256) for size in candidate_zone_sizes):
        raise ValueError("LGT zone size must be 128 or 256")
    candidates: list[tuple[int, int, int, bool]] = []
    x_values = (int(zones_x),) if zones_x is not None else range(1, 9)
    z_values = (int(zones_z),) if zones_z is not None else range(1, 9)
    for zs in candidate_zone_sizes:
        for zx in x_values:
            for zz in z_values:
                if n == zx * zz * zs * zs:
                    candidates.append((zx, zz, zs, False))
                if n == (zx * zz + 1) * zs * zs:
                    candidates.append((zx, zz, zs, True))
    if not candidates:
        raise ValueError(f"LGT size {n} does not match the requested or inferred dimensions")
    best_rank = max((candidate[3], candidate[2]) for candidate in candidates)
    finalists = [candidate for candidate in candidates if (candidate[3], candidate[2]) == best_rank]
    if len(finalists) != 1:
        possibilities = ", ".join(f"{zx}x{zz}@{zs}" for zx, zz, zs, _ in finalists)
        raise ValueError(f"Ambiguous LGT dimensions ({possibilities}); provide zones_x and zones_z")
    zones_x, zones_z, zone_size, has_border = finalists[0]
    if has_border:
        content = arr[zone_size * zone_size :]
    else:
        content = arr
    lightmap = np.empty((zones_z * zone_size, zones_x * zone_size), dtype=np.uint8)
    idx = 0
    for zy in range(zones_z):
        for zx in range(zones_x):
            block = content[idx : idx + zone_size * zone_size]
            idx += zone_size * zone_size
            lightmap[zy * zone_size : (zy + 1) * zone_size, zx * zone_size : (zx + 1) * zone_size] = block.reshape(
                zone_size, zone_size
            )
    return lightmap, zones_x, zones_z, zone_size


def compute_lgt_for_hg2(hg2: HG2Map, lgt_zone_size: int = 128) -> np.ndarray:
    """Convenience: compute LGT preview for a HG2Map instance."""
    return compute_lgt_lightmap(
        hg2.heights, hg2.zones_x, hg2.zones_z, lgt_zone_size=lgt_zone_size
    )
