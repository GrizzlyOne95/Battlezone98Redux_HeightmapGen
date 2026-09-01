"""Legacy Battlezone 1.x ``.HGT`` terrain, and its conversion to Redux ``.HG2``.

Format and algorithm notes are transcribed from the shipped GOG Battlezone 98
Redux executable (2.2.301, ``battlezone98redux.exe``, image base 0x400000).
Every constant below is tied to a specific address; nothing here is guessed.

Legacy HGT
----------
No header. ``nZones * 0x8000`` bytes of little-endian ``uint16``, laid out as
zone-major then row-major inside each zone, with 128x128 samples per zone
(``FUN_00785f50``, the engine's legacy sample fetch)::

    index = (z & 127) * 128 + (x & 127)
          + (x >> 7) * 0x4000
          + (z >> 7) * zones_x * 0x4000

Only the low 12 bits are height; ``FUN_00785f50`` masks every fetch with
``& 0xFFF``. The upper nibble carries non-height flags and is discarded by the
engine, so it is discarded here too.

The zone counts are *not* in the file. Redux takes them from the companion
``.TRN`` (``FUN_00786340``): ``zones = (int)(Width * 0.1) >> 7``, validated
against ``Width == zones * 1280``. One zone is therefore 1280 world units
across: 128 legacy samples at 10 world units, or 256 HG2 samples at 5.

Redux HG2
---------
12-byte header then ``nZones * 0x20000`` bytes of ``uint16``, 256x256 per zone,
same zone-major/row-major tiling (``FUN_00785c00`` returns
``base + 0xC + zone * 0x20000``). See :mod:`bzr_heightmap.hg2`.

The cook
--------
``FUN_00786340`` tries ``<mission>.hg2`` first and only falls back to
``<mission>.HGT`` when the HG2 is absent or its header fails validation. The
HGT path then runs ``FUN_00786200``: a pure 2x upsample that samples the legacy
grid at ``(out * 0.5)`` through ``FUN_00785fe0``, followed -- only when the
``-nohgtsmoothing`` command-line switch was not given -- by ``FUN_00785c80``,
a 3x3 box blur.

``FUN_00785fe0`` is *piecewise-planar*, not bilinear: it splits each legacy
cell along its (0,0)-(1,1) diagonal and evaluates the plane through the three
vertices of whichever triangle contains the sample. That is exactly the surface
the legacy engine rendered, so the upsample itself is not a loss of fidelity --
it reproduces the authored terrain between vertices and reproduces authored
vertices exactly. The 3x3 box blur is the destructive step, and it is the only
thing this module's default conversion omits.
"""
from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .hg2 import HG2Map, DEFAULT_ZONE_BITS, HG2_MAP_VERSION, HG2_STRUCTURE_VERSION

# --- legacy grid geometry (FUN_00785f50) ---------------------------------
LEGACY_ZONE_BITS = 7
LEGACY_ZONE_SIZE = 1 << LEGACY_ZONE_BITS          # 128 samples per zone edge
LEGACY_ZONE_BYTES = LEGACY_ZONE_SIZE * LEGACY_ZONE_SIZE * 2   # 0x8000
LEGACY_HEIGHT_MASK = 0x0FFF                        # `& 0xfff` in FUN_00785f50

# --- world scale (FUN_00786340 zone maths, FUN_007859d0 default fill) ----
ZONE_WORLD_SIZE = 1280.0        # Width == zones_x * 0x500
LEGACY_SAMPLE_SPACING = ZONE_WORLD_SIZE / LEGACY_ZONE_SIZE     # 10.0
HG2_SAMPLE_SPACING = ZONE_WORLD_SIZE / (1 << DEFAULT_ZONE_BITS)  # 5.0
# FUN_007859d0 fills default terrain with `(short)(Height / 0.1)`, so one raw
# height unit is 0.1 world units. Confidence: high (single call site, but the
# only place the engine converts an authored world height into a raw sample).
HEIGHT_UNIT_WORLD = 0.1

# --- interpolator constants (FUN_00785fe0, .rdata 0x008a2538) ------------
_F32 = np.float32
_K = _F32(0.10000000149011612)      # dword 0x008a2538, i.e. 0.1f
_KK = _F32(_K * _K)                 # [ebp-0x1c]; note this is *not* 0.01f
_TEN = _F32(10.0)                   # cvtsi2ss of the literal 0xA

ROUNDING_MODES = ("engine", "half-up")


class HGTFormatError(ValueError):
    """Raised when a file cannot be interpreted as legacy HGT terrain."""


# ---------------------------------------------------------------------------
# TRN dimensions
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(r"^\s*\[(?P<name>[^\]]*)\]")


def read_trn_zone_counts(path: os.PathLike | str) -> Optional[Tuple[int, int]]:
    """Zone counts from a ``.TRN``'s first ``[Size]`` section, or ``None``.

    Redux reads ``Width``/``Depth`` and computes ``(int)(v * 0.1) >> 7``
    (``FUN_00786340``). Some stock TRNs carry more than one ``[Size]`` block;
    the first one is authoritative -- ``lcbench.trn`` declares 5120 then 3840
    and the shipped ``lcbench.hg2`` header says 4x4, matching the first.
    """
    width = depth = None
    in_size = False
    seen_size = False
    with open(path, "r", errors="replace") as stream:
        for line in stream:
            section = _SECTION_RE.match(line)
            if section is not None:
                if in_size:
                    break                      # first [Size] block ends here
                in_size = section.group("name").strip().lower() == "size"
                seen_size = seen_size or in_size
                continue
            if not in_size:
                continue
            key, sep, value = line.partition("=")
            if not sep:
                continue
            key = key.strip().lower()
            try:
                number = float(value.strip())
            except ValueError:
                continue
            if key == "width":
                width = number
            elif key == "depth":
                depth = number
    if not seen_size or width is None or depth is None:
        return None
    zones_x = int(width * 0.1) >> LEGACY_ZONE_BITS
    zones_z = int(depth * 0.1) >> LEGACY_ZONE_BITS
    if zones_x <= 0 or zones_z <= 0:
        return None
    # The engine refuses the terrain unless the declared size is an exact
    # number of zones, so a TRN that fails this is not usable as a dimension
    # source either.
    if width != zones_x * ZONE_WORLD_SIZE or depth != zones_z * ZONE_WORLD_SIZE:
        return None
    return zones_x, zones_z


def find_trn(hgt_path: os.PathLike | str) -> Optional[str]:
    """Locate the ``.trn`` beside an HGT, tolerating case on case-sensitive FS."""
    stem, _ = os.path.splitext(str(hgt_path))
    for suffix in (".trn", ".TRN", ".Trn"):
        candidate = stem + suffix
        if os.path.exists(candidate):
            return candidate
    return None


def zone_count_candidates(size_bytes: int) -> List[Tuple[int, int]]:
    """Plausible ``(zones_x, zones_z)`` factorisations for a raw HGT size.

    HGT carries no dimensions, so a 12-zone file is genuinely ambiguous between
    4x3, 3x4, 6x2, ... Square factorisations are listed first because they are
    by far the most common in the stock and community corpus.
    """
    if size_bytes <= 0 or size_bytes % LEGACY_ZONE_BYTES:
        return []
    zones = size_bytes // LEGACY_ZONE_BYTES
    pairs = [(x, zones // x) for x in range(1, zones + 1) if zones % x == 0]
    return sorted(pairs, key=lambda p: (abs(p[0] - p[1]), p[0]))


# ---------------------------------------------------------------------------
# HGT
# ---------------------------------------------------------------------------

@dataclass
class HGTMap:
    """A parsed legacy heightmap.

    ``raw`` keeps the untouched 16-bit samples (including the flag nibble the
    engine masks off) so that a read/write round trip is byte-exact; ``heights``
    exposes the 12-bit height the engine actually uses.
    """

    raw: np.ndarray            # (zones_z * 128, zones_x * 128) uint16
    zones_x: int
    zones_z: int

    @property
    def heights(self) -> np.ndarray:
        return (self.raw & LEGACY_HEIGHT_MASK).astype(np.int32)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.zones_z * LEGACY_ZONE_SIZE, self.zones_x * LEGACY_ZONE_SIZE

    @property
    def flags(self) -> np.ndarray:
        """The high nibble the engine discards. Non-zero in most stock files."""
        return (self.raw >> 12).astype(np.uint8)

    @classmethod
    def read(cls, path: os.PathLike | str, zones_x: int, zones_z: int) -> "HGTMap":
        blob = np.fromfile(str(path), dtype="<u2")
        expected = zones_x * zones_z * LEGACY_ZONE_SIZE * LEGACY_ZONE_SIZE
        if blob.size != expected:
            raise HGTFormatError(
                f"{os.path.basename(str(path))}: {blob.size * 2} bytes does not match "
                f"{zones_x}x{zones_z} zones ({expected * 2} bytes)"
            )
        tiled = blob.reshape(zones_z, zones_x, LEGACY_ZONE_SIZE, LEGACY_ZONE_SIZE)
        full = tiled.transpose(0, 2, 1, 3).reshape(
            zones_z * LEGACY_ZONE_SIZE, zones_x * LEGACY_ZONE_SIZE
        )
        return cls(np.ascontiguousarray(full), zones_x, zones_z)

    @classmethod
    def read_auto(cls, path: os.PathLike | str) -> "HGTMap":
        """Read using the companion TRN's dimensions, else the squarest guess."""
        size = os.path.getsize(str(path))
        if size == 0:
            raise HGTFormatError(f"{os.path.basename(str(path))}: empty file")
        if size % LEGACY_ZONE_BYTES:
            raise HGTFormatError(
                f"{os.path.basename(str(path))}: {size} bytes is not a whole number of "
                f"{LEGACY_ZONE_BYTES}-byte zones"
            )
        trn = find_trn(path)
        if trn is not None:
            counts = read_trn_zone_counts(trn)
            if counts is not None and counts[0] * counts[1] * LEGACY_ZONE_BYTES == size:
                return cls.read(path, *counts)
        candidates = zone_count_candidates(size)
        if not candidates:
            raise HGTFormatError(f"{os.path.basename(str(path))}: cannot infer dimensions")
        return cls.read(path, *candidates[0])

    def write(self, path: os.PathLike | str) -> None:
        zs = LEGACY_ZONE_SIZE
        tiled = self.raw.reshape(self.zones_z, zs, self.zones_x, zs)
        tiled.transpose(0, 2, 1, 3).astype("<u2", copy=False).tofile(str(path))

    # -- conversion --------------------------------------------------------

    def to_hg2(self, *, smoothing: bool = False, rounding: str = "engine",
               map_version: int = HG2_MAP_VERSION) -> HG2Map:
        """Cook this legacy terrain into a Redux HG2 map.

        ``smoothing=False`` (the default) is the whole point of this module: it
        performs Redux's own 2x piecewise-planar upsample and then *skips*
        ``FUN_00785c80``, exactly as the shipped ``-nohgtsmoothing`` switch
        does. Legacy vertices survive bit-exact; only the samples introduced
        between them are new.

        ``smoothing=True`` reproduces Redux's default runtime cook instead, for
        parity checking.
        """
        grid = upsample(self.heights, self.zones_x, self.zones_z, rounding=rounding)
        if smoothing:
            grid = box_blur(grid)
        return HG2Map(
            grid,
            self.zones_x,
            self.zones_z,
            zone_bits=DEFAULT_ZONE_BITS,
            structure_version=HG2_STRUCTURE_VERSION,
            map_version=map_version,
        )


# ---------------------------------------------------------------------------
# the cook
# ---------------------------------------------------------------------------

def upsample(heights: np.ndarray, zones_x: int, zones_z: int, *,
             rounding: str = "engine") -> np.ndarray:
    """Redux's 2x legacy upsample (``FUN_00786200`` + ``FUN_00785fe0``).

    Output sample ``(X, Z)`` is the legacy surface evaluated at ``(X/2, Z/2)``,
    so even/even outputs land exactly on legacy vertices and reproduce them
    without modification.

    ``rounding="engine"`` reproduces the shipped code bit-for-bit: the whole
    expression is single-precision SSE and the final conversion is
    ``cvttss2si``, i.e. truncation toward zero. ``rounding="half-up"`` instead
    evaluates the plane in exact integer arithmetic and rounds halves up; some
    of Rebellion's own shipped HG2 files were produced that way. The two differ
    by at most one raw unit (0.1 world units) and only at half-sample
    positions.
    """
    if rounding not in ROUNDING_MODES:
        raise ValueError(f"rounding must be one of {ROUNDING_MODES}, got {rounding!r}")

    source = heights.astype(np.int32, copy=False)
    src_w = zones_x * LEGACY_ZONE_SIZE
    src_d = zones_z * LEGACY_ZONE_SIZE
    if source.shape != (src_d, src_w):
        raise ValueError(f"height grid {source.shape} does not match {zones_x}x{zones_z} zones")
    out_w = zones_x * (1 << DEFAULT_ZONE_BITS)
    out_d = zones_z * (1 << DEFAULT_ZONE_BITS)

    out_x = np.arange(out_w, dtype=np.int64)
    out_z = np.arange(out_d, dtype=np.int64)
    x0 = out_x >> 1
    z0 = out_z >> 1
    # The engine clamps the forward neighbour on the last row/column rather
    # than reading out of bounds (`iVar1 < zones * 0x80 - 1` in FUN_00785fe0).
    step_x = (x0 < src_w - 1).astype(np.int64)
    step_z = (z0 < src_d - 1).astype(np.int64)
    # Fractional offsets are exactly 0.0 or 0.5 because the source coordinate
    # is `out * 0.5`; keep them doubled so the integer path stays exact.
    frac_x2 = (out_x & 1)
    frac_z2 = (out_z & 1)

    X0 = x0[None, :]
    Z0 = z0[:, None]
    DX = step_x[None, :]
    DZ = step_z[:, None]
    FX2 = frac_x2[None, :]
    FZ2 = frac_z2[:, None]

    h00 = source[Z0, X0]
    h11 = source[Z0 + DZ, X0 + DX]
    h10 = source[Z0, X0 + DX]
    h01 = source[Z0 + DZ, X0]

    # `comiss fz, fx ; jbe` -> the (0,0)-(1,0)-(1,1) triangle when fz <= fx.
    lower = FZ2 <= FX2
    grad_x = np.where(lower, h00 - h10, h01 - h11)
    grad_z = np.where(lower, h10 - h11, h00 - h01)

    if rounding == "engine":
        gx = (grad_x.astype(_F32) * _KK).astype(_F32)
        gz = (grad_z.astype(_F32) * _KK).astype(_F32)
        fx = (FX2.astype(np.float64) * 0.5).astype(_F32)
        fz = (FZ2.astype(np.float64) * 0.5).astype(_F32)
        acc = ((fx * gx).astype(_F32) + (fz * gz).astype(_F32)).astype(_F32)
        acc = (acc * _TEN).astype(_F32)
        acc = (acc / _K).astype(_F32)
        delta = np.trunc(acc.astype(np.float64)).astype(np.int64)
    else:
        # 2 * (fx*grad_x + fz*grad_z), exact; round halves up.
        doubled = FX2 * grad_x.astype(np.int64) + FZ2 * grad_z.astype(np.int64)
        delta = (doubled + 1) >> 1

    result = h00.astype(np.int64) - delta
    # The engine stores the result through a 16-bit word with no clamp
    # (`mov ax, word ptr [ebp-0x44]`); interpolation between 12-bit samples
    # cannot leave that range, but mirror the truncation rather than assume it.
    return (result & 0xFFFF).astype(np.uint16)


def box_blur(grid: np.ndarray) -> np.ndarray:
    """Redux's post-cook 3x3 box blur (``FUN_00785c80``).

    Averages each sample with its in-bounds 8-neighbourhood out of place, and
    rounds with ``(2 * sum + n) / (2 * n)`` -- i.e. half away from zero, on
    values that are always positive. This runs unless ``-nohgtsmoothing`` was
    passed, and it is what erases authored stair-steps.
    """
    source = grid.astype(np.int64, copy=False)
    depth, width = source.shape
    total = np.zeros((depth, width), dtype=np.int64)
    count = np.zeros((depth, width), dtype=np.int64)
    for dz in (-1, 0, 1):
        for dx in (-1, 0, 1):
            z_lo, z_hi = max(0, -dz), min(depth, depth - dz)
            x_lo, x_hi = max(0, -dx), min(width, width - dx)
            total[z_lo:z_hi, x_lo:x_hi] += source[z_lo + dz:z_hi + dz, x_lo + dx:x_hi + dx]
            count[z_lo:z_hi, x_lo:x_hi] += 1
    return (((2 * total + count) // (2 * count)) & 0xFFFF).astype(np.uint16)


# ---------------------------------------------------------------------------
# HG2 header inspection (read-only; avoids hg2.HG2Map's 13-bit storage mask)
# ---------------------------------------------------------------------------

def read_hg2_header(path: os.PathLike | str) -> dict:
    """Header fields plus the validation Redux applies in ``FUN_00786340``."""
    size = os.path.getsize(str(path))
    with open(path, "rb") as stream:
        head = stream.read(12)
    if len(head) < 12:
        raise ValueError("HG2 header is truncated")
    structure_version, zone_bits, zones_x, zones_z, map_version = struct.unpack("<HHHHI", head)
    zone_size = 1 << zone_bits
    expected = 12 + zones_x * zones_z * zone_size * zone_size * 2
    return {
        "structure_version": structure_version,
        "zone_bits": zone_bits,
        "zones_x": zones_x,
        "zones_z": zones_z,
        "map_version": map_version,
        "size": size,
        "expected_size": expected,
        # FUN_00786340 checks size >= 13, structure_version == 1, zone_bits == 8,
        # the two zone counts against the TRN, and map_version >= 10. It does
        # *not* check the payload length, so a short file is accepted and then
        # read out of bounds.
        "header_valid": (
            size >= 13
            and structure_version == HG2_STRUCTURE_VERSION
            and zone_bits == DEFAULT_ZONE_BITS
            and map_version >= HG2_MAP_VERSION
        ),
        "size_consistent": size == expected,
    }
