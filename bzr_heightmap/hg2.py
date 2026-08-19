from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image

HG2_STORAGE_MASK = 0x1FFF
HG2_SAFE_MAX_HEIGHT = 0x0FFF
HG2_HEIGHT_MASK = HG2_STORAGE_MASK
HG2_MAX_HEIGHT = HG2_SAFE_MAX_HEIGHT
HG2_STRUCTURE_VERSION = 1
HG2_MAP_VERSION = 10
DEFAULT_ZONE_BITS = 8
BZ_ZONE_WORLD_SIZE = 1280.0


@dataclass
class HG2Map:
    heights: np.ndarray
    zones_x: int
    zones_z: int
    zone_bits: int = DEFAULT_ZONE_BITS
    structure_version: int = HG2_STRUCTURE_VERSION
    map_version: int = HG2_MAP_VERSION

    @property
    def zone_size(self) -> int:
        return 1 << self.zone_bits

    @property
    def shape(self) -> Tuple[int, int]:
        return self.zones_z * self.zone_size, self.zones_x * self.zone_size

    @property
    def world_size(self) -> Tuple[float, float]:
        return self.zones_x * BZ_ZONE_WORLD_SIZE, self.zones_z * BZ_ZONE_WORLD_SIZE

    def validate(self) -> None:
        if self.heights.shape != self.shape:
            raise ValueError(f"Height shape {self.heights.shape} does not match HG2 dimensions {self.shape}")
        if np.min(self.heights) < 0 or np.max(self.heights) > HG2_MAX_HEIGHT:
            raise ValueError(f"Generated HG2 height samples must be 0..{HG2_MAX_HEIGHT}")

    @classmethod
    def read(cls, path: os.PathLike | str) -> "HG2Map":
        with open(path, "rb") as stream:
            header = stream.read(12)
            if len(header) != 12:
                raise ValueError("HG2 header is truncated")
            structure_version, zone_bits, zones_x, zones_z, map_version = struct.unpack("<HHHHI", header)
            zone_size = 1 << zone_bits
            count = zones_x * zones_z * zone_size * zone_size
            raw = np.frombuffer(stream.read(), dtype="<u2")

        if raw.size != count:
            raise ValueError(f"HG2 sample count mismatch: expected {count}, found {raw.size}")

        raw = raw & HG2_STORAGE_MASK
        full = np.empty((zones_z * zone_size, zones_x * zone_size), dtype=np.uint16)
        cursor = 0
        for zone_z in range(zones_z):
            for zone_x in range(zones_x):
                zone = raw[cursor : cursor + zone_size * zone_size].reshape(zone_size, zone_size)
                z0, x0 = zone_z * zone_size, zone_x * zone_size
                full[z0 : z0 + zone_size, x0 : x0 + zone_size] = zone
                cursor += zone_size * zone_size
        return cls(full, zones_x, zones_z, zone_bits, structure_version, map_version)

    def write(self, path: os.PathLike | str) -> None:
        self.validate()
        heights = np.clip(np.rint(self.heights), 0, HG2_MAX_HEIGHT).astype("<u2") & HG2_STORAGE_MASK
        with open(path, "wb") as stream:
            stream.write(
                struct.pack(
                    "<HHHHI",
                    self.structure_version,
                    self.zone_bits,
                    self.zones_x,
                    self.zones_z,
                    self.map_version,
                )
            )
            zone_size = self.zone_size
            for zone_z in range(self.zones_z):
                for zone_x in range(self.zones_x):
                    z0, x0 = zone_z * zone_size, zone_x * zone_size
                    zone = heights[z0 : z0 + zone_size, x0 : x0 + zone_size]
                    stream.write(zone.astype("<u2", copy=False).tobytes(order="C"))

    def write_png16(self, path: os.PathLike | str) -> None:
        scaled = (np.clip(self.heights, 0, HG2_MAX_HEIGHT).astype(np.uint32) * 16).astype(np.uint16)
        Image.fromarray(scaled, mode="I;16").save(path)
