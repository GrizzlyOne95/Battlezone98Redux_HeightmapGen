from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .hg2 import DEFAULT_ZONE_BITS


@dataclass(frozen=True)
class GeneratorSettings:
    zones_x: int = 3
    zones_z: int = 3
    seed: int = 1
    relief: float = 1.0
    naturalization: float = 0.65
    detail: float = 0.55
    plateau_bias: float = 0.5
    feature_density: float = 0.5
    symmetry: str = "None"
    synthetic_pads: int = 0

    @property
    def shape(self) -> Tuple[int, int]:
        zone_size = 1 << DEFAULT_ZONE_BITS
        return self.zones_z * zone_size, self.zones_x * zone_size
