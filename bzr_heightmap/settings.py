from __future__ import annotations

from dataclasses import dataclass
import secrets
from typing import Tuple

from .hg2 import DEFAULT_ZONE_BITS

RANDOM_SEED_MAX = 2_147_483_647


def random_seed() -> int:
    """Return a fresh positive seed suitable for reproducible terrain generation."""
    return secrets.randbelow(RANDOM_SEED_MAX) + 1


def resolve_seed(value: int | str | None) -> int:
    """Resolve an integer seed or the user-facing ``random`` sentinel."""
    if value is None:
        return random_seed()
    if isinstance(value, int):
        return int(value)
    text = str(value).strip()
    if not text or text.lower() in {"random", "rand", "auto", "new"}:
        return random_seed()
    try:
        return int(text, 0)
    except ValueError as exc:
        raise ValueError("seed must be an integer or 'random'") from exc


@dataclass(frozen=True)
class GeneratorSettings:
    zones_x: int = 3
    zones_z: int = 3
    seed: int = 1
    relief: float = 1.0
    vertical_scale: float = 1.0
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
