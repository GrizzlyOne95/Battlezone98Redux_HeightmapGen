"""Testable GUI logic separated from Tk widgets.

Provides caching and debounce helpers so unit tests can verify regeneration
behaviour without requiring a live Tk main loop.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from bzr_heightmap.contrast import apply_vertical_scale
from bzr_heightmap.hg2 import HG2Map
from bzr_heightmap.settings import GeneratorSettings


@dataclass(frozen=True)
class RawCacheKey:
    style: str
    zones_x: int
    zones_z: int
    seed: int
    relief: float
    naturalization: float
    detail: float
    plateau_bias: float
    feature_density: float
    symmetry: str
    synthetic_pads: int

    @classmethod
    def from_settings(cls, style: str, s: GeneratorSettings) -> "RawCacheKey":
        return cls(
            style=style,
            zones_x=s.zones_x,
            zones_z=s.zones_z,
            seed=s.seed,
            relief=s.relief,
            naturalization=s.naturalization,
            detail=s.detail,
            plateau_bias=s.plateau_bias,
            feature_density=s.feature_density,
            symmetry=s.symmetry,
            synthetic_pads=s.synthetic_pads,
        )


def is_contrast_only_change(old: RawCacheKey | None, new: RawCacheKey) -> bool:
    """True if only vertical_scale changed (raw key identical)."""
    if old is None:
        return False
    return old == new


def apply_cached_contrast(raw: HG2Map, vertical_scale: float) -> HG2Map:
    """Apply vertical contrast to a cached raw map without mutating original."""
    # apply_vertical_scale returns new HG2Map and handles identity quickly
    return apply_vertical_scale(raw, vertical_scale)


def raw_generation_settings(settings: GeneratorSettings) -> GeneratorSettings:
    """Return the recipe settings that produce the cacheable raw terrain."""
    return replace(settings, vertical_scale=1.0)


DEBOUNCE_MS = 200


@dataclass
class LatestJobCoordinator:
    """Pure state machine for debounced, single-worker latest-job semantics."""

    latest_revision: int = 0
    active_revision: int | None = None
    pending: bool = False
    closing: bool = False

    def schedule(self) -> int:
        self.latest_revision += 1
        self.pending = True
        return self.latest_revision

    def start_latest(self) -> int | None:
        if self.closing or self.active_revision is not None or not self.pending:
            return None
        self.active_revision = self.latest_revision
        self.pending = False
        return self.active_revision

    def finish(self, revision: int) -> bool:
        if self.active_revision == revision:
            self.active_revision = None
        return not self.closing and (self.pending or revision != self.latest_revision)

    def accepts(self, revision: int) -> bool:
        return not self.closing and revision == self.latest_revision

    def close(self) -> None:
        self.closing = True
        self.pending = False


# Backward-compatible name for the original test helper.
DebounceState = LatestJobCoordinator


def raw_cache_key_for_test(style: str, zones_x: int, zones_z: int, seed: int) -> RawCacheKey:
    return RawCacheKey(style, zones_x, zones_z, seed, 1.0, 0.65, 0.55, 0.5, 0.5, "None", 0)
