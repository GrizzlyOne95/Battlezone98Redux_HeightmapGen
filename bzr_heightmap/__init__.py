from .analysis import describe_heightmap, make_preview, terrain_metrics, traversability_metrics
from .hg2 import (
    BZ_ZONE_WORLD_SIZE,
    DEFAULT_ZONE_BITS,
    HG2_HEIGHT_MASK,
    HG2_MAP_VERSION,
    HG2_MAX_HEIGHT,
    HG2_SAFE_MAX_HEIGHT,
    HG2_STORAGE_MASK,
    HG2_STRUCTURE_VERSION,
    HG2Map,
)
from .planetary import PLANETARY_RECIPES
from .recipes import RECIPES as CORE_RECIPES, generate as generate_core
from .settings import GeneratorSettings, RANDOM_SEED_MAX, random_seed, resolve_seed

RECIPES = {**CORE_RECIPES, **PLANETARY_RECIPES}


def generate(style: str, settings: GeneratorSettings) -> HG2Map:
    planetary = PLANETARY_RECIPES.get(style)
    if planetary is not None:
        return planetary(settings)
    return generate_core(style, settings)


__all__ = [
    "BZ_ZONE_WORLD_SIZE",
    "DEFAULT_ZONE_BITS",
    "GeneratorSettings",
    "RANDOM_SEED_MAX",
    "HG2_HEIGHT_MASK",
    "HG2_MAP_VERSION",
    "HG2_MAX_HEIGHT",
    "HG2_SAFE_MAX_HEIGHT",
    "HG2_STORAGE_MASK",
    "HG2_STRUCTURE_VERSION",
    "HG2Map",
    "PLANETARY_RECIPES",
    "RECIPES",
    "describe_heightmap",
    "generate",
    "make_preview",
    "random_seed",
    "resolve_seed",
    "terrain_metrics",
    "traversability_metrics",
]
