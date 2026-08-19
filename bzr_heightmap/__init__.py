from .analysis import describe_heightmap, make_preview, terrain_metrics, traversability_metrics
from .approved_planetary import APPROVED_PLANETARY_RECIPES
from .contrast import apply_vertical_scale
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
from .urban import URBAN_RECIPES

RECIPES = {
    **CORE_RECIPES,
    **PLANETARY_RECIPES,
    **APPROVED_PLANETARY_RECIPES,
    **URBAN_RECIPES,
}


def generate(style: str, settings: GeneratorSettings) -> HG2Map:
    urban = URBAN_RECIPES.get(style)
    if urban is not None:
        terrain = urban(settings)
    else:
        approved_planetary = APPROVED_PLANETARY_RECIPES.get(style)
        if approved_planetary is not None:
            terrain = approved_planetary(settings)
        else:
            planetary = PLANETARY_RECIPES.get(style)
            if planetary is not None:
                terrain = planetary(settings)
            else:
                terrain = generate_core(style, settings)
    return apply_vertical_scale(terrain, settings.vertical_scale)


__all__ = [
    "APPROVED_PLANETARY_RECIPES",
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
    "URBAN_RECIPES",
    "apply_vertical_scale",
    "describe_heightmap",
    "generate",
    "make_preview",
    "random_seed",
    "resolve_seed",
    "terrain_metrics",
    "traversability_metrics",
]
