from .analysis import describe_heightmap, make_preview, make_shaded_image, terrain_metrics, traversability_metrics
from .approved_planetary import APPROVED_PLANETARY_RECIPES
from .contrast import apply_vertical_scale
from .lgt import compute_lgt_lightmap, compute_lgt_for_hg2, lgt_to_brightness, read_lgt, write_lgt
from .preview import make_hg2_height_image, make_lgt_preview_image, make_shaded_preview_fullres
from .hg2 import (
    BZ_ZONE_WORLD_SIZE,
    DEFAULT_ZONE_BITS,
    HG2_HEIGHT_MASK,
    HG2_MAP_VERSION,
    HG2_MAX_HEIGHT,
    HG2_SAFE_MAX_HEIGHT,
    HG2_STORAGE_MASK,
    HG2_STORAGE_MAX_HEIGHT,
    HG2_STRUCTURE_VERSION,
    HG2Map,
)
from .hgt import (
    HGTFormatError,
    HGTMap,
    LEGACY_HEIGHT_MASK,
    LEGACY_ZONE_BYTES,
    LEGACY_ZONE_SIZE,
    ROUNDING_MODES,
    box_blur,
    find_trn,
    read_hg2_header,
    read_trn_zone_counts,
    upsample,
    zone_count_candidates,
)
from .planetary import PLANETARY_RECIPES
from .planetary_detail import PLANETARY_DETAIL_STYLES, enhance_planetary_terrain
from .planetary_surface import PLANETARY_SURFACE_STYLES, enhance_planetary_surface
from .recipes import RECIPES as CORE_RECIPES, generate as generate_core
from .settings import GeneratorSettings, RANDOM_SEED_MAX, random_seed, resolve_seed
from .stock_detail import STOCK_DETAIL_STYLES, stock_connectivity_metrics
from .stock_detail_v5 import enhance_stock_terrain
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
                if style in STOCK_DETAIL_STYLES:
                    terrain = enhance_stock_terrain(terrain, settings, style)
        if style in PLANETARY_DETAIL_STYLES:
            terrain = enhance_planetary_terrain(terrain, settings, style)
        if style in PLANETARY_SURFACE_STYLES:
            terrain = enhance_planetary_surface(terrain, settings, style)
    return apply_vertical_scale(terrain, settings.vertical_scale)


__all__ = [
    "APPROVED_PLANETARY_RECIPES",
    "BZ_ZONE_WORLD_SIZE",
    "DEFAULT_ZONE_BITS",
    "GeneratorSettings",
    "HGTFormatError",
    "HGTMap",
    "LEGACY_HEIGHT_MASK",
    "LEGACY_ZONE_BYTES",
    "LEGACY_ZONE_SIZE",
    "ROUNDING_MODES",
    "RANDOM_SEED_MAX",
    "HG2_HEIGHT_MASK",
    "HG2_MAP_VERSION",
    "HG2_MAX_HEIGHT",
    "HG2_SAFE_MAX_HEIGHT",
    "HG2_STORAGE_MASK",
    "HG2_STORAGE_MAX_HEIGHT",
    "HG2_STRUCTURE_VERSION",
    "HG2Map",
    "PLANETARY_DETAIL_STYLES",
    "PLANETARY_SURFACE_STYLES",
    "PLANETARY_RECIPES",
    "RECIPES",
    "STOCK_DETAIL_STYLES",
    "URBAN_RECIPES",
    "apply_vertical_scale",
    "box_blur",
    "compute_lgt_for_hg2",
    "compute_lgt_lightmap",
    "describe_heightmap",
    "enhance_planetary_surface",
    "enhance_planetary_terrain",
    "enhance_stock_terrain",
    "find_trn",
    "generate",
    "make_hg2_height_image",
    "make_lgt_preview_image",
    "make_preview",
    "make_shaded_image",
    "make_shaded_preview_fullres",
    "lgt_to_brightness",
    "random_seed",
    "read_hg2_header",
    "read_trn_zone_counts",
    "read_lgt",
    "resolve_seed",
    "stock_connectivity_metrics",
    "terrain_metrics",
    "upsample",
    "traversability_metrics",
    "write_lgt",
    "zone_count_candidates",
]
