from __future__ import annotations

import math
from typing import Callable, Dict, Tuple

import numpy as np

from .builder import TerrainBuilder
from .hg2 import HG2Map
from .noise import meander_path, organic_loop, vary_widths
from .settings import GeneratorSettings


def edge_point(builder: TerrainBuilder, side: str, t: float) -> Tuple[float, float]:
    if side == "left":
        return 0.0, t * (builder.h - 1)
    if side == "right":
        return builder.w - 1.0, t * (builder.h - 1)
    if side == "top":
        return t * (builder.w - 1), 0.0
    return t * (builder.w - 1), builder.h - 1.0


def terraced_labyrinth(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(0)
    m = min(b.h, b.w)
    plateau = np.clip(s.plateau_bias, 0, 1)
    levels = [0.0, 230.0, 500.0, 780.0] if plateau > 0.55 else [0.0, 260.0, 590.0]
    b.add_terraced_blobs(
        levels,
        m * (0.060 + 0.035 * (1 - s.feature_density)),
        1.5 + 1.8 * s.naturalization,
        threshold_bias=(plateau - 0.5) * -0.35,
        warp_px=m * 0.020 * s.naturalization,
    )
    b.add_detail(5.0 * s.detail, 22.0)
    return b.finalize(center_height=850.0, preserve_flats=True)


def cratered_divide(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(1150)
    m = min(b.h, b.w)
    b.add_fbm(420, m * 0.38, ridged=False, octaves=4)
    path = meander_path(edge_point(b, "left", 0.18), edge_point(b, "right", 0.80), 12, m * 0.11 * s.naturalization, b.rng)
    b.carve_path(path, depth=900, half_width=m * 0.035, bank=m * 0.10, rim=180)
    b.add_random_craters(int(8 + 22 * s.feature_density), (m * 0.025, m * 0.075), 4.0, 1.0)
    b.add_detail(65 * s.detail, m * 0.035)
    return b.finalize(center_height=1900.0, preserve_flats=False)


def ravine_network(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(2200)
    m = min(b.h, b.w)
    b.add_fbm(210, m * 0.22, ridged=True, octaves=4)
    trunk = meander_path(edge_point(b, "top", 0.42), edge_point(b, "bottom", 0.58), 14, m * 0.10 * s.naturalization, b.rng)
    b.carve_path(trunk, depth=1300, half_width=m * 0.028, bank=m * 0.065, rim=110)
    for i in range(2 + int(3 * s.feature_density)):
        side = "left" if i % 2 == 0 else "right"
        start = edge_point(b, side, float(b.rng.uniform(0.15, 0.85)))
        target = trunk[int(b.rng.integers(3, len(trunk) - 3))]
        branch = meander_path(start, target, 9, m * 0.07 * s.naturalization, b.rng)
        b.carve_path(branch, depth=950, half_width=m * 0.020, bank=m * 0.050, rim=80)
    b.add_detail(45 * s.detail, m * 0.025)
    return b.finalize(center_height=2600.0, preserve_flats=True)


def mountain_basin(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(700)
    m = min(b.h, b.w)
    b.add_fbm(850, m * 0.22, ridged=True, octaves=5)
    yy, xx = np.mgrid[0:b.h, 0:b.w].astype(np.float32)
    cx, cy = b.w * 0.50, b.h * 0.52
    radius = np.sqrt(((xx - cx) / (m * 0.44)) ** 2 + ((yy - cy) / (m * 0.40)) ** 2)
    b.a += np.exp(-0.5 * ((radius - 0.77) / 0.18) ** 2) * 720
    b.a -= np.exp(-0.5 * (radius / 0.46) ** 2) * 520
    b.add_random_craters(int(3 + 7 * s.feature_density), (m * 0.018, m * 0.05), 2.6, 0.7)
    b.add_detail(80 * s.detail, m * 0.025)
    if s.synthetic_pads == 0:
        b.flatten_pad(cx + m * 0.13, cy + m * 0.13, m * 0.055, m * 0.045, feather=m * 0.012, rectangular=True)
    return b.finalize(center_height=1700.0, preserve_flats=False)


def radial_badlands(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(1400)
    m = min(b.h, b.w)
    b.add_fbm(330, m * 0.28, ridged=True, octaves=4)
    cx, cy = b.w * 0.5, b.h * 0.5
    arms = 5 + int(5 * s.feature_density)
    for i in range(arms):
        angle = i * math.tau / arms + float(b.rng.uniform(-0.18, 0.18))
        end = (cx + math.cos(angle) * m * 0.68, cy + math.sin(angle) * m * 0.68)
        path = meander_path((cx, cy), end, 9, m * 0.045 * s.naturalization, b.rng)
        if i % 2:
            b.carve_path(path, depth=420, half_width=m * 0.012, bank=m * 0.045, rim=80)
        else:
            b.add_ridge_path(path, height=430, half_width=m * 0.012, falloff=m * 0.050)
    b.crater(cx, cy, m * 0.075, 380, 150)
    b.add_random_craters(int(4 + 10 * s.feature_density), (m * 0.015, m * 0.038), 2.2, 0.65)
    b.add_detail(65 * s.detail, m * 0.028)
    return b.finalize(center_height=1900.0, preserve_flats=False)


def ridged_wastes(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(700)
    m = min(b.h, b.w)
    b.add_fbm(520, m * 0.075, ridged=True, octaves=5)
    b.add_fbm(260, m * 0.30, ridged=False, octaves=3)
    for i in range(2 + int(3 * s.feature_density)):
        if i % 2:
            start, end = edge_point(b, "left", float(b.rng.uniform(.15, .85))), edge_point(b, "right", float(b.rng.uniform(.15, .85)))
        else:
            start, end = edge_point(b, "top", float(b.rng.uniform(.15, .85))), edge_point(b, "bottom", float(b.rng.uniform(.15, .85)))
        path = meander_path(start, end, 13, m * 0.09 * s.naturalization, b.rng)
        b.carve_path(path, depth=330, half_width=m * 0.010, bank=m * 0.025, rim=40)
    b.add_detail(35 * s.detail, m * 0.018)
    return b.finalize(center_height=1150.0, preserve_flats=False)


def serpentine_canyon(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(2350)
    m = min(b.h, b.w)
    path = meander_path(edge_point(b, "top", 0.18), edge_point(b, "bottom", 0.83), 18, m * (0.15 + 0.04 * s.naturalization), b.rng)
    b.carve_path(path, depth=1900, half_width=m * 0.026, bank=m * 0.055, rim=85)
    b.add_random_craters(int(6 + 14 * s.feature_density), (m * 0.012, m * 0.035), 1.8, 0.5)
    b.add_terraced_blobs([0, 90, 160], m * 0.11, 4.5, threshold_bias=0.25, warp_px=m * 0.015 * s.naturalization)
    b.add_detail(16 * s.detail, m * 0.024)
    return b.finalize(center_height=2700.0, preserve_flats=True)


def natural_badlands(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(850)
    m = min(b.h, b.w)
    b.add_fbm(620, m * 0.16, ridged=True, octaves=5)
    b.add_fbm(300, m * 0.42, ridged=False, octaves=4)
    for _ in range(int(2 + 6 * s.feature_density)):
        start = edge_point(b, str(b.rng.choice(["left", "top"])), float(b.rng.uniform(.1, .9)))
        end = edge_point(b, str(b.rng.choice(["right", "bottom"])), float(b.rng.uniform(.1, .9)))
        path = meander_path(start, end, 11, m * 0.10 * s.naturalization, b.rng)
        b.carve_path(path, depth=250, half_width=m * 0.012, bank=m * 0.035, rim=30)
    b.add_detail(70 * s.detail, m * 0.020)
    return b.finalize(center_height=1500.0, preserve_flats=False)


def campaign_canyon_network(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(2450)
    m = min(b.h, b.w)
    b.add_fbm(135, m * 0.34, ridged=False, octaves=3)
    trunk = meander_path(
        edge_point(b, "left", 0.28),
        edge_point(b, "right", 0.66),
        16,
        m * 0.16 * (0.35 + 0.65 * s.naturalization),
        b.rng,
    )
    widths = vary_widths(len(trunk), m * 0.032, 0.45 * s.naturalization, b.rng)
    b.carve_variable_corridor_level(
        trunk,
        760,
        widths,
        bank=m * 0.055,
        rim_height=120,
        edge_irregularity=m * 0.010 * s.naturalization,
    )
    for i in range(1 + int(round(2 * s.feature_density))):
        side = "top" if i % 2 == 0 else "bottom"
        start = edge_point(b, side, float(b.rng.uniform(0.18, 0.82)))
        target = trunk[int(b.rng.integers(len(trunk) // 4, len(trunk) * 3 // 4))]
        branch = meander_path(start, target, 10, m * 0.075 * s.naturalization, b.rng)
        branch_widths = vary_widths(len(branch), m * 0.021, 0.35 * s.naturalization, b.rng)
        b.carve_variable_corridor_level(
            branch,
            760,
            branch_widths,
            bank=m * 0.042,
            rim_height=80,
            edge_irregularity=m * 0.007 * s.naturalization,
        )
    if s.plateau_bias > 0.30:
        b.flatten_pad(b.w * .20, b.h * .78, m * .050, m * .040, target=2450, feather=m * .012, rectangular=True)
    b.add_detail(20 * s.detail, m * 0.030)
    return b.finalize(center_height=2250.0, preserve_flats=True)


def compartmented_plateau(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(2280)
    m = min(b.h, b.w)
    b.add_fbm(105, m * 0.36, ridged=False, octaves=3)
    anchors = [
        (0.30, 0.34, 0.18, 0.13, -0.35),
        (0.68, 0.42, 0.20, 0.15, 0.28),
        (0.48, 0.73, 0.22, 0.14, -0.08),
        (0.78, 0.76, 0.14, 0.11, 0.55),
    ]
    count = 2 + int(round(2 * s.feature_density))
    for ax, ay, rx, ry, rotation in anchors[:count]:
        cx = b.w * ax + float(b.rng.uniform(-m * .025, m * .025)) * s.naturalization
        cy = b.h * ay + float(b.rng.uniform(-m * .025, m * .025)) * s.naturalization
        loop = organic_loop(
            (cx, cy),
            m * rx,
            m * ry,
            14,
            m * 0.065 * s.naturalization,
            b.rng,
            rotation=rotation + float(b.rng.uniform(-.18, .18)),
        )
        widths = vary_widths(len(loop), m * .018, .38 * s.naturalization, b.rng, cycles=7)
        b.carve_variable_corridor_level(loop, 700, widths, bank=m * .036, rim_height=85, edge_irregularity=m * .006 * s.naturalization)
    connector = meander_path(edge_point(b, "left", .58), edge_point(b, "right", .46), 13, m * .08 * s.naturalization, b.rng)
    connector_widths = vary_widths(len(connector), m * .015, .28 * s.naturalization, b.rng)
    b.carve_variable_corridor_level(connector, 700, connector_widths, bank=m * .030, rim_height=55, edge_irregularity=m * .005 * s.naturalization)
    b.add_detail(18 * s.detail, m * 0.028)
    return b.finalize(center_height=2200.0, preserve_flats=True)


def sparse_mission_field(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(460)
    m = min(b.h, b.w)
    b.stamp_blob_shelf(
        target=0,
        area_fraction=0.28 + 0.20 * (1.0 - s.plateau_bias),
        feature_px=m * 0.38,
        feather=m * 0.018,
        warp_px=m * 0.025 * s.naturalization,
        protect_core=True,
    )
    for i in range(3 + int(6 * s.feature_density)):
        radius = float(b.rng.uniform(m * .025, m * .060))
        cx = float(b.rng.uniform(m * .12, b.w - m * .12))
        cy = float(b.rng.uniform(m * .12, b.h - m * .12))
        if i % 3 == 0:
            b.crater(cx, cy, radius, depth=radius * 2.1, rim_height=radius * .9)
        else:
            yy, xx = np.mgrid[0:b.h, 0:b.w].astype(np.float32)
            rr = np.hypot(xx - cx, yy - cy) / max(radius, 1)
            b.a += (np.exp(-0.5 * (rr / 0.62) ** 2) * (280 + 220 * s.relief)).astype(np.float32)
    b.flatten_pad(b.w * .25, b.h * .72, m * .050, m * .042, target=460, feather=m * .012, rectangular=True)
    b.flatten_pad(b.w * .73, b.h * .28, m * .050, m * .042, target=0, feather=m * .012, rectangular=True)
    b.add_detail(10 * s.detail, m * 0.030)
    return b.finalize(center_height=800.0, preserve_flats=True)


def walled_crater_basin(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(480)
    m = min(b.h, b.w)
    b.add_boundary_rim(
        height=520,
        inner_margin=m * 0.015,
        width=m * 0.055,
        irregularity=m * 0.030 * s.naturalization,
    )
    b.add_fbm(70, m * 0.28, ridged=False, octaves=3)
    b.add_random_craters(int(6 + 18 * s.feature_density), (m * .018, m * .070), 1.8, 0.75)
    b.flatten_pad(b.w * .50, b.h * .52, m * .13, m * .10, target=480, feather=m * .025, rectangular=False)
    b.add_detail(18 * s.detail, m * .025)
    return b.finalize(center_height=780.0, preserve_flats=True)


def escarpment_stronghold(s: GeneratorSettings) -> HG2Map:
    b = TerrainBuilder(s).set_level(860)
    m = min(b.h, b.w)
    b.add_fbm(150, m * .24, ridged=True, octaves=4)
    shelf_area = 0.30 + 0.22 * float(np.clip(s.plateau_bias, 0, 1))
    b.stamp_blob_shelf(
        2660,
        shelf_area,
        feature_px=m * .33,
        feather=m * .042,
        warp_px=m * .040 * s.naturalization,
        protect_core=True,
    )
    for i in range(1 + int(round(2 * s.feature_density))):
        side = ["left", "bottom", "right"][i % 3]
        start = edge_point(b, side, float(b.rng.uniform(.18, .82)))
        end = (b.w * float(b.rng.uniform(.38, .62)), b.h * float(b.rng.uniform(.38, .62)))
        path = meander_path(start, end, 10, m * .055 * s.naturalization, b.rng)
        widths = vary_widths(len(path), m * .017, .30 * s.naturalization, b.rng)
        b.carve_variable_corridor_level(path, 720, widths, bank=m * .040, rim_height=55, edge_irregularity=m * .006 * s.naturalization)
    b.flatten_pad(b.w * .54, b.h * .47, m * .060, m * .048, target=2660, feather=m * .014, rectangular=True)
    b.add_detail(25 * s.detail, m * .026)
    return b.finalize(center_height=1750.0, preserve_flats=True)


RECIPES: Dict[str, Callable[[GeneratorSettings], HG2Map]] = {
    "Terraced Labyrinth": terraced_labyrinth,
    "Cratered Divide": cratered_divide,
    "Ravine Network": ravine_network,
    "Mountain Basin": mountain_basin,
    "Radial Badlands": radial_badlands,
    "Ridged Wastes": ridged_wastes,
    "Serpentine Canyon": serpentine_canyon,
    "Natural Badlands": natural_badlands,
    "Campaign Canyon Network": campaign_canyon_network,
    "Compartmented Plateau": compartmented_plateau,
    "Sparse Mission Field": sparse_mission_field,
    "Walled Crater Basin": walled_crater_basin,
    "Escarpment Stronghold": escarpment_stronghold,
}


def generate(style: str, settings: GeneratorSettings) -> HG2Map:
    try:
        recipe = RECIPES[style]
    except KeyError as exc:
        raise ValueError(f"Unknown terrain style {style!r}. Choices: {', '.join(RECIPES)}") from exc
    return recipe(settings)
