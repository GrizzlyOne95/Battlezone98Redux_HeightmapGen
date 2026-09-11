from __future__ import annotations

import unittest

import numpy as np
from scipy import ndimage

import bzr_heightmap as hmg
from bzr_heightmap.planetary import callisto_craterlands, lunar_catena
from bzr_heightmap.planetary_detail import enhance_planetary_terrain
from bzr_heightmap.recipes import walled_crater_basin


RAW = {
    "Lunar Catena": lunar_catena,
    "Callisto Craterlands": callisto_craterlands,
    "Walled Crater Basin": walled_crater_basin,
}


def residual_rms(a: np.ndarray) -> float:
    f = np.asarray(a, dtype=np.float32)
    blur = ndimage.gaussian_filter(f, 2.0, mode="reflect")
    return float(np.sqrt(np.mean((f - blur) ** 2)))


def lap_p95(a: np.ndarray) -> float:
    return float(np.percentile(np.abs(ndimage.laplace(np.asarray(a, dtype=np.float32))), 95))


def passable_fraction(a: np.ndarray) -> float:
    f = np.asarray(a, dtype=np.float32)
    gy, gx = np.gradient(f * 0.1, 5.0, 5.0)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    return float(np.mean(slope <= 15.0))


class PlanetaryDetailTests(unittest.TestCase):
    def test_craterland_pass_is_deterministic_and_safe(self) -> None:
        for style in RAW:
            settings = hmg.GeneratorSettings(zones_x=1, zones_z=1, seed=24680)
            a = hmg.generate(style, settings)
            b = hmg.generate(style, settings)
            self.assertTrue(np.array_equal(a.heights, b.heights), style)
            a.validate()
            self.assertGreaterEqual(int(np.min(a.heights)), 0, style)
            self.assertLessEqual(int(np.max(a.heights)), hmg.HG2_SAFE_MAX_HEIGHT, style)

    def test_craterland_pass_adds_native_scale_detail_without_consuming_the_map(self) -> None:
        for style, recipe in RAW.items():
            settings = hmg.GeneratorSettings(zones_x=1, zones_z=1, seed=13579)
            raw = recipe(settings)
            enhanced = enhance_planetary_terrain(raw, settings, style)

            # A one-zone fixture compresses the recipe's macro features, so
            # only require measurable residual growth here. The 3x3 audit is
            # the production-scale magnitude/quality gate.
            self.assertGreater(residual_rms(enhanced.heights), residual_rms(raw.heights), style)
            self.assertGreaterEqual(lap_p95(enhanced.heights), lap_p95(raw.heights) + 2.0, style)
            self.assertGreater(passable_fraction(enhanced.heights), 0.70, style)

            unchanged = float(np.mean(enhanced.heights == raw.heights))
            self.assertGreater(unchanged, 0.30, style)

    def test_metadata_and_dimensions_are_preserved(self) -> None:
        settings = hmg.GeneratorSettings(zones_x=2, zones_z=1, seed=4242)
        raw = lunar_catena(settings)
        enhanced = enhance_planetary_terrain(raw, settings, "Lunar Catena")
        self.assertEqual(enhanced.shape, raw.shape)
        self.assertEqual(enhanced.zones_x, raw.zones_x)
        self.assertEqual(enhanced.zones_z, raw.zones_z)
        self.assertEqual(enhanced.zone_bits, raw.zone_bits)
        self.assertEqual(enhanced.structure_version, raw.structure_version)
        self.assertEqual(enhanced.map_version, raw.map_version)


if __name__ == "__main__":
    unittest.main()
