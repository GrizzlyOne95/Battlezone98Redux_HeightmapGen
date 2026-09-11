from __future__ import annotations

import unittest

import numpy as np

import bzr_heightmap as hmg
from bzr_heightmap.planetary import (
    europa_fracture_plains,
    pluto_basin,
    titan_basin_network,
    venus_shield,
)
from bzr_heightmap.planetary_surface import enhance_planetary_surface


RAW = {
    "Pluto Basin": pluto_basin,
    "Venus Shield": venus_shield,
    "Titan Basin Network": titan_basin_network,
    "Europa Fracture Plains": europa_fracture_plains,
}


def passable_fraction(a: np.ndarray) -> float:
    f = np.asarray(a, dtype=np.float32)
    gy, gx = np.gradient(f * 0.1, 5.0, 5.0)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    return float(np.mean(slope <= 15.0))


class PlanetarySurfaceTests(unittest.TestCase):
    def test_surface_pass_is_deterministic_and_safe(self) -> None:
        for style in RAW:
            settings = hmg.GeneratorSettings(zones_x=1, zones_z=1, seed=24680)
            a = hmg.generate(style, settings)
            b = hmg.generate(style, settings)
            self.assertTrue(np.array_equal(a.heights, b.heights), style)
            a.validate()
            self.assertGreaterEqual(int(np.min(a.heights)), 0, style)
            self.assertLessEqual(int(np.max(a.heights)), hmg.HG2_SAFE_MAX_HEIGHT, style)

    def test_surface_pass_adds_local_features_without_overwriting_macro_form(self) -> None:
        for style, recipe in RAW.items():
            settings = hmg.GeneratorSettings(zones_x=1, zones_z=1, seed=13579)
            raw = recipe(settings)
            enhanced = enhance_planetary_surface(raw, settings, style)

            delta = enhanced.heights.astype(np.float32) - raw.heights.astype(np.float32)
            self.assertGreater(float(np.sqrt(np.mean(delta * delta))), 0.20, style)
            self.assertGreater(passable_fraction(enhanced.heights), 0.70, style)

            unchanged = float(np.mean(enhanced.heights == raw.heights))
            self.assertGreater(unchanged, 0.25, style)
            self.assertLess(unchanged, 0.90, style)

    def test_metadata_and_dimensions_are_preserved(self) -> None:
        settings = hmg.GeneratorSettings(zones_x=2, zones_z=1, seed=4242)
        raw = europa_fracture_plains(settings)
        enhanced = enhance_planetary_surface(raw, settings, "Europa Fracture Plains")
        self.assertEqual(enhanced.shape, raw.shape)
        self.assertEqual(enhanced.zones_x, raw.zones_x)
        self.assertEqual(enhanced.zones_z, raw.zones_z)
        self.assertEqual(enhanced.zone_bits, raw.zone_bits)
        self.assertEqual(enhanced.structure_version, raw.structure_version)
        self.assertEqual(enhanced.map_version, raw.map_version)


if __name__ == "__main__":
    unittest.main()
