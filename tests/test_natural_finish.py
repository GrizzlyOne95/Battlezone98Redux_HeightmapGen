from __future__ import annotations

import unittest

import numpy as np
from scipy import ndimage

import bzr_heightmap as hmg
from bzr_heightmap.natural_finish import enhance_natural_finish
from bzr_heightmap.planetary import mars_rift
from bzr_heightmap.recipes import campaign_canyon_network, compartmented_plateau, escarpment_stronghold, sparse_mission_field


RAW = {
    "Campaign Canyon Network": campaign_canyon_network,
    "Compartmented Plateau": compartmented_plateau,
    "Escarpment Stronghold": escarpment_stronghold,
    "Mars Rift": mars_rift,
}


def residual_rms(a: np.ndarray) -> float:
    f = np.asarray(a, dtype=np.float32)
    blur = ndimage.gaussian_filter(f, 2.0, mode="reflect")
    return float(np.sqrt(np.mean((f - blur) ** 2)))


def passable_fraction(a: np.ndarray) -> float:
    f = np.asarray(a, dtype=np.float32)
    gy, gx = np.gradient(f * 0.1, 5.0, 5.0)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    return float(np.mean(slope <= 15.0))


class NaturalFinishTests(unittest.TestCase):
    def test_finish_is_deterministic_and_authoring_safe(self) -> None:
        for style in RAW:
            settings = hmg.GeneratorSettings(zones_x=1, zones_z=1, seed=24680)
            a = hmg.generate(style, settings)
            b = hmg.generate(style, settings)
            self.assertTrue(np.array_equal(a.heights, b.heights), style)
            a.validate()

    def test_finish_adds_local_detail_without_consuming_traversal(self) -> None:
        for style, recipe in RAW.items():
            settings = hmg.GeneratorSettings(zones_x=1, zones_z=1, seed=13579)
            raw = recipe(settings)
            enhanced = enhance_natural_finish(raw, settings, style)

            changed = float(np.mean(enhanced.heights != raw.heights))
            self.assertGreater(changed, 0.002, style)
            self.assertLess(changed, 0.28, style)
            self.assertGreaterEqual(residual_rms(enhanced.heights), residual_rms(raw.heights), style)
            self.assertGreater(passable_fraction(enhanced.heights), 0.62, style)

    def test_sparse_mission_field_remains_intentionally_unprofiled(self) -> None:
        settings = hmg.GeneratorSettings(zones_x=2, zones_z=2, seed=4242)
        raw = sparse_mission_field(settings)
        generated = hmg.generate("Sparse Mission Field", settings)
        self.assertTrue(np.array_equal(raw.heights, generated.heights))

    def test_metadata_and_dimensions_are_preserved(self) -> None:
        settings = hmg.GeneratorSettings(zones_x=2, zones_z=1, seed=4242)
        raw = campaign_canyon_network(settings)
        enhanced = enhance_natural_finish(raw, settings, "Campaign Canyon Network")
        self.assertEqual(enhanced.shape, raw.shape)
        self.assertEqual(enhanced.zones_x, raw.zones_x)
        self.assertEqual(enhanced.zones_z, raw.zones_z)
        self.assertEqual(enhanced.zone_bits, raw.zone_bits)
        self.assertEqual(enhanced.structure_version, raw.structure_version)
        self.assertEqual(enhanced.map_version, raw.map_version)


if __name__ == "__main__":
    unittest.main()
