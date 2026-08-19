import tempfile
import unittest
from pathlib import Path

import numpy as np

import bzr_heightmap as hmg


class HeightmapGeneratorTests(unittest.TestCase):
    def test_all_recipes_are_deterministic_and_safe(self):
        settings = hmg.GeneratorSettings(zones_x=1, zones_z=1, seed=12345)
        for name in hmg.RECIPES:
            with self.subTest(style=name):
                a = hmg.generate(name, settings)
                b = hmg.generate(name, settings)
                self.assertTrue(np.array_equal(a.heights, b.heights))
                self.assertGreaterEqual(int(a.heights.min()), 0)
                self.assertLessEqual(int(a.heights.max()), hmg.HG2_SAFE_MAX_HEIGHT)

    def test_hg2_round_trip_preserves_height_array(self):
        settings = hmg.GeneratorSettings(zones_x=2, zones_z=1, seed=7)
        original = hmg.generate("Campaign Canyon Network", settings)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "roundtrip.hg2"
            original.write(path)
            loaded = hmg.HG2Map.read(path)
        self.assertEqual((loaded.zones_x, loaded.zones_z), (2, 1))
        self.assertTrue(np.array_equal(original.heights, loaded.heights))

    def test_authored_corridor_keeps_exact_flat_core(self):
        settings = hmg.GeneratorSettings(zones_x=1, zones_z=1, seed=3, detail=1.0)
        terrain = hmg.generate("Campaign Canyon Network", settings)
        metrics = hmg.terrain_metrics(terrain.heights)
        self.assertGreater(metrics["exact_flat_pct"], 5.0)

    def test_symmetry_does_not_eliminate_authored_flats(self):
        settings = hmg.GeneratorSettings(zones_x=1, zones_z=1, seed=3, symmetry="4-way")
        terrain = hmg.generate("Campaign Canyon Network", settings)
        self.assertTrue(np.array_equal(terrain.heights, np.fliplr(terrain.heights)))
        self.assertTrue(np.array_equal(terrain.heights, np.flipud(terrain.heights)))
        self.assertGreater(hmg.terrain_metrics(terrain.heights)["exact_flat_pct"], 5.0)


if __name__ == "__main__":
    unittest.main()
