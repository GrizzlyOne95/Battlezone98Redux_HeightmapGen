import unittest

import numpy as np
from scipy import ndimage

import bzr_heightmap as hmg
from bzr_heightmap.stock_detail import repair_stock_connectivity


class StockDetailGrammarTests(unittest.TestCase):
    def test_connectivity_repair_links_disconnected_shelves(self):
        heights = np.full((128, 128), 900.0, dtype=np.float32)
        heights[:, 64:] += 520.0
        heights[64:, :] += 430.0

        before = hmg.stock_connectivity_metrics(heights, max_slope_deg=15.0)
        repaired, protected = repair_stock_connectivity(
            heights,
            np.random.default_rng(12345),
            target_map_fraction=0.45,
            target_passable_fraction=0.65,
            max_connectors=4,
            max_slope_deg=15.0,
        )
        after = hmg.stock_connectivity_metrics(repaired, max_slope_deg=15.0)

        self.assertGreater(after["largest_map_fraction"], before["largest_map_fraction"] + 0.10)
        self.assertGreater(after["largest_passable_fraction"], before["largest_passable_fraction"] + 0.10)
        self.assertTrue(np.any(protected))

    def test_detail_pass_adds_discrete_micro_structure_without_destroying_flats(self):
        heights = np.full((128, 128), 1800, dtype=np.uint16)
        terrain = hmg.HG2Map(heights, 1, 1, zone_bits=7)
        settings = hmg.GeneratorSettings(zones_x=1, zones_z=1, seed=424242, detail=0.8, feature_density=0.7)

        enhanced = hmg.enhance_stock_terrain(terrain, settings, "Natural Badlands")
        self.assertEqual(enhanced.heights.shape, heights.shape)
        self.assertGreater(int(np.ptp(enhanced.heights)), 20)

        lap_before = float(np.percentile(np.abs(ndimage.laplace(heights.astype(np.float32))), 95))
        lap_after = float(np.percentile(np.abs(ndimage.laplace(enhanced.heights.astype(np.float32))), 95))
        self.assertGreater(lap_after, lap_before + 2.0)

        center = enhanced.heights[1:-1, 1:-1]
        exact_flat = (
            (center == enhanced.heights[:-2, 1:-1])
            & (center == enhanced.heights[2:, 1:-1])
            & (center == enhanced.heights[1:-1, :-2])
            & (center == enhanced.heights[1:-1, 2:])
        )
        self.assertGreater(float(np.mean(exact_flat)), 0.45)

    def test_enhancement_is_deterministic_and_stock_safe(self):
        yy, xx = np.mgrid[0:128, 0:128].astype(np.float32)
        heights = np.rint(1500.0 + 120.0 * np.sin(xx / 18.0) + 80.0 * np.cos(yy / 23.0)).astype(np.uint16)
        terrain = hmg.HG2Map(heights, 1, 1, zone_bits=7)
        settings = hmg.GeneratorSettings(zones_x=1, zones_z=1, seed=8675309, detail=0.65, feature_density=0.55)

        first = hmg.enhance_stock_terrain(terrain, settings, "Cratered Divide")
        second = hmg.enhance_stock_terrain(terrain, settings, "Cratered Divide")

        self.assertTrue(np.array_equal(first.heights, second.heights))
        self.assertGreaterEqual(int(first.heights.min()), 0)
        self.assertLessEqual(int(first.heights.max()), hmg.HG2_SAFE_MAX_HEIGHT)
        first.validate()


if __name__ == "__main__":
    unittest.main()
