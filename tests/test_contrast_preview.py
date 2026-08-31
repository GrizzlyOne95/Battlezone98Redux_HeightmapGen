import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import bzr_heightmap as hmg


class VerticalScaleTests(unittest.TestCase):
    def make_terrain(self) -> hmg.HG2Map:
        x = np.linspace(900, 3100, 256, dtype=np.float32)
        heights = np.tile(x, (256, 1))
        heights[64:128, 64:128] = 1800
        return hmg.HG2Map(np.rint(heights).astype(np.uint16), 1, 1)

    def test_expansion_and_compression_scale_range(self):
        terrain = self.make_terrain()
        source_range = int(np.ptp(terrain.heights))
        compressed = hmg.apply_vertical_scale(terrain, 0.5)
        expanded = hmg.apply_vertical_scale(terrain, 1.25)
        self.assertAlmostEqual(int(np.ptp(compressed.heights)) / source_range, 0.5, delta=0.01)
        self.assertGreater(int(np.ptp(expanded.heights)), source_range)

    def test_clamps_to_safe_hg2_range(self):
        heights = np.full((256, 256), 2048, dtype=np.uint16)
        heights[0, 0] = 0
        heights[-1, -1] = hmg.HG2_SAFE_MAX_HEIGHT
        scaled = hmg.apply_vertical_scale(hmg.HG2Map(heights, 1, 1), 2.5)
        self.assertEqual(int(scaled.heights.min()), 0)
        self.assertEqual(int(scaled.heights.max()), hmg.HG2_SAFE_MAX_HEIGHT)

    def test_exact_flat_regions_remain_exact_and_source_is_unchanged(self):
        terrain = self.make_terrain()
        before = terrain.heights.copy()
        scaled = hmg.apply_vertical_scale(terrain, 0.73)
        self.assertEqual(np.unique(scaled.heights[64:128, 64:128]).size, 1)
        self.assertTrue(np.array_equal(terrain.heights, before))

    def test_scaling_is_deterministic(self):
        terrain = self.make_terrain()
        first = hmg.apply_vertical_scale(terrain, 1.17)
        second = hmg.apply_vertical_scale(terrain, 1.17)
        self.assertTrue(np.array_equal(first.heights, second.heights))

    def test_rejects_nonpositive_and_nonfinite_scale(self):
        terrain = self.make_terrain()
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                hmg.apply_vertical_scale(terrain, value)


class HeightPreviewTests(unittest.TestCase):
    def test_fixed_mapping_dimensions_and_orientation(self):
        heights = np.array([[0, 2048, 4095], [4095, 1024, 0]], dtype=np.uint16)
        image = hmg.make_hg2_height_image(heights)
        pixels = np.asarray(image)[:, :, 0]
        self.assertEqual(image.size, (3, 2))
        self.assertEqual((int(pixels[0, 0]), int(pixels[0, 2])), (0, 255))
        self.assertEqual((int(pixels[1, 0]), int(pixels[1, 2])), (255, 0))
        self.assertEqual(int(pixels[0, 1]), 127)

    def test_preview_does_not_mutate_source(self):
        heights = np.arange(256 * 256, dtype=np.uint16).reshape(256, 256) & 0x0FFF
        before = heights.copy()
        hmg.make_hg2_height_image(heights)
        self.assertTrue(np.array_equal(heights, before))

    def test_png16_export_preserves_height_times_sixteen(self):
        heights = np.arange(256 * 256, dtype=np.uint16).reshape(256, 256) & 0x0FFF
        terrain = hmg.HG2Map(heights, 1, 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "height.png"
            terrain.write_png16(path)
            loaded = np.asarray(Image.open(path), dtype=np.uint16)
        self.assertTrue(np.array_equal(loaded, heights.astype(np.uint32) * 16))


if __name__ == "__main__":
    unittest.main()
