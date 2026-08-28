import tempfile
import unittest
from pathlib import Path

import numpy as np

from bzr_heightmap.hg2 import HG2Map
from bzr_heightmap.lgt import compute_lgt_for_hg2, compute_lgt_lightmap, lgt_to_brightness, read_lgt, write_lgt
from bzr_heightmap.preview import make_lgt_preview_image


class LgtTests(unittest.TestCase):
    def test_dimensions_range_and_determinism_at_both_resolutions(self):
        y, x = np.mgrid[0:256, 0:512]
        heights = (1800 + 80 * np.sin(x / 31.0) + 50 * np.cos(y / 23.0)).astype(np.uint16)
        first = compute_lgt_lightmap(heights, 2, 1, lgt_zone_size=128)
        second = compute_lgt_lightmap(heights, 2, 1, lgt_zone_size=128)
        high_res = compute_lgt_lightmap(heights, 2, 1, lgt_zone_size=256)
        self.assertEqual(first.shape, (128, 256))
        self.assertEqual(high_res.shape, (256, 512))
        self.assertEqual(first.dtype, np.uint8)
        self.assertTrue(np.array_equal(first, second))
        self.assertGreaterEqual(int(first.min()), 0)
        self.assertLessEqual(int(first.max()), 255)

    def test_flat_terrain_and_ambient_floor(self):
        light = compute_lgt_lightmap(np.full((256, 256), 2000, dtype=np.uint16), 1, 1)
        self.assertEqual(np.unique(light).size, 1)
        self.assertAlmostEqual(float(light[0, 0]), 255.0 * np.sin(np.deg2rad(45.0)), delta=1.0)
        brightness = lgt_to_brightness(np.array([0, 255], dtype=np.uint8))
        self.assertAlmostEqual(float(brightness[0]), 0.25, places=6)
        self.assertAlmostEqual(float(brightness[1]), 1.0, places=6)

    def test_preview_applies_ambient_floor_and_matches_hg2_dimensions(self):
        heights = np.full((256, 256), 2000, dtype=np.uint16)
        image = make_lgt_preview_image(heights, 1, 1, lgt_zone_size=128)
        pixels = np.asarray(image)[:, :, 0]
        self.assertEqual(image.size, (256, 256))
        self.assertGreaterEqual(int(pixels.min()), 64)

    def test_sun_facing_slope_is_brighter_than_opposing_slope(self):
        z, x = np.mgrid[-128:128, -128:128]
        facing = (2000 + 3 * x - 3 * z).astype(np.float32)
        opposing = (2000 - 3 * x + 3 * z).astype(np.float32)
        facing_light = compute_lgt_lightmap(facing, 1, 1, lgt_zone_size=256)
        opposing_light = compute_lgt_lightmap(opposing, 1, 1, lgt_zone_size=256)
        self.assertGreater(float(np.mean(facing_light)), float(np.mean(opposing_light)))

    def test_hg2_to_lgt_block_mapping(self):
        y, x = np.mgrid[0:256, 0:256]
        heights = (1500 + x + 2 * y).astype(np.uint16)
        high = compute_lgt_lightmap(heights, 1, 1, lgt_zone_size=256)
        low = compute_lgt_lightmap(heights, 1, 1, lgt_zone_size=128)
        expected = np.rint(high.astype(np.float32).reshape(128, 2, 128, 2).mean(axis=(1, 3))).astype(np.uint8)
        self.assertLessEqual(int(np.max(np.abs(low.astype(np.int16) - expected.astype(np.int16)))), 1)

    def test_convenience_api_uses_hg2_dimensions(self):
        terrain = HG2Map(np.full((256, 512), 1000, dtype=np.uint16), 2, 1)
        self.assertEqual(compute_lgt_for_hg2(terrain).shape, (128, 256))

    def test_bordered_round_trip_preserves_south_first_orientation(self):
        zone_size = 128
        light = np.arange(2 * zone_size * 2 * zone_size, dtype=np.uint32).reshape(2 * zone_size, 2 * zone_size).astype(np.uint8)
        light[0, 0] = 231
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "layout.lgt"
            write_lgt(path, light, 2, 2)
            raw = path.read_bytes()
            loaded, zones_x, zones_z, loaded_zone_size = read_lgt(path, 2, 2)
        self.assertEqual(len(raw), 5 * zone_size * zone_size)
        self.assertEqual(set(raw[: zone_size * zone_size]), {231})
        self.assertEqual(raw[zone_size * zone_size : 2 * zone_size * zone_size], light[:zone_size, :zone_size].tobytes())
        self.assertEqual((zones_x, zones_z, loaded_zone_size), (2, 2, 128))
        self.assertTrue(np.array_equal(loaded, light))

    def test_companion_hg2_resolves_ambiguous_lgt_dimensions(self):
        light = np.arange(256 * 256, dtype=np.uint32).reshape(256, 256).astype(np.uint8)
        terrain = HG2Map(np.full((512, 512), 1000, dtype=np.uint16), 2, 2)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            terrain.write(root / "paired.HG2")
            write_lgt(root / "paired.lgt", light, 2, 2)
            loaded, zones_x, zones_z, zone_size = read_lgt(root / "paired.lgt")
        self.assertEqual((zones_x, zones_z, zone_size), (2, 2, 128))
        self.assertTrue(np.array_equal(loaded, light))

    def test_ambiguous_standalone_layout_requires_dimensions(self):
        light = np.zeros((256, 256), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ambiguous.lgt"
            write_lgt(path, light, 2, 2)
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                read_lgt(path)

    def test_unbordered_legacy_layout_reads_without_vertical_flip(self):
        zone_size = 128
        light = np.arange(zone_size * zone_size, dtype=np.uint32).reshape(zone_size, zone_size).astype(np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.lgt"
            path.write_bytes(light.tobytes())
            loaded, zones_x, zones_z, loaded_zone_size = read_lgt(path, 1, 1, zone_size=128)
        self.assertEqual((zones_x, zones_z, loaded_zone_size), (1, 1, 128))
        self.assertTrue(np.array_equal(loaded, light))


if __name__ == "__main__":
    unittest.main()
