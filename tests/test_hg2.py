import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from bzr_heightmap.hg2 import (
    HG2_MAP_VERSION,
    HG2_SAFE_MAX_HEIGHT,
    HG2_STORAGE_MAX_HEIGHT,
    HG2_STRUCTURE_VERSION,
    HG2Map,
)


class HG2CodecTests(unittest.TestCase):
    def test_header_and_zone_major_payload_are_canonical(self):
        zone_bits = 2
        zone_size = 1 << zone_bits
        heights = np.zeros((zone_size * 2, zone_size * 2), dtype=np.uint16)
        heights[0:zone_size, 0:zone_size] = 100
        heights[0:zone_size, zone_size:] = 200
        heights[zone_size:, 0:zone_size] = 300
        heights[zone_size:, zone_size:] = 400
        terrain = HG2Map(heights, zones_x=2, zones_z=2, zone_bits=zone_bits)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "layout.hg2"
            terrain.write(path)
            data = path.read_bytes()

        self.assertEqual(
            struct.unpack("<HHHHI", data[:12]),
            (HG2_STRUCTURE_VERSION, zone_bits, 2, 2, HG2_MAP_VERSION),
        )
        raw = np.frombuffer(data[12:], dtype="<u2")
        block = zone_size * zone_size
        self.assertTrue(np.all(raw[0 * block : 1 * block] == 100))
        self.assertTrue(np.all(raw[1 * block : 2 * block] == 200))
        self.assertTrue(np.all(raw[2 * block : 3 * block] == 300))
        self.assertTrue(np.all(raw[3 * block : 4 * block] == 400))

    def test_codec_preserves_full_13_bit_storage_range(self):
        heights = np.array([[0, HG2_SAFE_MAX_HEIGHT], [4096, HG2_STORAGE_MAX_HEIGHT]], dtype=np.uint16)
        terrain = HG2Map(heights, zones_x=1, zones_z=1, zone_bits=1)

        with self.assertRaises(ValueError):
            terrain.validate()

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "storage.hg2"
            terrain.write(path)
            loaded = HG2Map.read(path)

        self.assertTrue(np.array_equal(loaded.heights, heights))

    def test_png16_uses_full_range_height_times_eight(self):
        heights = np.array([[0, 1], [HG2_SAFE_MAX_HEIGHT, HG2_STORAGE_MAX_HEIGHT]], dtype=np.uint16)
        terrain = HG2Map(heights, zones_x=1, zones_z=1, zone_bits=1)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "height.png"
            terrain.write_png16(path)
            saved = np.array(Image.open(path), dtype=np.uint16)

        expected = (heights.astype(np.uint32) * 8).astype(np.uint16)
        self.assertTrue(np.array_equal(saved, expected))
        self.assertEqual(int(saved.max()), 65528)

    def test_read_rejects_truncated_payload(self):
        zone_bits = 2
        expected_samples = (1 << zone_bits) ** 2
        header = struct.pack("<HHHHI", HG2_STRUCTURE_VERSION, zone_bits, 1, 1, HG2_MAP_VERSION)
        payload = np.arange(expected_samples - 1, dtype="<u2").tobytes()

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "truncated.hg2"
            path.write_bytes(header + payload)
            with self.assertRaisesRegex(ValueError, "sample count mismatch"):
                HG2Map.read(path)


if __name__ == "__main__":
    unittest.main()
