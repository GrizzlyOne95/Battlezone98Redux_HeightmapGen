import tempfile
import unittest
from pathlib import Path

import numpy as np

from bzr_heightmap.hgt import (
    HGTFormatError,
    HGTMap,
    LEGACY_HEIGHT_MASK,
    LEGACY_ZONE_BYTES,
    LEGACY_ZONE_SIZE,
    box_blur,
    read_hg2_header,
    read_trn_zone_counts,
    upsample,
    zone_count_candidates,
)


def make_hgt(heights: np.ndarray, zones_x: int = 1, zones_z: int = 1,
             flags: int = 0) -> HGTMap:
    raw = (heights.astype(np.uint16) & LEGACY_HEIGHT_MASK) | np.uint16(flags << 12)
    return HGTMap(raw.astype(np.uint16), zones_x, zones_z)


class HGTFormatTests(unittest.TestCase):
    def test_read_write_round_trip_is_byte_exact(self):
        rng = np.random.default_rng(7)
        blob = rng.integers(0, 0x10000, size=2 * 2 * LEGACY_ZONE_SIZE ** 2, dtype=np.uint64
                            ).astype("<u2")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "a.hgt"
            dst = Path(tmp) / "b.hgt"
            blob.tofile(src)
            hgt = HGTMap.read(src, 2, 2)
            hgt.write(dst)
            self.assertEqual(src.read_bytes(), dst.read_bytes())

    def test_zone_tiling_matches_engine_indexing(self):
        # FUN_00785f50: index = (z&127)*128 + (x&127) + (x>>7)*0x4000 + (z>>7)*zones_x*0x4000
        zones_x, zones_z = 2, 3
        count = zones_x * zones_z * LEGACY_ZONE_SIZE ** 2
        blob = np.arange(count, dtype="<u2") & LEGACY_HEIGHT_MASK
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.hgt"
            blob.tofile(path)
            hgt = HGTMap.read(path, zones_x, zones_z)
        for x, z in ((0, 0), (127, 0), (128, 0), (0, 128), (255, 383), (200, 300)):
            expected = blob[
                (z & 127) * 128 + (x & 127)
                + (x >> 7) * 0x4000
                + (z >> 7) * zones_x * 0x4000
            ]
            self.assertEqual(hgt.heights[z, x], expected, f"mismatch at ({x},{z})")

    def test_height_is_masked_to_twelve_bits_and_flags_preserved(self):
        hgt = make_hgt(np.full((128, 128), 0x0123, dtype=np.uint16), flags=0x8)
        self.assertEqual(int(hgt.heights.max()), 0x0123)
        self.assertEqual(int(hgt.flags.max()), 0x8)
        self.assertEqual(int(hgt.raw[0, 0]), 0x8123)

    def test_wrong_dimensions_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.hgt"
            np.zeros(LEGACY_ZONE_SIZE ** 2, dtype="<u2").tofile(path)
            with self.assertRaises(HGTFormatError):
                HGTMap.read(path, 2, 2)

    def test_truncated_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.hgt"
            path.write_bytes(b"\0" * (LEGACY_ZONE_BYTES + 1024))
            with self.assertRaises(HGTFormatError):
                HGTMap.read_auto(path)

    def test_zone_count_candidates_prefers_square(self):
        self.assertEqual(zone_count_candidates(16 * LEGACY_ZONE_BYTES)[0], (4, 4))
        self.assertEqual(zone_count_candidates(12 * LEGACY_ZONE_BYTES)[0], (3, 4))
        self.assertEqual(zone_count_candidates(141312), [])


class TrnTests(unittest.TestCase):
    def _trn(self, tmp, text):
        path = Path(tmp) / "m.trn"
        path.write_text(text)
        return path

    def test_zone_counts_from_size_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._trn(tmp, "[Size]\nMinX=0\nWidth=5120\nDepth=3840\nHeight=0\n")
            self.assertEqual(read_trn_zone_counts(path), (4, 3))

    def test_first_size_section_wins(self):
        # lcbench.trn declares 5120 then 3840; the shipped lcbench.hg2 is 4x4.
        with tempfile.TemporaryDirectory() as tmp:
            path = self._trn(tmp, "[Size]\nWidth=5120\nDepth=5120\n\n[Sky]\nX=1\n\n"
                                  "[Size]\nWidth=3840\nDepth=3840\n")
            self.assertEqual(read_trn_zone_counts(path), (4, 4))

    def test_non_multiple_of_zone_size_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._trn(tmp, "[Size]\nWidth=5000\nDepth=5120\n")
            self.assertIsNone(read_trn_zone_counts(path))

    def test_missing_section_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(read_trn_zone_counts(self._trn(tmp, "[Sky]\nX=1\n")))


class ConversionTests(unittest.TestCase):
    def test_flat_terrain_stays_perfectly_flat(self):
        for smoothing in (False, True):
            for mode in ("engine", "half-up"):
                hgt = make_hgt(np.full((128, 128), 1234, dtype=np.uint16))
                out = hgt.to_hg2(smoothing=smoothing, rounding=mode).heights
                self.assertEqual(np.unique(out).tolist(), [1234],
                                 f"smoothing={smoothing} rounding={mode}")

    def test_output_dimensions_double_the_source(self):
        hgt = make_hgt(np.zeros((128 * 3, 128 * 2), dtype=np.uint16), zones_x=2, zones_z=3)
        hg2 = hgt.to_hg2()
        self.assertEqual(hg2.heights.shape, (3 * 256, 2 * 256))
        self.assertEqual((hg2.zones_x, hg2.zones_z), (2, 3))
        self.assertEqual(hg2.zone_bits, 8)

    def test_legacy_vertices_are_reproduced_exactly(self):
        rng = np.random.default_rng(11)
        heights = rng.integers(0, 4096, size=(128, 128), dtype=np.int64).astype(np.uint16)
        hgt = make_hgt(heights, flags=0x8)
        for mode in ("engine", "half-up"):
            out = hgt.to_hg2(rounding=mode).heights
            self.assertTrue(np.array_equal(out[::2, ::2].astype(np.int32), hgt.heights),
                            f"rounding={mode} moved an authored vertex")

    def test_smoothing_destroys_authored_vertices(self):
        # The whole reason this module exists.
        rng = np.random.default_rng(12)
        heights = rng.integers(0, 4096, size=(128, 128), dtype=np.int64).astype(np.uint16)
        hgt = make_hgt(heights)
        out = hgt.to_hg2(smoothing=True).heights
        self.assertFalse(np.array_equal(out[::2, ::2].astype(np.int32), hgt.heights))

    def test_staircase_keeps_its_discrete_levels(self):
        heights = np.zeros((128, 128), dtype=np.uint16)
        for step in range(8):
            heights[:, step * 16:(step + 1) * 16] = step * 256
        hgt = make_hgt(heights)
        sharp = hgt.to_hg2().heights
        blurred = hgt.to_hg2(smoothing=True).heights
        # A 16-sample legacy tread becomes 32 HG2 columns, so the first riser
        # sits between output columns 30 (legacy 15, height 0) and 32
        # (legacy 16, height 256). Unsmoothed, exactly one interpolated column
        # lies between the two treads; the blur widens that transition.
        def transition_width(grid):
            row = grid[0, :40].astype(np.int64)
            return int(np.count_nonzero((row > 0) & (row < 256)))

        self.assertEqual(transition_width(sharp), 1)
        self.assertEqual(transition_width(blurred), 3)
        # Unsmoothed, every tread is still exactly one of the authored levels;
        # the blur roughly doubles the number of distinct heights in the map.
        self.assertTrue(set(np.unique(sharp[:, 0:31]).tolist()) <= {0, 128})
        self.assertEqual(np.unique(sharp[:, 0:30]).tolist(), [0])
        self.assertLess(np.unique(sharp).size, np.unique(blurred).size)

    def test_hard_step_edge_stays_within_one_cell(self):
        heights = np.zeros((128, 128), dtype=np.uint16)
        heights[:, 64:] = 1000
        sharp = make_hgt(heights).to_hg2().heights
        # Legacy cell 63->64 is the only place a value between 0 and 1000 may
        # appear: columns 0..126 are still 0 and 129.. are still 1000.
        self.assertTrue(np.all(sharp[:, :127] == 0))
        self.assertTrue(np.all(sharp[:, 129:] == 1000))

    def test_single_impulse_is_not_spread_without_smoothing(self):
        heights = np.zeros((128, 128), dtype=np.uint16)
        heights[64, 64] = 4000
        sharp = make_hgt(heights).to_hg2().heights
        blurred = make_hgt(heights).to_hg2(smoothing=True).heights
        self.assertEqual(int(sharp[128, 128]), 4000)      # the peak survives
        self.assertLess(int(blurred[128, 128]), 4000)     # the blur cuts it down
        self.assertLess(int(np.count_nonzero(sharp)), int(np.count_nonzero(blurred)))

    def test_interpolation_is_planar_not_bilinear(self):
        # A cell whose diagonal split matters: bilinear would give 500 at the
        # cell centre, the engine's triangulation gives the plane value instead.
        heights = np.zeros((128, 128), dtype=np.uint16)
        heights[0, 0] = 0
        heights[0, 1] = 1000
        heights[1, 0] = 0
        heights[1, 1] = 1000
        out = make_hgt(heights).to_hg2(rounding="half-up").heights
        self.assertEqual(int(out[0, 1]), 500)     # midpoint of the 0..1000 edge
        self.assertEqual(int(out[0, 2]), 1000)    # the far vertex, exact

    def test_rounding_modes_differ_by_at_most_one_unit(self):
        rng = np.random.default_rng(13)
        heights = rng.integers(0, 4096, size=(128, 128), dtype=np.int64).astype(np.uint16)
        hgt = make_hgt(heights)
        a = hgt.to_hg2(rounding="engine").heights.astype(np.int64)
        b = hgt.to_hg2(rounding="half-up").heights.astype(np.int64)
        self.assertLessEqual(int(np.abs(a - b).max()), 1)

    def test_unknown_rounding_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            upsample(np.zeros((128, 128), dtype=np.int32), 1, 1, rounding="bicubic")

    def test_conversion_never_leaves_the_source_height_range(self):
        rng = np.random.default_rng(14)
        heights = rng.integers(300, 900, size=(128, 128), dtype=np.int64).astype(np.uint16)
        hgt = make_hgt(heights)
        out = hgt.to_hg2().heights
        self.assertGreaterEqual(int(out.min()), int(hgt.heights.min()))
        self.assertLessEqual(int(out.max()), int(hgt.heights.max()))


class BlurTests(unittest.TestCase):
    def test_box_blur_rounds_half_away_from_zero(self):
        # 2x2 grid: every sample sees all four, sum 0+0+1+2 = 3 over 4 -> 0.75 -> 1
        grid = np.array([[0, 0], [1, 2]], dtype=np.uint16)
        self.assertTrue(np.all(box_blur(grid) == 1))

    def test_box_blur_only_counts_in_bounds_neighbours(self):
        grid = np.full((3, 3), 10, dtype=np.uint16)
        grid[1, 1] = 19
        out = box_blur(grid)
        # corner sees 4 samples (three 10s and the 19): 49/4 = 12.25 -> 12
        self.assertEqual(int(out[0, 0]), 12)
        # centre sees all nine: 99/9 = 11
        self.assertEqual(int(out[1, 1]), 11)

    def test_box_blur_preserves_flat(self):
        self.assertTrue(np.all(box_blur(np.full((16, 16), 777, dtype=np.uint16)) == 777))


class Hg2HeaderTests(unittest.TestCase):
    def test_written_header_passes_the_engine_validation(self):
        hgt = make_hgt(np.full((128, 128), 500, dtype=np.uint16))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.hg2"
            hgt.to_hg2().write(path)
            header = read_hg2_header(path)
        self.assertTrue(header["header_valid"])
        self.assertTrue(header["size_consistent"])
        self.assertEqual(header["structure_version"], 1)
        self.assertEqual(header["zone_bits"], 8)
        self.assertEqual((header["zones_x"], header["zones_z"]), (1, 1))
        self.assertGreaterEqual(header["map_version"], 10)
        self.assertEqual(header["size"], 12 + 256 * 256 * 2)

    def test_map_version_below_ten_is_reported_invalid(self):
        hgt = make_hgt(np.zeros((128, 128), dtype=np.uint16))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.hg2"
            hgt.to_hg2(map_version=9).write(path)
            self.assertFalse(read_hg2_header(path)["header_valid"])


if __name__ == "__main__":
    unittest.main()
