import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from bzr_heightmap.gui_logic import LatestJobCoordinator, RawCacheKey, apply_cached_contrast, is_contrast_only_change, raw_generation_settings
from bzr_heightmap.hg2 import HG2Map
from bzr_heightmap.settings import GeneratorSettings
from scripts.analyze_local_corpus import (
    AUTHORED,
    GENERATED_SAMPLE,
    SYNTHETIC_TEST,
    aggregate_report,
    analyze_file,
    classify_terrain_class,
    discover_hg2_files,
)


class CorpusTests(unittest.TestCase):
    def test_case_insensitive_discovery(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            (root / "a.HG2").write_bytes(b"x")
            (root / "b.hg2").write_bytes(b"x")
            (root / "ignore.txt").write_bytes(b"x")
            names = {path.name for path in discover_hg2_files([root])}
        self.assertEqual(names, {"a.HG2", "b.hg2"})

    def test_normalized_windows_path_classification(self):
        generated = Path(r"C:\work\Battlezone98Redux_HeightmapGen\samples\hg2\sample.HG2")
        synthetic = Path(r"C:\work\BZR-OpenShim\reverse_engineering\test_missions\pilot\pilot.hg2")
        authored = Path(r"C:\games\Battlezone 98 Redux\addon\campaign\misn01.hg2")
        self.assertEqual(classify_terrain_class(generated), GENERATED_SAMPLE)
        self.assertEqual(classify_terrain_class(synthetic), SYNTHETIC_TEST)
        self.assertEqual(classify_terrain_class(authored), AUTHORED)

    def test_relative_sample_path_uses_resolved_repository_components(self):
        terrain = HG2Map(np.full((256, 256), 500, dtype=np.uint16), 1, 1)
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir) / "Battlezone98Redux_HeightmapGen"
            sample = repo / "samples" / "hg2" / "sample.HG2"
            sample.parent.mkdir(parents=True)
            terrain.write(sample)
            try:
                os.chdir(repo)
                record = analyze_file(Path("samples/hg2/sample.HG2"))
            finally:
                os.chdir(original_cwd)
        self.assertEqual(record["terrain_class"], GENERATED_SAMPLE)

    def test_deduplication_corrupt_input_and_accounting(self):
        terrain = HG2Map(np.full((256, 256), 1234, dtype=np.uint16), 1, 1)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            terrain.write(root / "first.HG2")
            terrain.write(root / "second.hg2")
            (root / "broken.HG2").write_bytes(b"bad")
            records = [analyze_file(path) for path in discover_hg2_files([root])]
        summary = aggregate_report(records)
        self.assertEqual((summary["total_paths"], summary["valid"], summary["invalid"]), (3, 2, 1))
        self.assertEqual(summary["unique_content_hashes"], 1)
        self.assertEqual(summary["duplicate_groups"], 1)
        self.assertEqual(summary["duplicate_paths_extra"], 1)
        self.assertEqual(sum(summary["unique_classification"].values()), 1)

    def test_analysis_never_writes_source_map(self):
        terrain = HG2Map(np.full((256, 256), 987, dtype=np.uint16), 1, 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "source.HG2"
            terrain.write(path)
            before_bytes = path.read_bytes()
            before_mtime = path.stat().st_mtime_ns
            record = analyze_file(path)
            after_mtime = path.stat().st_mtime_ns
            after_bytes = path.read_bytes()
        self.assertTrue(record["valid"])
        self.assertEqual(after_bytes, before_bytes)
        self.assertEqual(after_mtime, before_mtime)


class GuiLogicTests(unittest.TestCase):
    def setUp(self):
        self.settings = GeneratorSettings(zones_x=3, zones_z=3, seed=42, vertical_scale=1.0)

    def test_raw_cache_key_ignores_contrast_but_invalidates_generation_inputs(self):
        key = RawCacheKey.from_settings("Terraced Labyrinth", self.settings)
        contrast_key = RawCacheKey.from_settings("Terraced Labyrinth", replace(self.settings, vertical_scale=0.5))
        self.assertTrue(is_contrast_only_change(key, contrast_key))
        for changed_style, changed_settings in (
            ("Mountain Basin", self.settings),
            ("Terraced Labyrinth", replace(self.settings, seed=43)),
            ("Terraced Labyrinth", replace(self.settings, zones_x=4)),
            ("Terraced Labyrinth", replace(self.settings, relief=1.2)),
            ("Terraced Labyrinth", replace(self.settings, detail=0.8)),
        ):
            with self.subTest(style=changed_style, settings=changed_settings):
                self.assertNotEqual(key, RawCacheKey.from_settings(changed_style, changed_settings))

    def test_cached_contrast_reuses_layout_without_mutating_raw(self):
        heights = np.tile(np.arange(256, dtype=np.uint16), (256, 1)) + 1000
        raw = HG2Map(heights, 1, 1)
        before = raw.heights.copy()
        scaled = apply_cached_contrast(raw, 0.5)
        self.assertTrue(np.array_equal(raw.heights, before))
        self.assertLess(int(np.ptp(scaled.heights)), int(np.ptp(raw.heights)))

    def test_raw_generation_settings_only_resets_vertical_scale(self):
        settings = replace(self.settings, vertical_scale=1.4, relief=1.3, naturalization=0.2)
        raw = raw_generation_settings(settings)
        self.assertEqual(raw.vertical_scale, 1.0)
        self.assertEqual(raw.relief, 1.3)
        self.assertEqual(raw.naturalization, 0.2)

    def test_latest_job_coalesces_and_rejects_stale_results(self):
        jobs = LatestJobCoordinator()
        first = jobs.schedule()
        self.assertEqual(jobs.start_latest(), first)
        second = jobs.schedule()
        third = jobs.schedule()
        self.assertFalse(jobs.accepts(first))
        self.assertTrue(jobs.finish(first))
        self.assertEqual(jobs.start_latest(), third)
        self.assertFalse(jobs.accepts(second))
        self.assertTrue(jobs.accepts(third))
        self.assertFalse(jobs.finish(third))

    def test_close_blocks_pending_and_completed_jobs(self):
        jobs = LatestJobCoordinator()
        revision = jobs.schedule()
        self.assertEqual(jobs.start_latest(), revision)
        jobs.close()
        self.assertFalse(jobs.accepts(revision))
        self.assertFalse(jobs.finish(revision))
        self.assertIsNone(jobs.start_latest())


if __name__ == "__main__":
    unittest.main()
