# HG2 Corpus Summary (local discovery)

- total paths discovered: 509
- valid parses: 507
- invalid/corrupt: 2
- unique content hashes: 275
- duplicate groups: 133 (extra duplicate paths 232)
- path classification: {'synthetic_test': 23, 'authored': 458, 'generated_sample': 26}
- unique classification: {'authored': 249, 'generated_sample': 26, 'synthetic_test': 0}
- companion HG2/LGT path pairs: 447
- unique contents with LGT: 238
- unique authored contents with LGT: 238
- LGT layouts by valid path pair: {'bordered_256': 433, 'bordered_128': 9, 'unrecognized': 5}
- LGT layout sets by unique HG2 content: {'bordered_256': 230, 'bordered_128': 5, 'unrecognized': 2, 'bordered_256+unrecognized': 1}
- dimensions: {'4x4': 259, '3x3': 86, '2x2': 80, '4x3': 26, '1x1': 45, '2x4': 2, '3x4': 5, '4x5': 3, '8x8': 1}
- provenance: {'test_synthetic': 23, 'addon': 4, 'rotbd': 89, 'unknown': 113, 'stock': 152, 'isdf_chronicles': 56, 'workshop': 1, 'generated_sample': 26, 'campaign': 8, 'bzp': 35}

## Authored aggregate metrics (mean over unique authored contents)
- exact_flat_pct: 39.49
- dominant_level_pct: 32.53
- median_slope_deg: 7.75
- p95_slope_deg: 44.17
- range: 2419.92
- shelf_count_gt2pct: 1.76
- shelf_area_pct: 39.23
- largest_flat_component_pct: 27.05
- corridor_median_width_m: 613.38

Records written to hg2_corpus_report.json and hg2_corpus_report.csv (paths sanitized in summary).
