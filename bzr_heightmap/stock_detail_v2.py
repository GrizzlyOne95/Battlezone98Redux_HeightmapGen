from __future__ import annotations

import numpy as np
from scipy import ndimage

from .stock_detail import _slope_degrees

# Stable topology helpers retained from the V2 research pass and used by V5.
# The superseded V2/V3/V4 generation algorithms are intentionally not kept in
# the production branch surface.


def _component_state(a: np.ndarray, max_slope_deg: float = 15.0):
    passable = _slope_degrees(a) <= float(max_slope_deg)
    labels, n = ndimage.label(passable, structure=np.ones((3, 3), dtype=np.uint8))
    counts = np.bincount(labels.ravel())[1:] if n else np.asarray([], dtype=np.int64)
    return passable, labels, counts


def _nearest_component_pair(
    a: np.ndarray,
    labels: np.ndarray,
    counts: np.ndarray,
    rng: np.random.Generator,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if counts.size < 2:
        return None
    main_label = int(np.argmax(counts)) + 1
    main_mask = labels == main_label
    distance, nearest = ndimage.distance_transform_edt(~main_mask, return_indices=True)
    min_component = max(96, int(a.size * 0.003))
    choices = [
        i + 1
        for i in np.argsort(counts)[::-1]
        if i + 1 != main_label and int(counts[i]) >= min_component
    ]
    best = None
    for label_id in choices[:10]:
        ys, xs = np.nonzero(labels == label_id)
        if ys.size == 0:
            continue
        if ys.size > 10000:
            idx = rng.choice(ys.size, size=10000, replace=False)
            ys, xs = ys[idx], xs[idx]
        ny = nearest[0, ys, xs]
        nx = nearest[1, ys, xs]
        spatial = distance[ys, xs]
        height = np.abs(a[ys, xs] - a[ny, nx])
        score = spatial + height / 18.0
        j = int(np.argmin(score))
        candidate = (
            float(score[j]),
            (int(ys[j]), int(xs[j])),
            (int(ny[j]), int(nx[j])),
        )
        if best is None or candidate[0] < best[0]:
            best = candidate
    return None if best is None else (best[1], best[2])


def _random_centers(
    mask: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0 or count <= 0:
        return []
    count = min(int(count), int(ys.size))
    indices = rng.choice(ys.size, size=count, replace=False)
    return [(int(ys[i]), int(xs[i])) for i in np.atleast_1d(indices)]


__all__ = ["_component_state", "_nearest_component_pair", "_random_centers"]
