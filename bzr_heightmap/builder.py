from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from .hg2 import HG2Map, HG2_MAX_HEIGHT
from .noise import (
    fbm,
    polyline_distance,
    ridged_fbm,
    signed_blob_field,
    smoothstep01,
    variable_corridor_mask,
)
from .settings import GeneratorSettings


class TerrainBuilder:
    def __init__(self, settings: GeneratorSettings):
        self.settings = settings
        self.rng = np.random.default_rng(int(settings.seed) & 0xFFFFFFFF)
        self.h, self.w = settings.shape
        self.a = np.zeros((self.h, self.w), dtype=np.float32)
        self.protected = np.zeros((self.h, self.w), dtype=bool)

    def set_level(self, value: float) -> "TerrainBuilder":
        self.a.fill(float(value))
        return self

    def add_fbm(self, amplitude: float, feature_px: float, ridged: bool = False, octaves: int = 5) -> "TerrainBuilder":
        noise = ridged_fbm(self.a.shape, feature_px, self.rng, octaves) if ridged else fbm(self.a.shape, feature_px, self.rng, octaves=octaves)
        self.a += noise * float(amplitude)
        return self

    def add_terraced_blobs(
        self,
        levels: Sequence[float],
        feature_px: float,
        edge_px: float,
        threshold_bias: float = 0.0,
        warp_px: float = 0.0,
    ) -> "TerrainBuilder":
        if len(levels) < 2:
            return self
        field = signed_blob_field(self.a.shape, feature_px, self.rng, warp_px)
        bias = np.clip(float(threshold_bias), -0.65, 0.65)
        probs = np.linspace(0.0, 1.0, len(levels) + 1, dtype=np.float32)[1:-1]
        probs = np.clip(probs + bias * 0.18, 0.08, 0.92)
        thresholds = np.quantile(field, probs)
        idx = np.digitize(field, thresholds)
        terraced = np.take(np.asarray(levels, dtype=np.float32), idx)
        if edge_px > 0:
            terraced = ndimage.gaussian_filter(terraced, sigma=float(edge_px), mode="reflect")
        if self.settings.naturalization > 0:
            gy, gx = np.gradient(terraced)
            edge = np.hypot(gx, gy)
            edge /= max(float(np.percentile(edge, 99)), 1e-5)
            rough = fbm(self.a.shape, max(10.0, feature_px * 0.18), self.rng, octaves=3)
            terraced += rough * np.clip(edge, 0.0, 1.0) * 22.0 * self.settings.naturalization * self.settings.relief
        self.a += terraced
        return self

    def carve_path(
        self,
        points: Sequence[Tuple[float, float]],
        depth: float,
        half_width: float,
        bank: float,
        rim: float = 0.0,
    ) -> "TerrainBuilder":
        dist = polyline_distance(self.a.shape, points)
        inner = np.clip(dist / max(half_width, 1e-3), 0.0, 1.0)
        floor_profile = smoothstep01(inner)
        shoulder = smoothstep01(np.clip((dist - half_width) / max(bank, 1e-3), 0.0, 1.0))
        influence = np.where(dist <= half_width, 1.0 - 0.18 * floor_profile, 1.0 - shoulder)
        influence[dist >= half_width + bank] = 0.0
        self.a -= float(depth) * influence.astype(np.float32)
        if rim:
            sigma = max(bank * 0.35, 1.0)
            rim_profile = np.exp(-0.5 * ((dist - (half_width + bank * 0.35)) / sigma) ** 2)
            rim_profile[dist > half_width + bank * 1.5] = 0.0
            self.a += float(rim) * rim_profile.astype(np.float32)
        return self

    def add_ridge_path(
        self,
        points: Sequence[Tuple[float, float]],
        height: float,
        half_width: float,
        falloff: float,
    ) -> "TerrainBuilder":
        dist = polyline_distance(self.a.shape, points)
        x = np.clip((dist - half_width) / max(falloff, 1e-3), 0.0, 1.0)
        profile = 1.0 - smoothstep01(x)
        profile[dist > half_width + falloff] = 0.0
        self.a += float(height) * profile.astype(np.float32)
        return self

    def crater(
        self,
        cx: float,
        cy: float,
        radius: float,
        depth: float,
        rim_height: float,
        ellipse: float = 1.0,
    ) -> "TerrainBuilder":
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        ex = max(radius * ellipse, 1.0)
        ey = max(radius / max(ellipse, 1e-3), 1.0)
        r = np.sqrt(((xx - cx) / ex) ** 2 + ((yy - cy) / ey) ** 2)
        bowl = np.clip(1.0 - r, 0.0, 1.0)
        bowl = bowl * bowl * (3.0 - 2.0 * bowl)
        rim = np.exp(-0.5 * ((r - 1.02) / 0.12) ** 2)
        rim[r > 1.45] = 0.0
        self.a -= float(depth) * bowl
        self.a += float(rim_height) * rim
        return self

    def flatten_pad(
        self,
        cx: float,
        cy: float,
        radius_x: float,
        radius_y: float,
        target: Optional[float] = None,
        feather: float = 14.0,
        rectangular: bool = False,
    ) -> "TerrainBuilder":
        # Work only in the pad's finite feather footprint. Large authored city
        # maps can contain scores of building platforms; allocating full-map
        # coordinate grids for every pad made 8x8-zone layouts needlessly slow.
        margin_x = float(radius_x) + max(float(feather), 0.0) + 2.0
        margin_y = float(radius_y) + max(float(feather), 0.0) + 2.0
        minx = max(0, int(np.floor(float(cx) - margin_x)))
        maxx = min(self.w, int(np.ceil(float(cx) + margin_x + 1.0)))
        miny = max(0, int(np.floor(float(cy) - margin_y)))
        maxy = min(self.h, int(np.ceil(float(cy) + margin_y + 1.0)))
        if minx >= maxx or miny >= maxy:
            return self
        yy, xx = np.mgrid[miny:maxy, minx:maxx].astype(np.float32)
        if rectangular:
            dx = np.abs(xx - cx) - radius_x
            dy = np.abs(yy - cy) - radius_y
            outside = np.hypot(np.maximum(dx, 0), np.maximum(dy, 0))
            inside = np.minimum(np.maximum(dx, dy), 0)
            signed_distance = outside + inside
        else:
            signed_distance = (
                np.sqrt(((xx - cx) / max(radius_x, 1.0)) ** 2 + ((yy - cy) / max(radius_y, 1.0)) ** 2) - 1.0
            ) * min(radius_x, radius_y)
        weight = 1.0 - smoothstep01((signed_distance + feather) / max(feather * 2.0, 1e-3))
        area = self.a[miny:maxy, minx:maxx]
        if target is None:
            core = weight > 0.95
            target = float(np.median(area[core])) if np.any(core) else float(np.median(area))
        area[:] = area * (1.0 - weight) + float(target) * weight
        self.protected[miny:maxy, minx:maxx] |= weight >= 0.985
        return self

    def stamp_mask_level(self, mask: np.ndarray, target: float, feather: float = 12.0, protect_core: bool = True) -> "TerrainBuilder":
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.a.shape:
            raise ValueError("stamp mask shape mismatch")
        if feather <= 0:
            weight = mask.astype(np.float32)
        else:
            outside_dist = ndimage.distance_transform_edt(~mask).astype(np.float32)
            weight = np.where(mask, 1.0, 1.0 - smoothstep01(outside_dist / float(feather))).astype(np.float32)
        self.a = self.a * (1.0 - weight) + float(target) * weight
        if protect_core:
            self.protected |= mask
        return self

    def stamp_blob_shelf(
        self,
        target: float,
        area_fraction: float,
        feature_px: float,
        feather: float = 12.0,
        warp_px: float = 0.0,
        protect_core: bool = True,
    ) -> "TerrainBuilder":
        area_fraction = float(np.clip(area_fraction, 0.02, 0.98))
        field = signed_blob_field(self.a.shape, feature_px, self.rng, warp_px)
        threshold = float(np.quantile(field, 1.0 - area_fraction))
        return self.stamp_mask_level(field >= threshold, target, feather, protect_core)

    def carve_corridor_level(
        self,
        points: Sequence[Tuple[float, float]],
        floor_height: float,
        half_width: float,
        bank: float,
        rim_height: float = 0.0,
        protect_floor: bool = True,
    ) -> "TerrainBuilder":
        dist = polyline_distance(self.a.shape, points)
        half_width = max(float(half_width), 1.0)
        bank = max(float(bank), 1.0)
        floor = dist <= half_width
        transition = (dist > half_width) & (dist < half_width + bank)
        weight = np.zeros_like(self.a, dtype=np.float32)
        weight[floor] = 1.0
        weight[transition] = 1.0 - smoothstep01((dist[transition] - half_width) / bank)
        self.a = self.a * (1.0 - weight) + float(floor_height) * weight
        if protect_floor:
            self.protected |= floor
        if rim_height:
            sigma = max(bank * 0.22, 1.0)
            center = half_width + bank * 0.78
            rim = np.exp(-0.5 * ((dist - center) / sigma) ** 2)
            rim[dist > half_width + bank * 1.35] = 0.0
            self.a += float(rim_height) * rim.astype(np.float32)
        return self

    def carve_variable_corridor_level(
        self,
        points: Sequence[Tuple[float, float]],
        floor_height: float,
        half_widths: Sequence[float] | float,
        bank: float,
        rim_height: float = 0.0,
        edge_irregularity: float = 0.0,
        protect_floor: bool = True,
    ) -> "TerrainBuilder":
        floor = variable_corridor_mask(self.a.shape, points, half_widths)
        outside = ndimage.distance_transform_edt(~floor).astype(np.float32)
        bank = max(float(bank), 1.0)
        effective = outside
        if edge_irregularity > 0:
            edge_noise = fbm(self.a.shape, max(16.0, bank * 2.4), self.rng, octaves=3, persistence=0.52)
            effective = np.maximum(0.0, outside + edge_noise * float(edge_irregularity))
        weight = 1.0 - smoothstep01(effective / bank)
        weight[floor] = 1.0
        weight[outside >= bank * 1.15] = 0.0
        self.a = self.a * (1.0 - weight) + float(floor_height) * weight
        if protect_floor:
            self.protected |= floor
        if rim_height:
            sigma = max(bank * 0.20, 1.0)
            rim = np.exp(-0.5 * ((effective - bank * 0.82) / sigma) ** 2)
            rim[outside > bank * 1.35] = 0.0
            rim[floor] = 0.0
            self.a += float(rim_height) * rim.astype(np.float32)
        return self

    def add_boundary_rim(self, height: float, inner_margin: float, width: float, irregularity: float = 0.0) -> "TerrainBuilder":
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        distance = np.minimum.reduce([xx, yy, self.w - 1.0 - xx, self.h - 1.0 - yy])
        if irregularity > 0:
            distance += fbm(self.a.shape, max(24.0, min(self.h, self.w) * 0.12), self.rng, octaves=3) * float(irregularity)
        profile = 1.0 - smoothstep01(np.clip((distance - float(inner_margin)) / max(float(width), 1.0), 0.0, 1.0))
        profile[distance > inner_margin + width] = 0.0
        self.a += float(height) * profile.astype(np.float32)
        return self

    def add_random_craters(self, count: int, radius_range: Tuple[float, float], depth_scale: float, rim_scale: float) -> "TerrainBuilder":
        for _ in range(max(0, int(count))):
            radius = float(self.rng.uniform(*radius_range))
            cx = float(self.rng.uniform(radius * 1.3, max(radius * 1.3 + 1, self.w - radius * 1.3)))
            cy = float(self.rng.uniform(radius * 1.3, max(radius * 1.3 + 1, self.h - radius * 1.3)))
            self.crater(cx, cy, radius, radius * depth_scale, radius * rim_scale, float(self.rng.uniform(0.75, 1.35)))
        return self

    def add_detail(self, amplitude: float, feature_px: float) -> "TerrainBuilder":
        if amplitude <= 0:
            return self
        detail = fbm(self.a.shape, feature_px, self.rng, octaves=4, persistence=0.48) * float(amplitude)
        detail[self.protected] = 0.0
        self.a += detail
        return self

    def smooth(self, sigma: float, amount: float = 1.0) -> "TerrainBuilder":
        if sigma <= 0 or amount <= 0:
            return self
        original = self.a.copy()
        blurred = ndimage.gaussian_filter(self.a, sigma=float(sigma), mode="reflect")
        self.a = self.a * (1.0 - amount) + blurred * amount
        self.a[self.protected] = original[self.protected]
        return self

    def apply_symmetry(self, mode: str) -> "TerrainBuilder":
        mode = (mode or "None").lower()
        h, w = self.a.shape
        if mode == "mirror x":
            left = self.a[:, : (w + 1) // 2].copy()
            protected = self.protected[:, : (w + 1) // 2].copy()
            self.a[:, w // 2 :] = np.fliplr(left[:, : w - w // 2])
            self.protected[:, w // 2 :] = np.fliplr(protected[:, : w - w // 2])
        elif mode == "mirror z":
            top = self.a[: (h + 1) // 2, :].copy()
            protected = self.protected[: (h + 1) // 2, :].copy()
            self.a[h // 2 :, :] = np.flipud(top[: h - h // 2, :])
            self.protected[h // 2 :, :] = np.flipud(protected[: h - h // 2, :])
        elif mode == "2-way rotational":
            top = self.a[: (h + 1) // 2, :].copy()
            protected = self.protected[: (h + 1) // 2, :].copy()
            self.a[h // 2 :, :] = np.flipud(np.fliplr(top[: h - h // 2, :]))
            self.protected[h // 2 :, :] = np.flipud(np.fliplr(protected[: h - h // 2, :]))
        elif mode == "4-way":
            self.apply_symmetry("mirror x")
            self.apply_symmetry("mirror z")
        return self

    def finalize(self, center_height: float = 1800.0, preserve_flats: bool = True) -> HG2Map:
        relief = max(0.05, float(self.settings.relief))
        median = float(np.median(self.a))
        self.a = center_height + (self.a - median) * relief
        if self.settings.synthetic_pads > 0:
            for i in range(self.settings.synthetic_pads):
                angle = (i / max(self.settings.synthetic_pads, 1)) * math.tau + 0.35
                radius = min(self.h, self.w) * 0.24
                cx = self.w * 0.5 + math.cos(angle) * radius
                cy = self.h * 0.5 + math.sin(angle) * radius
                self.flatten_pad(cx, cy, 28, 22, feather=9, rectangular=True)
        self.apply_symmetry(self.settings.symmetry)
        if not preserve_flats:
            self.smooth(0.55, 0.35)
        self.a = np.clip(np.rint(self.a), 0, HG2_MAX_HEIGHT).astype(np.uint16)
        return HG2Map(self.a, self.settings.zones_x, self.settings.zones_z)
