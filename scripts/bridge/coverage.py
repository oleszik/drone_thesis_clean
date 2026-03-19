from __future__ import annotations

import numpy as np


class CoverageGridTracker:
    def __init__(
        self,
        width_m: float,
        height_m: float,
        cell_size_m: float,
        radius_m: float,
        boundary_cells: int = 2,
        track_pass_counts: bool = False,
    ) -> None:
        self.width_m = max(1e-3, float(width_m))
        self.height_m = max(1e-3, float(height_m))
        self.cell_size = max(1e-3, float(cell_size_m))
        self.radius = max(0.0, float(radius_m))
        self.x_min = -0.5 * self.width_m
        self.y_min = -0.5 * self.height_m
        self.nx = max(1, int(np.ceil(self.width_m / self.cell_size)))
        self.ny = max(1, int(np.ceil(self.height_m / self.cell_size)))
        self.covered = np.zeros((self.nx, self.ny), dtype=bool)
        self.pass_counts = np.zeros((self.nx, self.ny), dtype=np.int32) if bool(track_pass_counts) else None
        self.covered_cells = 0
        self.revisit_steps = 0
        ix = np.arange(self.nx, dtype=np.int32)[:, None]
        iy = np.arange(self.ny, dtype=np.int32)[None, :]
        edge = max(1, int(boundary_cells))
        self.boundary_mask = (ix < edge) | (ix >= (self.nx - edge)) | (iy < edge) | (iy >= (self.ny - edge))

    @property
    def total_cells(self) -> int:
        return int(max(1, self.nx * self.ny))

    def _mark_disk(self, x: float, y: float) -> tuple[int, int]:
        return self._mark_disk_radius(float(x), float(y), float(self.radius))

    def _mark_disk_radius(self, x: float, y: float, radius_m: float, count_hits: bool = False) -> tuple[int, int]:
        cs = self.cell_size
        r = max(0.0, float(radius_m))
        r2 = r * r
        ix0 = int(np.floor((x - r - self.x_min) / cs))
        ix1 = int(np.floor((x + r - self.x_min) / cs))
        iy0 = int(np.floor((y - r - self.y_min) / cs))
        iy1 = int(np.floor((y + r - self.y_min) / cs))
        ix0 = int(np.clip(ix0, 0, self.nx - 1))
        ix1 = int(np.clip(ix1, 0, self.nx - 1))
        iy0 = int(np.clip(iy0, 0, self.ny - 1))
        iy1 = int(np.clip(iy1, 0, self.ny - 1))
        new_hits = 0
        revisit_hits = 0
        for ix in range(ix0, ix1 + 1):
            cx = self.x_min + (float(ix) + 0.5) * cs
            dx = cx - x
            for iy in range(iy0, iy1 + 1):
                cy = self.y_min + (float(iy) + 0.5) * cs
                dy = cy - y
                if (dx * dx + dy * dy) > r2:
                    continue
                if bool(count_hits) and (self.pass_counts is not None):
                    self.pass_counts[ix, iy] = int(self.pass_counts[ix, iy] + 1)
                if bool(self.covered[ix, iy]):
                    revisit_hits += 1
                else:
                    self.covered[ix, iy] = True
                    new_hits += 1
        return int(new_hits), int(revisit_hits)

    def _mark_ellipse(self, x: float, y: float, rx_m: float, ry_m: float, count_hits: bool = False) -> tuple[int, int]:
        cs = self.cell_size
        rx = max(1e-6, float(rx_m))
        ry = max(1e-6, float(ry_m))
        ix0 = int(np.floor((x - rx - self.x_min) / cs))
        ix1 = int(np.floor((x + rx - self.x_min) / cs))
        iy0 = int(np.floor((y - ry - self.y_min) / cs))
        iy1 = int(np.floor((y + ry - self.y_min) / cs))
        ix0 = int(np.clip(ix0, 0, self.nx - 1))
        ix1 = int(np.clip(ix1, 0, self.nx - 1))
        iy0 = int(np.clip(iy0, 0, self.ny - 1))
        iy1 = int(np.clip(iy1, 0, self.ny - 1))
        new_hits = 0
        revisit_hits = 0
        inv_rx2 = 1.0 / max(1e-9, rx * rx)
        inv_ry2 = 1.0 / max(1e-9, ry * ry)
        for ix in range(ix0, ix1 + 1):
            cx = self.x_min + (float(ix) + 0.5) * cs
            dx = cx - x
            for iy in range(iy0, iy1 + 1):
                cy = self.y_min + (float(iy) + 0.5) * cs
                dy = cy - y
                if ((dx * dx) * inv_rx2 + (dy * dy) * inv_ry2) > 1.0:
                    continue
                if bool(count_hits) and (self.pass_counts is not None):
                    self.pass_counts[ix, iy] = int(self.pass_counts[ix, iy] + 1)
                if bool(self.covered[ix, iy]):
                    revisit_hits += 1
                else:
                    self.covered[ix, iy] = True
                    new_hits += 1
        return int(new_hits), int(revisit_hits)

    def update(self, x: float, y: float) -> tuple[int, int]:
        new_hits, revisit_hits = self._mark_disk(float(x), float(y))
        if new_hits > 0:
            self.covered_cells += int(new_hits)
        elif revisit_hits > 0:
            self.revisit_steps += 1
        return int(new_hits), int(revisit_hits)

    def update_disk_radius(self, x: float, y: float, radius_m: float, count_hits: bool = False) -> tuple[int, int]:
        new_hits, revisit_hits = self._mark_disk_radius(float(x), float(y), float(radius_m), count_hits=bool(count_hits))
        if new_hits > 0:
            self.covered_cells += int(new_hits)
        elif revisit_hits > 0:
            self.revisit_steps += 1
        return int(new_hits), int(revisit_hits)

    def update_ellipse(self, x: float, y: float, rx_m: float, ry_m: float, count_hits: bool = False) -> tuple[int, int]:
        new_hits, revisit_hits = self._mark_ellipse(float(x), float(y), float(rx_m), float(ry_m), count_hits=bool(count_hits))
        if new_hits > 0:
            self.covered_cells += int(new_hits)
        elif revisit_hits > 0:
            self.revisit_steps += 1
        return int(new_hits), int(revisit_hits)

    def coverage_frac(self) -> float:
        return float(self.covered_cells / max(1, self.total_cells))

    def coverage_frac_at_least(self, min_passes: int) -> float:
        k = max(1, int(min_passes))
        if self.pass_counts is None:
            if k <= 1:
                return self.coverage_frac()
            return 0.0
        return float(np.count_nonzero(self.pass_counts >= k) / max(1, self.total_cells))

    def boundary_covered_frac(self) -> float:
        total = int(np.count_nonzero(self.boundary_mask))
        if total <= 0:
            return 0.0
        return float(np.count_nonzero(self.covered & self.boundary_mask) / max(1, total))

    def interior_covered_frac(self) -> float:
        mask = ~self.boundary_mask
        total = int(np.count_nonzero(mask))
        if total <= 0:
            return 0.0
        return float(np.count_nonzero(self.covered & mask) / max(1, total))

    def local_patch(self, x: float, y: float, patch_size: int) -> np.ndarray:
        p = max(1, int(patch_size))
        if p % 2 == 0:
            p += 1
        half = p // 2
        ix_center = int(np.floor((x - self.x_min) / self.cell_size))
        iy_center = int(np.floor((y - self.y_min) / self.cell_size))
        patch = np.ones((p, p), dtype=np.float32)
        for dx in range(-half, half + 1):
            gx = ix_center + dx
            if gx < 0 or gx >= self.nx:
                continue
            for dy in range(-half, half + 1):
                gy = iy_center + dy
                if gy < 0 or gy >= self.ny:
                    continue
                patch[dx + half, dy + half] = 1.0 if bool(self.covered[gx, gy]) else 0.0
        return patch.reshape(-1)

    def boundary_features(self, x: float, y: float) -> tuple[float, float]:
        x_max = self.x_min + float(self.nx) * self.cell_size
        y_max = self.y_min + float(self.ny) * self.cell_size
        d_x = min(float(x) - self.x_min, x_max - float(x))
        d_y = min(float(y) - self.y_min, y_max - float(y))
        span_x = max(1e-6, x_max - self.x_min)
        span_y = max(1e-6, y_max - self.y_min)
        return float(np.clip(d_x / span_x, 0.0, 1.0)), float(np.clip(d_y / span_y, 0.0, 1.0))
