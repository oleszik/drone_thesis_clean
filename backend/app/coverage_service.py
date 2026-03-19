from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CoverageConfig:
    origin_lng: float
    origin_lat: float
    bounds_w_m: float
    bounds_h_m: float
    cell_size_m: float
    footprint_radius_m: float


def _meters_to_deg_lat(m: float) -> float:
    return float(m) / 111320.0


def _meters_to_deg_lng(m: float, lat_deg: float) -> float:
    c = math.cos(math.radians(float(lat_deg)))
    return float(m) / (111320.0 * max(0.1, abs(c)))


def _ll_to_xy_m(origin_lng: float, origin_lat: float, lng: float, lat: float) -> tuple[float, float]:
    d_lat = float(lat) - float(origin_lat)
    d_lng = float(lng) - float(origin_lng)
    y = d_lat * 111320.0
    x = d_lng * 111320.0 * max(0.1, abs(math.cos(math.radians(float(origin_lat)))))
    return x, y


def _point_in_polygon_xy(x: float, y: float, poly_xy: list[tuple[float, float]]) -> bool:
    n = len(poly_xy)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly_xy[i]
        xj, yj = poly_xy[j]
        cross = ((yi > y) != (yj > y))
        if cross:
            denom = (yj - yi) if (yj - yi) != 0 else 1e-12
            x_at_y = (xj - xi) * (y - yi) / denom + xi
            if x < x_at_y:
                inside = not inside
        j = i
    return inside


class CoverageService:
    def __init__(self, cfg: CoverageConfig) -> None:
        self._lock = threading.Lock()
        self._cfg = cfg
        self._rows = max(1, int(math.ceil(cfg.bounds_h_m / max(0.5, cfg.cell_size_m))))
        self._cols = max(1, int(math.ceil(cfg.bounds_w_m / max(0.5, cfg.cell_size_m))))
        self._grid: list[list[int]] = [[0 for _ in range(self._cols)] for _ in range(self._rows)]
        self._covered_cells = 0
        self._total_hits = 0
        self._revisit_hits = 0
        self._start_ts = time.time()
        self._elapsed_active_s = 0.0
        self._last_footprint: list[list[float]] = []
        self._last_pose_lng_lat: list[float] | None = None
        self._last_update_t_unix: float | None = None
        self._roi_polygon_lng_lat: list[list[float]] = []
        self._active_mask: list[list[bool]] = [[True for _ in range(self._cols)] for _ in range(self._rows)]
        self._active_total_cells = self._rows * self._cols
        self._build_active_mask_locked()

    def _rebuild_grid(self) -> None:
        self._rows = max(1, int(math.ceil(self._cfg.bounds_h_m / max(0.5, self._cfg.cell_size_m))))
        self._cols = max(1, int(math.ceil(self._cfg.bounds_w_m / max(0.5, self._cfg.cell_size_m))))
        self._grid = [[0 for _ in range(self._cols)] for _ in range(self._rows)]
        self._active_mask = [[True for _ in range(self._cols)] for _ in range(self._rows)]
        self._active_total_cells = self._rows * self._cols

    def _build_active_mask_locked(self) -> None:
        if len(self._roi_polygon_lng_lat) < 3:
            self._active_mask = [[True for _ in range(self._cols)] for _ in range(self._rows)]
            self._active_total_cells = self._rows * self._cols
            return

        poly_xy = [
            _ll_to_xy_m(self._cfg.origin_lng, self._cfg.origin_lat, float(p[0]), float(p[1]))
            for p in self._roi_polygon_lng_lat
            if isinstance(p, (list, tuple)) and len(p) >= 2
        ]
        if len(poly_xy) < 3:
            self._active_mask = [[True for _ in range(self._cols)] for _ in range(self._rows)]
            self._active_total_cells = self._rows * self._cols
            return

        cell = max(0.5, self._cfg.cell_size_m)
        half_w = self._cfg.bounds_w_m * 0.5
        half_h = self._cfg.bounds_h_m * 0.5
        mask: list[list[bool]] = []
        active = 0
        for rr in range(self._rows):
            row_mask: list[bool] = []
            cy = half_h - (rr + 0.5) * cell
            for cc in range(self._cols):
                cx = (cc + 0.5) * cell - half_w
                ok = _point_in_polygon_xy(cx, cy, poly_xy)
                row_mask.append(ok)
                if ok:
                    active += 1
            mask.append(row_mask)
        if active <= 0:
            self._active_mask = [[True for _ in range(self._cols)] for _ in range(self._rows)]
            self._active_total_cells = self._rows * self._cols
            return
        self._active_mask = mask
        self._active_total_cells = active

    def reset(
        self,
        origin_lng: float | None = None,
        origin_lat: float | None = None,
        bounds_w_m: float | None = None,
        bounds_h_m: float | None = None,
        cell_size_m: float | None = None,
        roi_polygon_lng_lat: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if (
                origin_lng is not None
                or origin_lat is not None
                or bounds_w_m is not None
                or bounds_h_m is not None
                or cell_size_m is not None
                or roi_polygon_lng_lat is not None
            ):
                if roi_polygon_lng_lat is not None:
                    self._roi_polygon_lng_lat = [list(p[:2]) for p in roi_polygon_lng_lat if isinstance(p, (list, tuple)) and len(p) >= 2]
                self._cfg = CoverageConfig(
                    origin_lng=float(origin_lng) if origin_lng is not None else self._cfg.origin_lng,
                    origin_lat=float(origin_lat) if origin_lat is not None else self._cfg.origin_lat,
                    bounds_w_m=float(bounds_w_m) if bounds_w_m is not None else self._cfg.bounds_w_m,
                    bounds_h_m=float(bounds_h_m) if bounds_h_m is not None else self._cfg.bounds_h_m,
                    cell_size_m=max(0.5, float(cell_size_m)) if cell_size_m is not None else self._cfg.cell_size_m,
                    footprint_radius_m=self._cfg.footprint_radius_m,
                )
                self._rebuild_grid()
                self._build_active_mask_locked()
            else:
                self._grid = [[0 for _ in range(self._cols)] for _ in range(self._rows)]
            self._covered_cells = 0
            self._total_hits = 0
            self._revisit_hits = 0
            self._start_ts = time.time()
            self._elapsed_active_s = 0.0
            self._last_footprint = []
            self._last_pose_lng_lat = None
            self._last_update_t_unix = None
        return {"ok": True}

    def update_from_point(self, lng: float, lat: float, t_unix: float | None = None) -> None:
        with self._lock:
            ts = float(t_unix) if t_unix is not None else time.time()
            if self._last_update_t_unix is not None:
                dt = max(0.0, ts - float(self._last_update_t_unix))
                # Guard against clock jumps and long pauses.
                self._elapsed_active_s += min(dt, 2.0)
            self._last_pose_lng_lat = [float(lng), float(lat)]
            self._last_update_t_unix = ts
            x, y = _ll_to_xy_m(self._cfg.origin_lng, self._cfg.origin_lat, lng, lat)
            half_w = self._cfg.bounds_w_m * 0.5
            half_h = self._cfg.bounds_h_m * 0.5

            # Build footprint polygon (square approximation for debug display).
            r = self._cfg.footprint_radius_m
            d_lng = _meters_to_deg_lng(r, lat)
            d_lat = _meters_to_deg_lat(r)
            self._last_footprint = [
                [lng - d_lng, lat - d_lat],
                [lng + d_lng, lat - d_lat],
                [lng + d_lng, lat + d_lat],
                [lng - d_lng, lat + d_lat],
                [lng - d_lng, lat - d_lat],
            ]

            # Ignore points far outside configured bounds.
            if x < -half_w - r or x > half_w + r or y < -half_h - r or y > half_h + r:
                return

            cell = max(0.5, self._cfg.cell_size_m)
            min_c = int(math.floor((x - r + half_w) / cell))
            max_c = int(math.floor((x + r + half_w) / cell))
            min_r = int(math.floor((half_h - (y + r)) / cell))
            max_r = int(math.floor((half_h - (y - r)) / cell))

            min_c = max(0, min(self._cols - 1, min_c))
            max_c = max(0, min(self._cols - 1, max_c))
            min_r = max(0, min(self._rows - 1, min_r))
            max_r = max(0, min(self._rows - 1, max_r))

            r2 = r * r
            for rr in range(min_r, max_r + 1):
                cy = half_h - (rr + 0.5) * cell
                for cc in range(min_c, max_c + 1):
                    cx = (cc + 0.5) * cell - half_w
                    if not self._active_mask[rr][cc]:
                        continue
                    if (cx - x) * (cx - x) + (cy - y) * (cy - y) > r2:
                        continue
                    prev = self._grid[rr][cc]
                    self._grid[rr][cc] = prev + 1
                    self._total_hits += 1
                    if prev == 0:
                        self._covered_cells += 1
                    else:
                        self._revisit_hits += 1

    def get_coverage(self) -> dict[str, Any]:
        with self._lock:
            total_cells = self._active_total_cells
            coverage_pct = (100.0 * self._covered_cells / total_cells) if total_cells > 0 else 0.0
            overlap_pct = (100.0 * self._revisit_hits / self._total_hits) if self._total_hits > 0 else 0.0
            elapsed = max(0.0, self._elapsed_active_s)
            cell = max(0.5, self._cfg.cell_size_m)
            half_w = self._cfg.bounds_w_m * 0.5
            half_h = self._cfg.bounds_h_m * 0.5
            covered_cells: list[dict[str, Any]] = []
            for rr in range(self._rows):
                for cc in range(self._cols):
                    cnt = self._grid[rr][cc]
                    if cnt <= 0 or not self._active_mask[rr][cc]:
                        continue
                    x0 = cc * cell - half_w
                    x1 = (cc + 1) * cell - half_w
                    y1 = half_h - rr * cell
                    y0 = half_h - (rr + 1) * cell
                    lat0 = self._cfg.origin_lat + _meters_to_deg_lat(y0)
                    lat1 = self._cfg.origin_lat + _meters_to_deg_lat(y1)
                    lng0 = self._cfg.origin_lng + _meters_to_deg_lng(x0, self._cfg.origin_lat)
                    lng1 = self._cfg.origin_lng + _meters_to_deg_lng(x1, self._cfg.origin_lat)
                    covered_cells.append(
                        {
                            "r": rr,
                            "c": cc,
                            "count": cnt,
                            "lat_min": min(lat0, lat1),
                            "lat_max": max(lat0, lat1),
                            "lng_min": min(lng0, lng1),
                            "lng_max": max(lng0, lng1),
                        }
                    )
            return {
                "origin": {"lng": self._cfg.origin_lng, "lat": self._cfg.origin_lat},
                "rows": self._rows,
                "cols": self._cols,
                "active_total_cells": self._active_total_cells,
                "cell_size_m": cell,
                "footprint_radius_m": self._cfg.footprint_radius_m,
                "covered_cells": covered_cells,
                "stats": {
                    "coverage_pct": coverage_pct,
                    "overlap_pct": overlap_pct,
                    "covered_cells": self._covered_cells,
                    "total_cells": total_cells,
                    "total_hits": self._total_hits,
                    "revisit_hits": self._revisit_hits,
                    "time_elapsed_s": elapsed,
                },
            }

    def get_scan_debug(self) -> dict[str, Any]:
        with self._lock:
            return {
                "footprint_polygon_lng_lat": list(self._last_footprint),
                "origin": {"lng": self._cfg.origin_lng, "lat": self._cfg.origin_lat},
                "footprint_radius_m": self._cfg.footprint_radius_m,
                "last_pose_lng_lat": list(self._last_pose_lng_lat) if self._last_pose_lng_lat else None,
                "last_update_t_unix": self._last_update_t_unix,
            }
