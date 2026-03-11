from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any


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


def _xy_to_ll(origin_lng: float, origin_lat: float, x_m: float, y_m: float) -> tuple[float, float]:
    lat = float(origin_lat) + _meters_to_deg_lat(y_m)
    lng = float(origin_lng) + _meters_to_deg_lng(x_m, origin_lat)
    return lng, lat


def _polygon_area_m2(poly_xy: list[tuple[float, float]]) -> float:
    if len(poly_xy) < 3:
        return 0.0
    a = 0.0
    for i in range(len(poly_xy)):
        x1, y1 = poly_xy[i]
        x2, y2 = poly_xy[(i + 1) % len(poly_xy)]
        a += (x1 * y2) - (x2 * y1)
    return abs(a) * 0.5


def _scanline_segments(poly_xy: list[tuple[float, float]], y: float) -> list[tuple[float, float]]:
    xs: list[float] = []
    n = len(poly_xy)
    for i in range(n):
        x1, y1 = poly_xy[i]
        x2, y2 = poly_xy[(i + 1) % n]
        if y1 == y2:
            continue
        ymin = min(y1, y2)
        ymax = max(y1, y2)
        if y < ymin or y >= ymax:
            continue
        t = (y - y1) / (y2 - y1)
        x = x1 + t * (x2 - x1)
        xs.append(x)
    xs.sort()
    segs: list[tuple[float, float]] = []
    for i in range(0, len(xs) - 1, 2):
        xa = xs[i]
        xb = xs[i + 1]
        if xb > xa:
            segs.append((xa, xb))
    return segs


@dataclass(frozen=True)
class MissionPreview:
    expected_coverage_pct: float
    estimated_time_s: float
    number_of_passes: int
    path_length_m: float


class MissionService:
    def __init__(self, footprint_radius_m: float = 6.0) -> None:
        self._lock = threading.Lock()
        self._area_polygon: list[list[float]] = []
        self._start_position: list[float] | None = None
        self._path_waypoints: list[list[float]] = []
        self._preview = MissionPreview(0.0, 0.0, 0, 0.0)
        self._spacing_m = 8.0
        self._speed_m_s = 3.0
        self._footprint_radius_m = max(0.5, float(footprint_radius_m))

        self._sim_running = False
        self._sim_paused = False
        self._sim_done = False
        self._sim_index = 0
        self._sim_pos: list[float] | None = None
        self._sim_yaw_deg: float | None = None

    def _has_path_locked(self) -> bool:
        return len(self._path_waypoints) >= 2

    def _sim_state_locked(self) -> dict[str, Any]:
        pose = None
        if self._sim_pos is not None:
            pose = {
                "lng": float(self._sim_pos[0]),
                "lat": float(self._sim_pos[1]),
                "yaw_deg": self._sim_yaw_deg,
                "simulated": True,
            }
        return {
            "sim_running": bool(self._sim_running),
            "sim_paused": bool(self._sim_paused),
            "sim_done": bool(self._sim_done),
            "pose": pose,
        }

    def set_area(self, polygon_lng_lat: list[list[float]]) -> dict[str, Any]:
        cleaned: list[list[float]] = []
        for p in polygon_lng_lat:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            lng = float(p[0])
            lat = float(p[1])
            if not (math.isfinite(lng) and math.isfinite(lat)):
                continue
            cleaned.append([lng, lat])
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
            cleaned = cleaned[:-1]
        if len(cleaned) < 3:
            raise ValueError("scan area polygon requires at least 3 points")
        with self._lock:
            self._area_polygon = cleaned
            self._path_waypoints = []
            self._preview = MissionPreview(0.0, 0.0, 0, 0.0)
            self._sim_running = False
            self._sim_paused = False
            self._sim_done = False
            self._sim_index = 0
            self._sim_pos = None
            self._sim_yaw_deg = None
        return {"ok": True, "points": len(cleaned)}

    def set_start_position(self, lng: float, lat: float) -> dict[str, Any]:
        ll = [float(lng), float(lat)]
        with self._lock:
            self._start_position = ll
            if not self._sim_running:
                self._sim_pos = list(ll)
                self._sim_yaw_deg = None
                self._sim_done = False
        return {"ok": True, "start_position": ll}

    def clear(self) -> dict[str, Any]:
        with self._lock:
            self._area_polygon = []
            self._start_position = None
            self._path_waypoints = []
            self._preview = MissionPreview(0.0, 0.0, 0, 0.0)
            self._sim_running = False
            self._sim_paused = False
            self._sim_done = False
            self._sim_index = 0
            self._sim_pos = None
            self._sim_yaw_deg = None
        return {"ok": True}

    def generate_scan(self, spacing_m: float = 8.0, speed_m_s: float = 3.0, start_scan: bool = False) -> dict[str, Any]:
        spacing = max(1.0, float(spacing_m))
        speed = max(0.2, float(speed_m_s))
        with self._lock:
            area = list(self._area_polygon)
            start = list(self._start_position) if self._start_position else None
        if len(area) < 3:
            raise ValueError("scan area is not set")
        if start is None:
            raise ValueError("start position is not set")

        origin_lng = float(area[0][0])
        origin_lat = float(area[0][1])
        poly_xy = [_ll_to_xy_m(origin_lng, origin_lat, p[0], p[1]) for p in area]
        ys = [p[1] for p in poly_xy]
        min_y = min(ys)
        max_y = max(ys)

        y = min_y
        row = 0
        pass_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
        while y <= max_y + 1e-6:
            segs = _scanline_segments(poly_xy, y)
            for xa, xb in segs:
                if row % 2 == 0:
                    a = (xa, y)
                    b = (xb, y)
                else:
                    a = (xb, y)
                    b = (xa, y)
                pass_segments.append((a, b))
                row += 1
            y += spacing

        if not pass_segments:
            raise ValueError("scan generator produced no passes for area")

        path_xy: list[tuple[float, float]] = []
        for idx, seg in enumerate(pass_segments):
            a, b = seg
            if idx == 0:
                path_xy.append(a)
                path_xy.append(b)
                continue
            prev = path_xy[-1]
            if (prev[0] != a[0]) or (prev[1] != a[1]):
                path_xy.append(a)
            path_xy.append(b)

        path_lng_lat: list[list[float]] = []
        for p in path_xy:
            lng, lat = _xy_to_ll(origin_lng, origin_lat, p[0], p[1])
            path_lng_lat.append([lng, lat])
        path_lng_lat.insert(0, list(start))

        path_len = 0.0
        for i in range(1, len(path_xy)):
            dx = path_xy[i][0] - path_xy[i - 1][0]
            dy = path_xy[i][1] - path_xy[i - 1][1]
            path_len += math.hypot(dx, dy)
        start_xy = _ll_to_xy_m(origin_lng, origin_lat, start[0], start[1])
        first_xy = path_xy[0]
        path_len += math.hypot(first_xy[0] - start_xy[0], first_xy[1] - start_xy[1])

        area_m2 = _polygon_area_m2(poly_xy)
        sweep_width = self._footprint_radius_m * 2.0
        expected_cov = 0.0
        if area_m2 > 0 and spacing > 0:
            expected_cov = min(100.0, 100.0 * min(1.0, sweep_width / spacing))

        preview = MissionPreview(
            expected_coverage_pct=expected_cov,
            estimated_time_s=(path_len / speed) if speed > 0 else 0.0,
            number_of_passes=len(pass_segments),
            path_length_m=path_len,
        )

        with self._lock:
            self._path_waypoints = path_lng_lat
            self._preview = preview
            self._spacing_m = spacing
            self._speed_m_s = speed
            self._sim_index = 0
            self._sim_pos = list(start)
            self._sim_yaw_deg = None
            self._sim_done = False
            self._sim_running = bool(start_scan and self._has_path_locked())
            self._sim_paused = False

        return self.get_path()

    def sim_start(self) -> dict[str, Any]:
        with self._lock:
            if not self._has_path_locked():
                raise ValueError("scan path is not generated")
            if self._sim_pos is None:
                if self._start_position is None:
                    raise ValueError("start position is not set")
                self._sim_pos = list(self._start_position)
            self._sim_running = True
            self._sim_paused = False
            if self._sim_done:
                self._sim_done = False
        return self.get_sim_state()

    def sim_pause(self) -> dict[str, Any]:
        with self._lock:
            if self._sim_running:
                self._sim_paused = True
        return self.get_sim_state()

    def sim_stop(self) -> dict[str, Any]:
        with self._lock:
            self._sim_running = False
            self._sim_paused = False
            self._sim_done = False
            self._sim_index = 0
            self._sim_yaw_deg = None
            if self._start_position is not None:
                self._sim_pos = list(self._start_position)
            elif self._path_waypoints:
                self._sim_pos = list(self._path_waypoints[0])
            else:
                self._sim_pos = None
        return self.get_sim_state()

    def step(self, dt: float) -> dict[str, Any]:
        with self._lock:
            if not self._sim_running or self._sim_paused or self._sim_done:
                return self._sim_state_locked()
            if not self._has_path_locked():
                return self._sim_state_locked()
            if self._sim_pos is None:
                self._sim_pos = list(self._path_waypoints[0])
                self._sim_index = 0

            remaining = max(0.0, float(dt)) * self._speed_m_s
            while remaining > 0 and self._sim_running and not self._sim_paused and not self._sim_done:
                if self._sim_index >= len(self._path_waypoints) - 1:
                    self._sim_running = False
                    self._sim_done = True
                    break
                a = self._sim_pos
                b = self._path_waypoints[self._sim_index + 1]
                seg_dx, seg_dy = _ll_to_xy_m(a[0], a[1], b[0], b[1])
                seg_len = math.hypot(seg_dx, seg_dy)
                if seg_len < 0.05:
                    self._sim_index += 1
                    self._sim_pos = list(b)
                    continue
                travel = min(seg_len, remaining)
                frac = travel / seg_len
                self._sim_pos = [a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac]
                remaining -= travel

                heading_rad = math.atan2(seg_dx, seg_dy)
                self._sim_yaw_deg = math.degrees(heading_rad)

                if travel >= seg_len - 1e-6:
                    self._sim_index += 1
                    self._sim_pos = list(b)
                    if self._sim_index >= len(self._path_waypoints) - 1:
                        self._sim_running = False
                        self._sim_done = True
                        break

            return self._sim_state_locked()

    def get_sim_state(self) -> dict[str, Any]:
        with self._lock:
            return self._sim_state_locked()

    def get_sim_vehicle(self) -> dict[str, Any] | None:
        state = self.get_sim_state()
        pose = state.get("pose")
        if pose is None:
            return None
        return {
            "lng": float(pose["lng"]),
            "lat": float(pose["lat"]),
            "yaw_deg": pose.get("yaw_deg"),
            "simulated": True,
            "active": bool(state.get("sim_running")) and not bool(state.get("sim_paused")),
            "sim_running": bool(state.get("sim_running")),
            "sim_paused": bool(state.get("sim_paused")),
            "sim_done": bool(state.get("sim_done")),
        }

    def get_path(self) -> dict[str, Any]:
        with self._lock:
            sim_state = self._sim_state_locked()
            sim = sim_state.get("pose")
            if sim is not None:
                sim = {
                    **sim,
                    "active": bool(sim_state["sim_running"]) and not bool(sim_state["sim_paused"]),
                    "sim_running": bool(sim_state["sim_running"]),
                    "sim_paused": bool(sim_state["sim_paused"]),
                    "sim_done": bool(sim_state["sim_done"]),
                }
            return {
                "scan_area_polygon_lng_lat": list(self._area_polygon),
                "start_position_lng_lat": list(self._start_position) if self._start_position else None,
                "waypoints_lng_lat": list(self._path_waypoints),
                "config": {
                    "spacing_m": self._spacing_m,
                    "speed_m_s": self._speed_m_s,
                },
                "coverage_preview": {
                    "expected_coverage_pct": self._preview.expected_coverage_pct,
                    "estimated_time_s": self._preview.estimated_time_s,
                    "number_of_passes": self._preview.number_of_passes,
                    "path_length_m": self._preview.path_length_m,
                },
                "sim": sim,
                "sim_running": bool(sim_state["sim_running"]),
                "sim_paused": bool(sim_state["sim_paused"]),
                "sim_done": bool(sim_state["sim_done"]),
            }
