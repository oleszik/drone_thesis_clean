from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any

from .fence_service import point_inside_polygon


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


def _ll_distance_m(a_lng: float, a_lat: float, b_lng: float, b_lat: float) -> float:
    dx, dy = _ll_to_xy_m(a_lng, a_lat, b_lng, b_lat)
    return math.hypot(dx, dy)


def _path_length_lng_lat_m(path_lng_lat: list[list[float]]) -> float:
    total = 0.0
    for idx in range(1, len(path_lng_lat)):
        a = path_lng_lat[idx - 1]
        b = path_lng_lat[idx]
        total += _ll_distance_m(float(a[0]), float(a[1]), float(b[0]), float(b[1]))
    return total


def _rotate_xy(x_m: float, y_m: float, angle_rad: float) -> tuple[float, float]:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return (x_m * c) - (y_m * s), (x_m * s) + (y_m * c)


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


def _normalize_scan_angle(angle_rad: float) -> float:
    return float(angle_rad) % math.pi


def _build_scan_path_xy(
    poly_xy: list[tuple[float, float]],
    start_xy: tuple[float, float],
    spacing: float,
    footprint_radius: float,
) -> tuple[list[tuple[float, float]], float, int, float, float]:
    ys = [p[1] for p in poly_xy]
    min_y = min(ys)
    max_y = max(ys)
    span_y = max(0.0, max_y - min_y)
    radius = max(0.1, float(footprint_radius))

    row_ys: list[float]
    if span_y <= (2.0 * radius):
        row_ys = [0.5 * (min_y + max_y)]
    else:
        interior_span = max(0.0, span_y - (2.0 * radius))
        row_count = max(2, int(math.ceil(interior_span / max(0.1, spacing))) + 1)
        actual_gap = interior_span / max(1, row_count - 1)
        row_ys = [min_y + radius + (idx * actual_gap) for idx in range(row_count)]

    base_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for y in row_ys:
        segs = _scanline_segments(poly_xy, y)
        for xa, xb in segs:
            base_segments.append(((xa, y), (xb, y)))

    if not base_segments:
        raise ValueError("scan generator produced no passes for area")

    best_path_xy: list[tuple[float, float]] | None = None
    best_total_len = float("inf")
    best_lead_in = 0.0
    best_return = 0.0

    for reverse_order in (False, True):
        ordered_segments = list(reversed(base_segments)) if reverse_order else list(base_segments)
        for first_forward in (False, True):
            path_xy: list[tuple[float, float]] = []
            for idx, seg in enumerate(ordered_segments):
                left, right = seg
                forward = first_forward if idx % 2 == 0 else not first_forward
                a, b = (left, right) if forward else (right, left)
                if idx == 0:
                    path_xy.append(a)
                    path_xy.append(b)
                    continue
                prev = path_xy[-1]
                if (prev[0] != a[0]) or (prev[1] != a[1]):
                    path_xy.append(a)
                path_xy.append(b)

            internal_len = 0.0
            for idx in range(1, len(path_xy)):
                dx = path_xy[idx][0] - path_xy[idx - 1][0]
                dy = path_xy[idx][1] - path_xy[idx - 1][1]
                internal_len += math.hypot(dx, dy)

            lead_in_m = math.hypot(path_xy[0][0] - start_xy[0], path_xy[0][1] - start_xy[1])
            return_to_home_m = math.hypot(path_xy[-1][0] - start_xy[0], path_xy[-1][1] - start_xy[1])
            total_len = internal_len + lead_in_m + return_to_home_m

            if total_len < (best_total_len - 1e-6):
                best_path_xy = path_xy
                best_total_len = total_len
                best_lead_in = lead_in_m
                best_return = return_to_home_m

    if best_path_xy is None:
        raise ValueError("scan generator produced no path variants for area")

    return best_path_xy, best_total_len, len(base_segments), best_lead_in, best_return


def _yaw_deg_from_vector(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(dx, dy))


@dataclass(frozen=True)
class MissionPreview:
    expected_coverage_pct: float
    estimated_time_s: float
    number_of_passes: int
    path_length_m: float
    sweep_angle_deg: float
    lead_in_m: float
    return_to_home_m: float
    overlap_pct_est: float


@dataclass(frozen=True)
class ScanPlan:
    path_xy: list[tuple[float, float]]
    path_length_m: float
    pass_count: int
    sweep_angle_deg: float
    lead_in_m: float
    return_to_home_m: float
    spacing_m: float
    expected_coverage_pct: float
    overlap_pct_est: float


@dataclass(frozen=True)
class OrbitPlan:
    waypoints_lng_lat: list[list[float]]
    waypoint_meta: list[dict[str, Any]]
    path_length_m: float
    radius_m: float
    total_laps: int
    points_per_lap: int
    clockwise: bool
    yaw_to_center: bool
    layers: list[dict[str, Any]]
    first_altitude_m: float
    max_altitude_m: float


class MissionService:
    TINY_ALT_MIN_M = 1.0
    TINY_ALT_MAX_M = 6.0
    TINY_FORWARD_MIN_M = 0.5
    TINY_FORWARD_MAX_M = 6.0
    TINY_HOVER_MIN_S = 2.0
    TINY_HOVER_MAX_S = 20.0
    TINY_SPEED_MIN_M_S = 0.2
    TINY_SPEED_MAX_M_S = 2.0

    def __init__(self, footprint_radius_m: float = 6.0) -> None:
        self._lock = threading.Lock()
        self._mission_type = "ground_scan"
        self._area_polygon: list[list[float]] = []
        self._orbit_center: list[float] | None = None
        self._start_position: list[float] | None = None
        self._landing_position: list[float] | None = None
        self._path_waypoints: list[list[float]] = []
        self._waypoint_meta: list[dict[str, Any]] = []
        self._orbit_meta: dict[str, Any] = {}
        self._preview = MissionPreview(0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
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
            self._mission_type = "ground_scan"
            self._area_polygon = cleaned
            self._orbit_center = None
            self._path_waypoints = []
            self._waypoint_meta = []
            self._orbit_meta = {}
            self._preview = MissionPreview(0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
            self._sim_running = False
            self._sim_paused = False
            self._sim_done = False
            self._sim_index = 0
            self._sim_pos = None
            self._sim_yaw_deg = None
        return {"ok": True, "points": len(cleaned)}

    def set_reference_position(self, lng: float, lat: float) -> dict[str, Any]:
        ll = [float(lng), float(lat)]
        with self._lock:
            self._start_position = ll
            if not self._sim_running:
                self._sim_pos = list(ll)
                self._sim_yaw_deg = None
                self._sim_done = False
        return {"ok": True, "reference_position": ll}

    def set_orbit_center(self, lng: float, lat: float) -> dict[str, Any]:
        ll = [float(lng), float(lat)]
        with self._lock:
            self._mission_type = "orbit_scan"
            self._orbit_center = ll
            self._area_polygon = []
            self._path_waypoints = []
            self._waypoint_meta = []
            self._orbit_meta = {}
            self._preview = MissionPreview(0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
            self._sim_running = False
            self._sim_paused = False
            self._sim_done = False
            self._sim_index = 0
            self._sim_yaw_deg = None
        return {"ok": True, "orbit_center": ll}

    def set_landing_position(self, lng: float, lat: float) -> dict[str, Any]:
        ll = [float(lng), float(lat)]
        with self._lock:
            self._landing_position = ll
        return {"ok": True, "landing_position": ll}

    def clear_landing_position(self) -> dict[str, Any]:
        with self._lock:
            self._landing_position = None
        return {"ok": True}

    def clear(self) -> dict[str, Any]:
        with self._lock:
            self._mission_type = "ground_scan"
            self._area_polygon = []
            self._orbit_center = None
            self._start_position = None
            self._landing_position = None
            self._path_waypoints = []
            self._waypoint_meta = []
            self._orbit_meta = {}
            self._preview = MissionPreview(0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
            self._sim_running = False
            self._sim_paused = False
            self._sim_done = False
            self._sim_index = 0
            self._sim_pos = None
            self._sim_yaw_deg = None
        return {"ok": True}

    def _candidate_angles(self, poly_xy: list[tuple[float, float]]) -> set[float]:
        candidate_angles = {_normalize_scan_angle(0.0)}
        for i in range(len(poly_xy)):
            x1, y1 = poly_xy[i]
            x2, y2 = poly_xy[(i + 1) % len(poly_xy)]
            dx = x2 - x1
            dy = y2 - y1
            if math.hypot(dx, dy) < 0.05:
                continue
            base = _normalize_scan_angle(math.atan2(dy, dx))
            candidate_angles.add(base)
            candidate_angles.add(_normalize_scan_angle(base + (math.pi * 0.5)))
        return candidate_angles

    def _plan_for_spacing(
        self,
        poly_xy: list[tuple[float, float]],
        start_xy: tuple[float, float],
        spacing: float,
    ) -> ScanPlan:
        candidate_angles = self._candidate_angles(poly_xy)

        best_path_xy: list[tuple[float, float]] | None = None
        best_path_len = float("inf")
        best_pass_count = 0
        best_angle = 0.0
        best_lead_in = 0.0
        best_return = 0.0
        for angle in sorted(candidate_angles):
            rotated_poly = [_rotate_xy(x, y, -angle) for x, y in poly_xy]
            rotated_start = _rotate_xy(start_xy[0], start_xy[1], -angle)
            try:
                rotated_path, rotated_len, pass_count, lead_in_m, return_to_home_m = _build_scan_path_xy(
                    rotated_poly,
                    rotated_start,
                    spacing,
                    self._footprint_radius_m,
                )
            except ValueError:
                continue
            candidate_path = [_rotate_xy(x, y, angle) for x, y in rotated_path]
            if rotated_len < (best_path_len - 1e-6) or (
                abs(rotated_len - best_path_len) <= 1e-6 and pass_count < best_pass_count
            ):
                best_path_xy = candidate_path
                best_path_len = rotated_len
                best_pass_count = pass_count
                best_angle = angle
                best_lead_in = lead_in_m
                best_return = return_to_home_m

        if not best_path_xy:
            raise ValueError("scan generator produced no passes for area")

        sweep_width = self._footprint_radius_m * 2.0
        coverage_ratio = min(1.0, sweep_width / spacing) if spacing > 0 else 0.0
        overlap_ratio = max(0.0, (sweep_width - spacing) / max(sweep_width, 1e-6))
        return ScanPlan(
            path_xy=best_path_xy,
            path_length_m=best_path_len,
            pass_count=best_pass_count,
            sweep_angle_deg=math.degrees(best_angle),
            lead_in_m=best_lead_in,
            return_to_home_m=best_return,
            spacing_m=spacing,
            expected_coverage_pct=100.0 * coverage_ratio,
            overlap_pct_est=100.0 * overlap_ratio,
        )

    def _auto_spacing_candidates(self) -> list[float]:
        sweep_width = self._footprint_radius_m * 2.0
        factors = [0.55, 0.65, 0.75, 0.85, 0.9, 0.95, 1.0]
        return sorted({max(1.0, sweep_width * factor) for factor in factors})

    def _select_spacing_plan(
        self,
        poly_xy: list[tuple[float, float]],
        start_xy: tuple[float, float],
        requested_spacing: float,
        auto_spacing: bool,
    ) -> ScanPlan:
        if not auto_spacing:
            return self._plan_for_spacing(poly_xy, start_xy, max(1.0, float(requested_spacing)))

        area_m2 = _polygon_area_m2(poly_xy)
        candidates: list[tuple[float, ScanPlan]] = []
        for spacing in self._auto_spacing_candidates():
            try:
                plan = self._plan_for_spacing(poly_xy, start_xy, spacing)
            except ValueError:
                continue
            coverage_ratio = max(0.1, plan.expected_coverage_pct / 100.0)
            overlap_ratio = max(0.0, plan.overlap_pct_est / 100.0)
            cost = (plan.path_length_m * (1.0 + 0.45 * overlap_ratio)) / coverage_ratio
            candidates.append((cost, plan))

        if not candidates:
            raise ValueError("scan generator could not find an automatic spacing plan")

        if area_m2 <= 900.0:
            preferred = [item for item in candidates if item[1].expected_coverage_pct >= 99.0]
        elif area_m2 <= 2500.0:
            preferred = [item for item in candidates if item[1].expected_coverage_pct >= 98.0]
        else:
            preferred = [item for item in candidates if item[1].expected_coverage_pct >= 97.0]
        if not preferred:
            preferred = [item for item in candidates if item[1].expected_coverage_pct >= 95.0]
        if not preferred:
            preferred = candidates

        best_plan: ScanPlan | None = None
        best_cost = float("inf")
        for cost, plan in preferred:
            small_area_overlap_penalty = 0.0
            if area_m2 <= 900.0:
                small_area_overlap_penalty = 0.15 * plan.overlap_pct_est
            elif area_m2 <= 2500.0:
                small_area_overlap_penalty = 0.08 * plan.overlap_pct_est
            effective_cost = cost + small_area_overlap_penalty
            if effective_cost < (best_cost - 1e-6) or (
                abs(effective_cost - best_cost) <= 1e-6 and plan.path_length_m < (best_plan.path_length_m if best_plan else float("inf"))
            ):
                best_plan = plan
                best_cost = effective_cost
        if best_plan is None:
            raise ValueError("scan generator could not find an automatic spacing plan")
        return best_plan

    def _build_orbit_plan(
        self,
        center_lng: float,
        center_lat: float,
        start_lng: float,
        start_lat: float,
        radius_m: float,
        points_per_lap: int,
        clockwise: bool,
        yaw_to_center: bool,
        layers: list[dict[str, Any]],
    ) -> OrbitPlan:
        radius = max(1.0, float(radius_m))
        density = max(8, int(points_per_lap))
        direction = -1.0 if clockwise else 1.0
        origin_lng = float(center_lng)
        origin_lat = float(center_lat)
        center_xy = (0.0, 0.0)
        start_xy = _ll_to_xy_m(origin_lng, origin_lat, start_lng, start_lat)
        clean_layers: list[dict[str, Any]] = []
        for idx, layer in enumerate(layers):
            altitude_m = max(1.0, float(layer.get("altitude_m") or 0.0))
            laps = max(1, int(layer.get("laps") or 1))
            clean_layers.append({"altitude_m": altitude_m, "laps": laps, "layer_index": idx})
        if not clean_layers:
            raise ValueError("orbit scan requires at least one layer")

        orbit_waypoints: list[list[float]] = []
        orbit_meta: list[dict[str, Any]] = []
        total_laps = 0
        total_ring_length = 0.0
        vertical_transition_m = 0.0

        for layer in clean_layers:
            altitude_m = float(layer["altitude_m"])
            laps = int(layer["laps"])
            layer_index = int(layer["layer_index"])
            for lap_index in range(laps):
                for point_index in range(density):
                    orbit_index = len(orbit_waypoints)
                    angle = direction * ((2.0 * math.pi * point_index) / density)
                    px = radius * math.cos(angle)
                    py = radius * math.sin(angle)
                    lng, lat = _xy_to_ll(origin_lng, origin_lat, px, py)
                    yaw_deg = None
                    if yaw_to_center:
                        yaw_deg = _yaw_deg_from_vector(center_xy[0] - px, center_xy[1] - py)
                    orbit_waypoints.append([lng, lat])
                    orbit_meta.append(
                        {
                            "yaw_deg": yaw_deg,
                            "altitude_m": altitude_m,
                            "orbit_index": orbit_index,
                            "lap_index": lap_index,
                            "layer_index": layer_index,
                        }
                    )
                total_laps += 1
                total_ring_length += 2.0 * math.pi * radius
            if layer_index < len(clean_layers) - 1:
                next_alt = float(clean_layers[layer_index + 1]["altitude_m"])
                vertical_transition_m += abs(next_alt - altitude_m)

        if orbit_waypoints:
            orbit_waypoints.append(list(orbit_waypoints[0]))
            closing_meta = dict(orbit_meta[0])
            closing_meta["kind"] = "orbit_close"
            orbit_meta.append(closing_meta)

        first_altitude_m = float(clean_layers[0]["altitude_m"])
        max_altitude_m = max(float(layer["altitude_m"]) for layer in clean_layers)
        start_to_entry_m = math.hypot(start_xy[0] - radius, start_xy[1])
        return_to_home_m = math.hypot(start_xy[0] - radius, start_xy[1])
        path_length_m = math.hypot(start_to_entry_m, first_altitude_m)
        path_length_m += total_ring_length + vertical_transition_m
        path_length_m += math.hypot(return_to_home_m, first_altitude_m)

        full_waypoints = [[float(start_lng), float(start_lat)], *orbit_waypoints, [float(start_lng), float(start_lat)]]
        full_meta = (
            [{"yaw_deg": None, "altitude_m": first_altitude_m, "kind": "start"}]
            + orbit_meta
            + [{"yaw_deg": None, "altitude_m": first_altitude_m, "kind": "return"}]
        )
        return OrbitPlan(
            waypoints_lng_lat=full_waypoints,
            waypoint_meta=full_meta,
            path_length_m=path_length_m,
            radius_m=radius,
            total_laps=total_laps,
            points_per_lap=density,
            clockwise=bool(clockwise),
            yaw_to_center=bool(yaw_to_center),
            layers=[{"altitude_m": float(layer["altitude_m"]), "laps": int(layer["laps"])} for layer in clean_layers],
            first_altitude_m=first_altitude_m,
            max_altitude_m=max_altitude_m,
        )

    def _normalize_orbit_layers(
        self,
        altitude_m: float,
        laps: int,
        layers: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for layer in layers or []:
            if not isinstance(layer, dict):
                continue
            altitude_value = layer.get("altitude_m")
            laps_value = layer.get("laps")
            if altitude_value is None:
                continue
            try:
                normalized.append(
                    {
                        "altitude_m": max(1.0, float(altitude_value)),
                        "laps": max(1, int(laps_value or 1)),
                    }
                )
            except Exception:
                continue
        if normalized:
            return normalized
        return [{"altitude_m": max(1.0, float(altitude_m)), "laps": max(1, int(laps))}]

    def generate_tiny_mission(
        self,
        *,
        start_lng: float,
        start_lat: float,
        heading_deg: float | None = None,
        takeoff_alt_m: float = 3.0,
        hover_before_s: float = 8.0,
        forward_m: float = 3.0,
        hover_after_s: float = 8.0,
        speed_m_s: float = 1.0,
        start_scan: bool = False,
        fence_polygon_lng_lat: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        altitude = min(self.TINY_ALT_MAX_M, max(self.TINY_ALT_MIN_M, float(takeoff_alt_m)))
        hover_1 = min(self.TINY_HOVER_MAX_S, max(self.TINY_HOVER_MIN_S, float(hover_before_s)))
        hover_2 = min(self.TINY_HOVER_MAX_S, max(self.TINY_HOVER_MIN_S, float(hover_after_s)))
        move_forward_m = min(self.TINY_FORWARD_MAX_M, max(self.TINY_FORWARD_MIN_M, float(forward_m)))
        speed = min(self.TINY_SPEED_MAX_M_S, max(self.TINY_SPEED_MIN_M_S, float(speed_m_s)))

        start = [float(start_lng), float(start_lat)]
        yaw_deg = float(heading_deg) if heading_deg is not None else 0.0
        yaw_rad = math.radians(yaw_deg)
        dx = math.sin(yaw_rad) * move_forward_m
        dy = math.cos(yaw_rad) * move_forward_m
        target_lng, target_lat = _xy_to_ll(start[0], start[1], dx, dy)
        target = [float(target_lng), float(target_lat)]

        if fence_polygon_lng_lat is not None:
            for lng, lat in (start, target):
                if not point_inside_polygon(float(lng), float(lat), fence_polygon_lng_lat):
                    raise ValueError("tiny mission target is outside configured fence")

        path_lng_lat = [list(start), list(target), list(start)]
        profile = [
            {"step": 1, "action": "takeoff", "alt_m": altitude},
            {"step": 2, "action": "hold", "duration_s": hover_1},
            {"step": 3, "action": "move_forward", "distance_m": move_forward_m, "speed_m_s": speed},
            {"step": 4, "action": "hold", "duration_s": hover_2},
            {"step": 5, "action": "rtl"},
        ]
        out_and_back_m = 2.0 * move_forward_m
        preview = MissionPreview(
            expected_coverage_pct=0.0,
            estimated_time_s=(out_and_back_m / speed) + hover_1 + hover_2,
            number_of_passes=1,
            path_length_m=out_and_back_m,
            sweep_angle_deg=yaw_deg,
            lead_in_m=0.0,
            return_to_home_m=move_forward_m,
            overlap_pct_est=0.0,
        )

        with self._lock:
            self._mission_type = "tiny_mission"
            self._area_polygon = []
            self._path_waypoints = path_lng_lat
            self._waypoint_meta = [
                {"yaw_deg": yaw_deg, "kind": "start"},
                {"yaw_deg": yaw_deg, "kind": "forward"},
                {"yaw_deg": yaw_deg, "kind": "return"},
            ]
            self._orbit_meta = {
                "preset": "tiny_mission",
                "command_profile": profile,
                "takeoff_alt_m": altitude,
                "hover_before_s": hover_1,
                "forward_m": move_forward_m,
                "hover_after_s": hover_2,
                "speed_m_s": speed,
                "heading_deg": yaw_deg,
            }
            self._preview = preview
            self._spacing_m = 0.0
            self._speed_m_s = speed
            self._start_position = list(start)
            self._sim_index = 0
            self._sim_pos = list(start)
            self._sim_yaw_deg = yaw_deg
            self._sim_done = False
            self._sim_running = bool(start_scan and self._has_path_locked())
            self._sim_paused = False

        return self.get_path()

    def generate_scan(
        self,
        spacing_m: float = 8.0,
        speed_m_s: float = 3.0,
        start_scan: bool = False,
        auto_spacing: bool = False,
    ) -> dict[str, Any]:
        speed = max(0.2, float(speed_m_s))
        with self._lock:
            area = list(self._area_polygon)
            start = list(self._start_position) if self._start_position else None
            landing = list(self._landing_position) if self._landing_position else None
        if len(area) < 3:
            raise ValueError("scan area is not set")
        if start is None:
            raise ValueError("start position is not set")

        origin_lng = float(area[0][0])
        origin_lat = float(area[0][1])
        poly_xy = [_ll_to_xy_m(origin_lng, origin_lat, p[0], p[1]) for p in area]
        start_xy = _ll_to_xy_m(origin_lng, origin_lat, start[0], start[1])
        plan = self._select_spacing_plan(poly_xy, start_xy, spacing_m, auto_spacing)

        path_lng_lat: list[list[float]] = []
        for p in plan.path_xy:
            lng, lat = _xy_to_ll(origin_lng, origin_lat, p[0], p[1])
            path_lng_lat.append([lng, lat])
        path_lng_lat.insert(0, list(start))
        if landing is not None:
            if path_lng_lat[-1] != list(landing):
                path_lng_lat.append(list(landing))
        elif path_lng_lat[-1] != list(start):
            path_lng_lat.append(list(start))

        path_length_m = _path_length_lng_lat_m(path_lng_lat)
        return_leg_m = _ll_distance_m(
            float(path_lng_lat[-2][0]),
            float(path_lng_lat[-2][1]),
            float(path_lng_lat[-1][0]),
            float(path_lng_lat[-1][1]),
        ) if len(path_lng_lat) >= 2 else 0.0

        preview = MissionPreview(
            expected_coverage_pct=plan.expected_coverage_pct,
            estimated_time_s=(path_length_m / speed) if speed > 0 else 0.0,
            number_of_passes=plan.pass_count,
            path_length_m=path_length_m,
            sweep_angle_deg=plan.sweep_angle_deg,
            lead_in_m=plan.lead_in_m,
            return_to_home_m=return_leg_m,
            overlap_pct_est=plan.overlap_pct_est,
        )

        with self._lock:
            self._mission_type = "ground_scan"
            self._path_waypoints = path_lng_lat
            self._waypoint_meta = [{"yaw_deg": None} for _ in path_lng_lat]
            self._orbit_meta = {}
            self._preview = preview
            self._spacing_m = plan.spacing_m
            self._speed_m_s = speed
            self._sim_index = 0
            self._sim_pos = list(start)
            self._sim_yaw_deg = None
            self._sim_done = False
            self._sim_running = bool(start_scan and self._has_path_locked())
            self._sim_paused = False

        return self.get_path()

    def generate_orbit_scan(
        self,
        radius_m: float,
        altitude_m: float,
        laps: int,
        points_per_lap: int,
        clockwise: bool,
        yaw_to_center: bool,
        speed_m_s: float = 3.0,
        start_scan: bool = False,
        layers: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        speed = max(0.2, float(speed_m_s))
        with self._lock:
            center = list(self._orbit_center) if self._orbit_center else None
            start = list(self._start_position) if self._start_position else None
        if center is None:
            raise ValueError("orbit center is not set")
        if start is None:
            raise ValueError("start position is not set")
        orbit_layers = self._normalize_orbit_layers(altitude_m, laps, layers)

        plan = self._build_orbit_plan(
            center_lng=float(center[0]),
            center_lat=float(center[1]),
            start_lng=float(start[0]),
            start_lat=float(start[1]),
            radius_m=radius_m,
            points_per_lap=points_per_lap,
            clockwise=clockwise,
            yaw_to_center=yaw_to_center,
            layers=orbit_layers,
        )
        preview = MissionPreview(
            expected_coverage_pct=0.0,
            estimated_time_s=(plan.path_length_m / speed) if speed > 0 else 0.0,
            number_of_passes=plan.total_laps,
            path_length_m=plan.path_length_m,
            sweep_angle_deg=0.0,
            lead_in_m=0.0,
            return_to_home_m=0.0,
            overlap_pct_est=0.0,
        )

        with self._lock:
            self._mission_type = "orbit_scan"
            self._area_polygon = []
            self._path_waypoints = [list(p) for p in plan.waypoints_lng_lat]
            self._waypoint_meta = [dict(p) for p in plan.waypoint_meta]
            self._orbit_meta = {
                "radius_m": plan.radius_m,
                "altitude_m": plan.first_altitude_m,
                "first_altitude_m": plan.first_altitude_m,
                "max_altitude_m": plan.max_altitude_m,
                "laps": plan.total_laps,
                "points_per_lap": plan.points_per_lap,
                "clockwise": plan.clockwise,
                "yaw_to_center": plan.yaw_to_center,
                "layers": [dict(layer) for layer in plan.layers],
                "layer_count": len(plan.layers),
            }
            self._preview = preview
            self._spacing_m = 0.0
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

                next_meta = self._waypoint_meta[self._sim_index + 1] if (self._sim_index + 1) < len(self._waypoint_meta) else {}
                if next_meta.get("yaw_deg") is not None:
                    self._sim_yaw_deg = float(next_meta["yaw_deg"])
                else:
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
                "mission_type": self._mission_type,
                "scan_area_polygon_lng_lat": list(self._area_polygon),
                "orbit_center_lng_lat": list(self._orbit_center) if self._orbit_center else None,
                "start_position_lng_lat": list(self._start_position) if self._start_position else None,
                "landing_position_lng_lat": list(self._landing_position) if self._landing_position else None,
                "waypoints_lng_lat": list(self._path_waypoints),
                "waypoint_meta": [dict(item) for item in self._waypoint_meta],
                "config": {
                    "spacing_m": self._spacing_m,
                    "speed_m_s": self._speed_m_s,
                    **self._orbit_meta,
                },
                "coverage_preview": {
                    "expected_coverage_pct": self._preview.expected_coverage_pct,
                    "estimated_time_s": self._preview.estimated_time_s,
                    "number_of_passes": self._preview.number_of_passes,
                    "path_length_m": self._preview.path_length_m,
                    "sweep_angle_deg": self._preview.sweep_angle_deg,
                    "lead_in_m": self._preview.lead_in_m,
                    "return_to_home_m": self._preview.return_to_home_m,
                    "overlap_pct_est": self._preview.overlap_pct_est,
                },
                "sim": sim,
                "sim_running": bool(sim_state["sim_running"]),
                "sim_paused": bool(sim_state["sim_paused"]),
                "sim_done": bool(sim_state["sim_done"]),
            }
