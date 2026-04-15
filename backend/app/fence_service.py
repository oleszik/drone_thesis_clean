from __future__ import annotations

import math
from typing import Any

from .config import BackendConfig


def _meters_to_deg_lat(m: float) -> float:
    return float(m) / 111320.0


def _meters_to_deg_lng(m: float, lat_deg: float) -> float:
    c = math.cos(math.radians(float(lat_deg)))
    return float(m) / (111320.0 * max(0.1, abs(c)))


def _safe_float(v: Any) -> float | None:
    try:
        out = float(v)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _polygon_area_m2(polygon_lng_lat: list[list[float]] | list[tuple[float, float]] | None) -> float:
    poly = _clean_polygon(polygon_lng_lat)
    if len(poly) < 3:
        return 0.0
    lat_ref = sum(float(p[1]) for p in poly) / len(poly)
    meters_per_deg_lng = 111320.0 * max(0.1, abs(math.cos(math.radians(lat_ref))))
    origin_lng = float(poly[0][0])
    origin_lat = float(poly[0][1])
    xy: list[tuple[float, float]] = []
    for lng, lat in poly:
        x = (float(lng) - origin_lng) * meters_per_deg_lng
        y = (float(lat) - origin_lat) * 111320.0
        xy.append((x, y))
    acc = 0.0
    for i in range(len(xy)):
        x1, y1 = xy[i]
        x2, y2 = xy[(i + 1) % len(xy)]
        acc += (x1 * y2) - (x2 * y1)
    return abs(acc) * 0.5


def _clean_polygon(points: list[list[float]] | list[tuple[float, float]] | None) -> list[list[float]]:
    cleaned: list[list[float]] = []
    for p in points or []:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        lng = _safe_float(p[0])
        lat = _safe_float(p[1])
        if lng is None or lat is None:
            continue
        cleaned.append([lng, lat])
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
        cleaned = cleaned[:-1]
    return cleaned


def _rect_polygon_from_config(cfg: BackendConfig) -> list[list[float]]:
    center_lng = float(cfg.map_default_center_lng)
    center_lat = float(cfg.map_default_center_lat)
    half_w = max(0.0, float(cfg.map_bounds_w_m) * 0.5)
    half_h = max(0.0, float(cfg.map_bounds_h_m) * 0.5)
    d_lng = _meters_to_deg_lng(half_w, center_lat)
    d_lat = _meters_to_deg_lat(half_h)
    return [
        [center_lng - d_lng, center_lat - d_lat],
        [center_lng + d_lng, center_lat - d_lat],
        [center_lng + d_lng, center_lat + d_lat],
        [center_lng - d_lng, center_lat + d_lat],
    ]


def get_operating_fence(cfg: BackendConfig) -> dict[str, Any]:
    polygon_cfg = _clean_polygon(cfg.allowed_fence_polygon_lng_lat)
    if len(polygon_cfg) >= 3:
        area_m2 = _polygon_area_m2(polygon_cfg)
        return {
            "configured": True,
            "source": "polygon",
            "polygon_lng_lat": polygon_cfg,
            "point_count": len(polygon_cfg),
            "area_m2": area_m2,
            "max_mission_area_m2": area_m2,
        }

    rect_available = float(cfg.map_bounds_w_m) > 1.0 and float(cfg.map_bounds_h_m) > 1.0
    if rect_available:
        rect_poly = _rect_polygon_from_config(cfg)
        area_m2 = _polygon_area_m2(rect_poly)
        return {
            "configured": True,
            "source": "map_bounds_rect",
            "polygon_lng_lat": rect_poly,
            "point_count": len(rect_poly),
            "area_m2": area_m2,
            "max_mission_area_m2": area_m2,
        }

    return {
        "configured": False,
        "source": "none",
        "polygon_lng_lat": [],
        "point_count": 0,
        "area_m2": 0.0,
        "max_mission_area_m2": 0.0,
    }


def _point_on_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> bool:
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > 1e-10:
        return False
    dot = (px - ax) * (bx - ax) + (py - ay) * (by - ay)
    if dot < 0:
        return False
    seg_len_sq = (bx - ax) * (bx - ax) + (by - ay) * (by - ay)
    return dot <= seg_len_sq


def point_inside_polygon(lng: float, lat: float, polygon_lng_lat: list[list[float]]) -> bool:
    poly = _clean_polygon(polygon_lng_lat)
    if len(poly) < 3:
        return False

    x = float(lng)
    y = float(lat)

    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = float(poly[i][0]), float(poly[i][1])
        x2, y2 = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])

        if _point_on_segment(x, y, x1, y1, x2, y2):
            return True

        intersects = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / ((y2 - y1) + 1e-15) + x1)
        if intersects:
            inside = not inside
    return inside


def mission_points_from_payload(mission_path: dict[str, Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []

    for key in ("waypoints_lng_lat", "scan_area_polygon_lng_lat"):
        for p in (mission_path.get(key) or []):
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                lng = _safe_float(p[0])
                lat = _safe_float(p[1])
                if lng is not None and lat is not None:
                    points.append((lng, lat))

    for key in ("start_position_lng_lat", "orbit_center_lng_lat", "landing_position_lng_lat"):
        p = mission_path.get(key)
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            lng = _safe_float(p[0])
            lat = _safe_float(p[1])
            if lng is not None and lat is not None:
                points.append((lng, lat))

    return points


def check_points_inside_fence(points: list[tuple[float, float]], fence: dict[str, Any]) -> tuple[bool, list[tuple[float, float]]]:
    polygon = fence.get("polygon_lng_lat") if isinstance(fence, dict) else []
    if not bool(fence.get("configured")):
        return False, points

    outside = [(lng, lat) for lng, lat in points if not point_inside_polygon(lng, lat, polygon)]
    return len(outside) == 0, outside


def validate_mission_payload_inside_fence(mission_path: dict[str, Any], fence: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    points = mission_points_from_payload(mission_path)
    if not bool(fence.get("configured")):
        return False, "operation fence is not configured", {"checked_points": len(points), "outside_points": len(points)}
    if not points:
        return True, "no mission points to validate yet", {"checked_points": 0, "outside_points": 0}

    ok, outside = check_points_inside_fence(points, fence)
    if not ok:
        return (
            False,
            f"{len(outside)} mission point(s) outside configured fence",
            {
                "checked_points": len(points),
                "outside_points": len(outside),
                "outside_sample": [[outside[0][0], outside[0][1]]] if outside else [],
            },
        )

    return True, "mission points are inside configured fence", {"checked_points": len(points), "outside_points": 0}


def validate_mission_area_size_within_fence(
    area_polygon_lng_lat: list[list[float]] | list[tuple[float, float]],
    fence: dict[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    area_m2 = _polygon_area_m2(area_polygon_lng_lat)
    fence_poly = _clean_polygon(fence.get("polygon_lng_lat") if isinstance(fence, dict) else [])
    max_area_m2 = _safe_float((fence or {}).get("max_mission_area_m2")) if isinstance(fence, dict) else None
    if max_area_m2 is None or max_area_m2 <= 0.0:
        max_area_m2 = _polygon_area_m2(fence_poly)
    if max_area_m2 <= 0.0:
        return False, "operation fence area is unavailable", {"requested_area_m2": area_m2, "max_area_m2": 0.0}
    if area_m2 > (max_area_m2 * 1.001):
        return (
            False,
            f"mission area exceeds max allowed area ({area_m2:.1f} m^2 > {max_area_m2:.1f} m^2)",
            {"requested_area_m2": area_m2, "max_area_m2": max_area_m2},
        )
    return True, "mission area size within allowed limit", {"requested_area_m2": area_m2, "max_area_m2": max_area_m2}
