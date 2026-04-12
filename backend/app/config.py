from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import yaml


@dataclass(frozen=True)
class BackendConfig:
    runtime_mode: str
    host: str
    port: int
    frontend_origin: str
    log_level: str
    mavlink_url: str
    mavlink_reconnect_sec: float
    mavlink_heartbeat_timeout_sec: float
    default_takeoff_alt_m: float
    map_provider: str
    tencent_key: str
    map_default_center_lng: float
    map_default_center_lat: float
    map_default_zoom: int
    map_bounds_w_m: float
    map_bounds_h_m: float
    track_max_points: int
    coverage_cell_size_m: float
    coverage_footprint_radius_m: float
    allowed_fence_polygon_lng_lat: list[list[float]]
    real_serial_default_port: str
    real_serial_default_baud: int


def _parse_polygon_payload(raw: Any) -> list[list[float]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    if not isinstance(raw, list):
        return []
    out: list[list[float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            lng = float(item[0])
            lat = float(item[1])
        except Exception:
            continue
        out.append([lng, lat])
    if len(out) >= 2 and out[0] == out[-1]:
        out = out[:-1]
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    return {}


def load_config(repo_root: Path | None = None) -> BackendConfig:
    root = repo_root or Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")
    cfg = _load_yaml(root / "config.yaml")
    backend = cfg.get("backend", {}) if isinstance(cfg.get("backend"), dict) else {}
    frontend = cfg.get("frontend", {}) if isinstance(cfg.get("frontend"), dict) else {}
    maps = cfg.get("maps", {}) if isinstance(cfg.get("maps"), dict) else {}
    mav = cfg.get("mavlink", {}) if isinstance(cfg.get("mavlink"), dict) else {}

    runtime_mode = str(os.getenv("RUNTIME_MODE", backend.get("runtime_mode", "simulation"))).strip().lower()
    if runtime_mode not in {"simulation", "real_mission"}:
        runtime_mode = "simulation"

    host = str(os.getenv("BACKEND_HOST", backend.get("host", "127.0.0.1")))
    port = int(os.getenv("BACKEND_PORT", backend.get("port", 8000)))
    frontend_origin = str(
        os.getenv("FRONTEND_ORIGIN", frontend.get("origin", "http://127.0.0.1:5173"))
    )
    log_level = str(os.getenv("LOG_LEVEL", backend.get("log_level", "INFO"))).upper()
    mavlink_url = str(os.getenv("MAVLINK_URL", mav.get("url", "udpin:127.0.0.1:14550")))
    mavlink_reconnect_sec = float(os.getenv("MAVLINK_RECONNECT_SEC", mav.get("reconnect_sec", 1.0)))
    mavlink_heartbeat_timeout_sec = float(
        os.getenv("MAVLINK_HEARTBEAT_TIMEOUT_SEC", mav.get("heartbeat_timeout_sec", 5.0))
    )
    default_takeoff_alt_m = float(
        os.getenv("DEFAULT_TAKEOFF_ALT_M", mav.get("default_takeoff_alt_m", 10.0))
    )
    map_provider = str(os.getenv("MAP_PROVIDER", maps.get("provider", "tencent")))
    tencent_key = str(os.getenv("TENCENT_MAP_KEY", ""))
    map_default_center_lng = float(os.getenv("MAP_DEFAULT_CENTER_LNG", maps.get("default_center_lng", 116.397428)))
    map_default_center_lat = float(os.getenv("MAP_DEFAULT_CENTER_LAT", maps.get("default_center_lat", 39.909230)))
    map_default_zoom = int(os.getenv("MAP_DEFAULT_ZOOM", maps.get("default_zoom", 16)))
    map_bounds_w_m = float(os.getenv("MAP_BOUNDS_W_M", maps.get("bounds_w_m", 120.0)))
    map_bounds_h_m = float(os.getenv("MAP_BOUNDS_H_M", maps.get("bounds_h_m", 120.0)))
    track_max_points = int(os.getenv("TRACK_MAX_POINTS", maps.get("track_max_points", 2000)))
    coverage_cell_size_m = float(os.getenv("COVERAGE_CELL_SIZE_M", maps.get("coverage_cell_size_m", 5.0)))
    coverage_footprint_radius_m = float(
        os.getenv("COVERAGE_FOOTPRINT_RADIUS_M", maps.get("coverage_footprint_radius_m", 6.0))
    )
    allowed_fence_polygon_lng_lat = _parse_polygon_payload(
        os.getenv("ALLOWED_FENCE_POLYGON_LNG_LAT", maps.get("allowed_fence_polygon_lng_lat", []))
    )
    real_serial_default_port = str(os.getenv("REAL_SERIAL_DEFAULT_PORT", backend.get("real_serial_default_port", "")))
    real_serial_default_baud = int(os.getenv("REAL_SERIAL_DEFAULT_BAUD", backend.get("real_serial_default_baud", 57600)))
    return BackendConfig(
        runtime_mode=runtime_mode,
        host=host,
        port=port,
        frontend_origin=frontend_origin,
        log_level=log_level,
        mavlink_url=mavlink_url,
        mavlink_reconnect_sec=mavlink_reconnect_sec,
        mavlink_heartbeat_timeout_sec=mavlink_heartbeat_timeout_sec,
        default_takeoff_alt_m=default_takeoff_alt_m,
        map_provider=map_provider,
        tencent_key=tencent_key,
        map_default_center_lng=map_default_center_lng,
        map_default_center_lat=map_default_center_lat,
        map_default_zoom=map_default_zoom,
        map_bounds_w_m=map_bounds_w_m,
        map_bounds_h_m=map_bounds_h_m,
        track_max_points=track_max_points,
        coverage_cell_size_m=coverage_cell_size_m,
        coverage_footprint_radius_m=coverage_footprint_radius_m,
        allowed_fence_polygon_lng_lat=allowed_fence_polygon_lng_lat,
        real_serial_default_port=real_serial_default_port,
        real_serial_default_baud=real_serial_default_baud,
    )
