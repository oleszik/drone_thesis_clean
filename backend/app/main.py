from __future__ import annotations
import logging
import math
import ssl
import threading
import time
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from .config import load_config
from .coverage_service import CoverageConfig, CoverageService
from .mavlink_service import MavlinkService, MavlinkSettings
from .mission_service import MissionService
from .run_manager import RunManager
from .services.mavlink.serial_radio_transport import RealRadioService
from .services.mavlink.transport import normalize_runtime_mode, resolve_transport_url
from .services.mission.real_mission_service import RealMissionService
from .services.mission.simulation_mission_service import SimulationMissionService
from .services.safety.autonomy_guard import AutonomyGuard
from .services.safety.fence_service import (
    check_points_inside_fence,
    get_operating_fence,
    validate_mission_area_size_within_fence,
    validate_mission_payload_inside_fence,
)
from .services.safety.readiness_service import ReadinessService
from .sse import encode_sse, encode_sse_comment, sleep_or_disconnect, stable_json
from .sitl_executor import SitlExecutor


class ModeRequest(BaseModel):
    mode: str


class TakeoffRequest(BaseModel):
    alt_m: float | None = None


class MissionAreaRequest(BaseModel):
    polygon_lng_lat: list[list[float]]


class MissionStartRequest(BaseModel):
    lng: float
    lat: float


class MissionOrbitLayerRequest(BaseModel):
    altitude_m: float
    laps: int = 1


class MissionOrbitGenerateRequest(BaseModel):
    radius_m: float = 12.0
    altitude_m: float = 10.0
    laps: int = 1
    points_per_lap: int = 24
    clockwise: bool = True
    yaw_to_center: bool = True
    speed_m_s: float = 3.0
    start_scan: bool = False
    layers: list[MissionOrbitLayerRequest] | None = None


class MissionGenerateRequest(BaseModel):
    spacing_m: float = 8.0
    speed_m_s: float = 3.0
    start_scan: bool = False
    auto_spacing: bool = False


class TinyMissionRequest(BaseModel):
    mission_profile: str = "out_and_back"
    takeoff_alt_m: float = 3.0
    hover_before_s: float = 8.0
    forward_m: float = 3.0
    hover_after_s: float = 8.0
    speed_m_s: float = 1.0
    heading_deg: float | None = None
    start_lng: float | None = None
    start_lat: float | None = None
    start_scan: bool = False


class SimTickRequest(BaseModel):
    dt: float = 0.1


class SitlStartRequest(BaseModel):
    alt_m: float = 10.0
    accept_radius_m: float = 3.0


class RunStartRequest(BaseModel):
    scenario: dict[str, object] | None = None
    controller: dict[str, object] | None = None
    notes: str | None = None


class RunNotesRequest(BaseModel):
    notes: str = ""


class RealRadioConnectRequest(BaseModel):
    serial_port: str
    serial_baud: int = 57600


class CompassNorthReferenceRequest(BaseModel):
    north_heading_deg: float = 0.0


def _model_to_dict(model: BaseModel) -> dict[str, object]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def create_app() -> FastAPI:
    cfg = load_config(Path(__file__).resolve().parents[2])
    runtime_mode = normalize_runtime_mode(cfg.runtime_mode)
    logging.basicConfig(
        level=getattr(logging, cfg.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    app = FastAPI(title="Drone Thesis API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[cfg.frontend_origin],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    mav = MavlinkService(
        MavlinkSettings(
            connection_url=resolve_transport_url(runtime_mode=runtime_mode, configured_url=cfg.mavlink_url),
            reconnect_sec=cfg.mavlink_reconnect_sec,
            heartbeat_timeout_sec=cfg.mavlink_heartbeat_timeout_sec,
            default_takeoff_alt_m=cfg.default_takeoff_alt_m,
            track_max_points=cfg.track_max_points,
        )
    )
    coverage = CoverageService(
        CoverageConfig(
            origin_lng=cfg.map_default_center_lng,
            origin_lat=cfg.map_default_center_lat,
            bounds_w_m=cfg.map_bounds_w_m,
            bounds_h_m=cfg.map_bounds_h_m,
            cell_size_m=cfg.coverage_cell_size_m,
            footprint_radius_m=cfg.coverage_footprint_radius_m,
        )
    )
    mission = MissionService(footprint_radius_m=cfg.coverage_footprint_radius_m)
    readiness = ReadinessService(cfg)
    sitl = SitlExecutor(mav=mav, mission=mission)
    sim_mission = SimulationMissionService(mission=mission, sitl=sitl)
    real_mission = RealMissionService(mission=mission)
    real_radio = RealRadioService(
        reconnect_sec=cfg.mavlink_reconnect_sec,
        heartbeat_timeout_sec=cfg.mavlink_heartbeat_timeout_sec,
        default_takeoff_alt_m=cfg.default_takeoff_alt_m,
        track_max_points=cfg.track_max_points,
        default_serial_port=cfg.real_serial_default_port,
        default_serial_baud=cfg.real_serial_default_baud,
    )
    real_executor = SitlExecutor(mav=real_radio, mission=mission)
    runs = RunManager()
    cov_stop_evt = threading.Event()
    cov_thread: threading.Thread | None = None

    def _snapshot() -> dict[str, object]:
        return {
            "status": mav.get_status(),
            "telemetry": mav.get_telemetry(),
            "track": track(limit=5000),
            "mission_path": mission.get_path(),
            "coverage": coverage.get_coverage(),
            "sitl_state": sitl.get_state(),
            "map_state": map_state(),
        }

    def _bridge_state_payload() -> dict[str, object]:
        return {
            "status": mav.get_status(),
            "map_state": map_state(),
            "mission_path": mission.get_path(),
            "mission_sim": mission.get_sim_state(),
            "sitl_state": sitl.get_state(),
            "run": runs.current(),
        }

    def _coverage_payload() -> dict[str, object]:
        return {
            "coverage": coverage.get_coverage(),
            "scan_debug": coverage.get_scan_debug(),
        }

    def _track_delta_payload(last_t_unix: float | None, limit: int = 600) -> tuple[dict[str, object] | None, float | None]:
        items = mav.get_track(limit=limit)
        if last_t_unix is None:
            latest = float(items[-1]["t_unix"]) if items else None
            return {
                "reset": True,
                "items": items,
                "latest_t_unix": latest,
            }, latest
        delta = [item for item in items if float(item.get("t_unix", 0.0)) > float(last_t_unix)]
        if not delta:
            return None, last_t_unix
        latest = float(delta[-1]["t_unix"])
        return {
            "reset": False,
            "items": delta,
            "latest_t_unix": latest,
        }, latest

    def _meters_to_deg_lat(m: float) -> float:
        return float(m) / 111320.0

    def _meters_to_deg_lng(m: float, lat_deg: float) -> float:
        c = math.cos(math.radians(float(lat_deg)))
        return float(m) / (111320.0 * max(0.1, abs(c)))

    def _meters_between_lng(lng1: float, lng2: float, lat_ref: float) -> float:
        return abs(float(lng2) - float(lng1)) * 111320.0 * max(0.1, abs(math.cos(math.radians(float(lat_ref)))))

    def _meters_between_lat(lat1: float, lat2: float) -> float:
        return abs(float(lat2) - float(lat1)) * 111320.0

    def _operating_fence() -> dict[str, object]:
        return get_operating_fence(cfg)

    def _require_points_inside_fence(points_lng_lat: list[list[float]], label: str) -> None:
        fence = _operating_fence()
        if not bool(fence.get("configured")):
            raise ValueError("operation fence is not configured")
        points: list[tuple[float, float]] = []
        for p in points_lng_lat:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            try:
                points.append((float(p[0]), float(p[1])))
            except Exception as exc:
                raise ValueError(f"invalid {label} point: {p}") from exc
        if not points:
            raise ValueError(f"{label} requires at least one valid point")
        ok, outside = check_points_inside_fence(points, fence)
        if not ok:
            sample = outside[0] if outside else (None, None)
            raise ValueError(
                f"{label} has points outside configured fence (outside_count={len(outside)}, sample=[{sample[0]}, {sample[1]}])"
            )

    def _require_mission_payload_inside_fence(mission_payload: dict[str, object], *, context: str) -> None:
        fence = _operating_fence()
        ok, msg, details = validate_mission_payload_inside_fence(mission_payload, fence)
        if ok:
            return
        raise ValueError(f"{context}: {msg}; details={details}")

    def _recommended_coverage_cell_size(bounds_w_m: float, bounds_h_m: float) -> float:
        area_m2 = max(1.0, float(bounds_w_m) * float(bounds_h_m))
        # Keep cells fine enough for a readable mission heatmap while bounding
        # total cell count for larger areas.
        footprint_floor = max(0.75, float(cfg.coverage_footprint_radius_m) * 0.18)
        density_floor = math.sqrt(area_m2 / 32000.0)
        return min(float(cfg.coverage_cell_size_m), max(footprint_floor, density_floor))

    def _reset_coverage_for_mission() -> None:
        mission_payload = mission.get_path()
        waypoints = mission_payload.get("waypoints_lng_lat") or []
        area_poly = mission_payload.get("scan_area_polygon_lng_lat") or []
        if len(area_poly) >= 3:
            lngs = [float(p[0]) for p in area_poly]
            lats = [float(p[1]) for p in area_poly]
            min_lng, max_lng = min(lngs), max(lngs)
            min_lat, max_lat = min(lats), max(lats)
            center_lng = 0.5 * (min_lng + max_lng)
            center_lat = 0.5 * (min_lat + max_lat)
            width_m = _meters_between_lng(min_lng, max_lng, center_lat)
            height_m = _meters_between_lat(min_lat, max_lat)
            margin_m = max(15.0, 2.0 * float(cfg.coverage_footprint_radius_m))
            cell_size_m = _recommended_coverage_cell_size(width_m, height_m)
            coverage.reset(
                origin_lng=center_lng,
                origin_lat=center_lat,
                bounds_w_m=max(20.0, width_m + 2.0 * margin_m),
                bounds_h_m=max(20.0, height_m + 2.0 * margin_m),
                cell_size_m=cell_size_m,
                roi_polygon_lng_lat=[[float(p[0]), float(p[1])] for p in area_poly if isinstance(p, (list, tuple)) and len(p) >= 2],
            )
            return
        if len(waypoints) >= 1:
            coverage.reset(
                origin_lng=float(waypoints[0][0]),
                origin_lat=float(waypoints[0][1]),
                cell_size_m=min(float(cfg.coverage_cell_size_m), max(0.75, float(cfg.coverage_footprint_radius_m) * 0.18)),
                roi_polygon_lng_lat=None,
            )
            return
        coverage.reset()

    def _map_origin() -> tuple[float, float]:
        return float(cfg.map_default_center_lng), float(cfg.map_default_center_lat)

    def _resolve_live_reference_lng_lat() -> tuple[float, float] | None:
        tele = mav.get_telemetry()
        lon = tele.get("lon")
        lat = tele.get("lat")
        if lon is not None and lat is not None:
            return float(lon), float(lat)
        sim_vehicle = mission.get_sim_vehicle()
        if sim_vehicle is not None:
            return float(sim_vehicle["lng"]), float(sim_vehicle["lat"])
        return None

    def _sync_mission_reference_from_live_or_map(*, require_live: bool = False) -> tuple[float, float]:
        ref = _resolve_live_reference_lng_lat()
        if ref is None:
            if require_live:
                raise ValueError("reference position is unavailable; wait for live GPS telemetry")
            ref = _map_origin()
        _require_points_inside_fence([[float(ref[0]), float(ref[1])]], "reference position")
        mission.set_reference_position(float(ref[0]), float(ref[1]))
        return float(ref[0]), float(ref[1])

    def _resolve_real_live_reference_lng_lat() -> tuple[float, float] | None:
        tele = real_radio.telemetry()
        lon = tele.get("lon")
        lat = tele.get("lat")
        if lon is not None and lat is not None:
            return float(lon), float(lat)
        return None

    def _sync_real_mission_reference_from_live_or_map(*, require_live: bool = False) -> tuple[float, float]:
        ref = _resolve_real_live_reference_lng_lat()
        if ref is None:
            if require_live:
                raise ValueError("real GPS reference is unavailable; wait for radio telemetry")
            ref = _map_origin()
        _require_points_inside_fence([[float(ref[0]), float(ref[1])]], "real reference position")
        mission.set_reference_position(float(ref[0]), float(ref[1]))
        return float(ref[0]), float(ref[1])

    def _sync_real_planning_reference() -> tuple[float, float]:
        ref = _resolve_real_live_reference_lng_lat()
        if ref is None:
            payload = mission.get_path()
            area = payload.get("scan_area_polygon_lng_lat") or []
            orbit_center = payload.get("orbit_center_lng_lat")
            if isinstance(area, list) and area and isinstance(area[0], list) and len(area[0]) >= 2:
                ref = (float(area[0][0]), float(area[0][1]))
            elif isinstance(orbit_center, list) and len(orbit_center) >= 2:
                ref = (float(orbit_center[0]), float(orbit_center[1]))
            else:
                ref = _map_origin()
        _require_points_inside_fence([[float(ref[0]), float(ref[1])]], "real planning reference position")
        mission.set_reference_position(float(ref[0]), float(ref[1]))
        return float(ref[0]), float(ref[1])

    @app.on_event("startup")
    def _on_startup() -> None:
        mav.start()
        cov_stop_evt.clear()

        def _cov_loop() -> None:
            while not cov_stop_evt.is_set():
                coverage_enabled = sitl.is_scan_active()
                if coverage_enabled:
                    tele = mav.get_telemetry()
                    lng = tele.get("lon")
                    lat = tele.get("lat")
                    if lng is not None and lat is not None:
                        try:
                            coverage.update_from_point(float(lng), float(lat), tele.get("updated_at_unix"))
                        except Exception:
                            pass
                time.sleep(0.25)
        nonlocal cov_thread
        cov_thread = threading.Thread(target=_cov_loop, name="coverage-sampler", daemon=True)
        cov_thread.start()

    @app.on_event("shutdown")
    def _on_shutdown() -> None:
        cov_stop_evt.set()
        if cov_thread is not None:
            cov_thread.join(timeout=1.5)
        mav.stop()
        try:
            real_radio.disconnect()
        except Exception:
            pass

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {
            "ok": True,
            "service": "backend",
            "runtime_mode": runtime_mode,
            "map_provider": cfg.map_provider,
            "tencent_key_configured": bool(cfg.tencent_key),
        }

    @app.get("/api/sim/health")
    def sim_health() -> dict[str, object]:
        return health()

    @app.get("/api/real/health")
    def real_health() -> dict[str, object]:
        return health()

    @app.get("/api/status")
    def status() -> dict[str, object]:
        return mav.get_status()

    @app.get("/api/sim/status")
    def sim_status() -> dict[str, object]:
        return status()

    @app.get("/api/real/status")
    def real_status() -> dict[str, object]:
        return real_radio.status()

    @app.get("/api/readiness")
    def readiness_state() -> dict[str, object]:
        try:
            return readiness.evaluate(
                status=mav.get_status(),
                telemetry=mav.get_telemetry(),
                mission_path=mission.get_path(),
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"readiness evaluation failed: {exc}")

    @app.get("/api/sim/readiness")
    def sim_readiness_state() -> dict[str, object]:
        return readiness_state()

    @app.get("/api/real/readiness")
    def real_readiness_state() -> dict[str, object]:
        snap = readiness.evaluate(
            status=real_radio.status().get("mav_status") or {},
            telemetry=real_radio.telemetry(),
            mission_path=mission.get_path(),
        )
        radio_state = real_radio.status()
        hb_age = radio_state.get("last_heartbeat_age_s")
        tele_age = radio_state.get("last_telemetry_age_s")
        radio_connected = bool(radio_state.get("connected"))
        heartbeat_fresh = radio_connected and hb_age is not None and float(hb_age) <= max(2.0, float(cfg.mavlink_heartbeat_timeout_sec))
        telemetry_fresh = radio_connected and tele_age is not None and float(tele_age) <= 2.5

        checks = list(snap.get("checks") or [])
        checks.extend(
            [
                {
                    "key": "radio_connected",
                    "ok": radio_connected,
                    "severity": "critical",
                    "message": "serial radio connected" if radio_connected else "serial radio disconnected",
                    "value": {
                        "serial_port": radio_state.get("serial_port"),
                        "serial_baud": radio_state.get("serial_baud"),
                    },
                },
                {
                    "key": "radio_heartbeat_fresh",
                    "ok": heartbeat_fresh,
                    "severity": "critical",
                    "message": "radio heartbeat is fresh" if heartbeat_fresh else f"radio heartbeat stale ({hb_age})",
                    "value": hb_age,
                },
                {
                    "key": "radio_telemetry_fresh",
                    "ok": telemetry_fresh,
                    "severity": "critical",
                    "message": "radio telemetry is fresh" if telemetry_fresh else f"radio telemetry stale/lost ({tele_age})",
                    "value": tele_age,
                },
            ]
        )
        blocking = [
            str(item.get("message") or "")
            for item in checks
            if item.get("severity") == "critical" and not bool(item.get("ok"))
        ]
        can_manual = bool(radio_connected)
        can_autonomous = bool(snap.get("can_autonomous")) and heartbeat_fresh and telemetry_fresh and radio_connected
        mav_status = radio_state.get("mav_status") if isinstance(radio_state.get("mav_status"), dict) else {}
        return {
            **snap,
            "overall_ready": bool(can_autonomous),
            "can_manual": can_manual,
            "can_autonomous": can_autonomous,
            "checks": checks,
            "blocking_reasons": blocking,
            "radio": radio_state,
            "last_status_text": str(mav_status.get("last_status_text") or ""),
        }

    def _readiness_snapshot() -> dict[str, object]:
        return readiness.evaluate(
            status=mav.get_status(),
            telemetry=mav.get_telemetry(),
            mission_path=mission.get_path(),
        )

    def _real_readiness_snapshot() -> dict[str, object]:
        return real_readiness_state()

    autonomy_guard = AutonomyGuard(snapshot_provider=_readiness_snapshot)
    real_autonomy_guard = AutonomyGuard(snapshot_provider=_real_readiness_snapshot)

    def _require_autonomy_ready(action: str) -> dict[str, object]:
        return autonomy_guard.require(action)

    def _require_real_autonomy_ready(action: str) -> dict[str, object]:
        return real_autonomy_guard.require(action)

    def _require_real_generation_ready(action: str) -> dict[str, object]:
        snap = real_readiness_state()
        checks = {str(item.get("key")): item for item in (snap.get("checks") or []) if isinstance(item, dict)}
        required = [
            "mavlink_connected",
            "heartbeat_age_sec",
            "gps_ok",
            "ekf_ok",
            "battery_ok",
            "home_position_set",
            "fence_configured",
            "backend_config_ok",
        ]
        blockers = [str(checks.get(key, {}).get("message") or key) for key in required if not bool(checks.get(key, {}).get("ok"))]
        if not blockers:
            return snap
        raise HTTPException(
            status_code=409,
            detail={
                "error": f"autonomy_blocked:{action}",
                "blocking_reasons": blockers,
                "readiness_snapshot": snap,
            },
        )

    @app.post("/api/connection/connect")
    def connection_connect() -> dict[str, object]:
        out = _run_control(mav.connect)
        runs.log("CONNECTED", out)
        return out

    @app.post("/api/connection/disconnect")
    def connection_disconnect() -> dict[str, object]:
        out = _run_control(mav.disconnect)
        runs.log("DISCONNECTED", out)
        return out

    @app.get("/api/telemetry")
    def telemetry() -> dict[str, object]:
        return mav.get_telemetry()

    @app.get("/api/sim/telemetry")
    def sim_telemetry() -> dict[str, object]:
        return telemetry()

    @app.get("/api/real/telemetry")
    def real_telemetry() -> dict[str, object]:
        return real_radio.telemetry()

    @app.get("/api/stream/telemetry")
    async def stream_telemetry(request: Request) -> StreamingResponse:
        async def gen():
            seq = 0
            while True:
                if await request.is_disconnected():
                    break
                payload = {
                    "status": mav.get_status(),
                    "telemetry": mav.get_telemetry(),
                }
                seq += 1
                yield encode_sse(data=payload, event="telemetry", event_id=str(seq))
                if await sleep_or_disconnect(request, 0.2):
                    break

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    @app.get("/api/sim/stream/telemetry")
    async def sim_stream_telemetry(request: Request) -> StreamingResponse:
        return await stream_telemetry(request)

    @app.get("/api/real/stream/telemetry")
    async def real_stream_telemetry(request: Request) -> StreamingResponse:
        async def gen():
            seq = 0
            while True:
                if await request.is_disconnected():
                    break
                payload = {
                    "status": real_radio.status().get("mav_status") or {},
                    "telemetry": real_radio.telemetry(),
                }
                seq += 1
                yield encode_sse(data=payload, event="telemetry", event_id=str(seq))
                if await sleep_or_disconnect(request, 0.2):
                    break

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    @app.get("/api/stream/bridge_state")
    async def stream_bridge_state(request: Request) -> StreamingResponse:
        async def gen():
            seq = 0
            last = ""
            while True:
                if await request.is_disconnected():
                    break
                payload = _bridge_state_payload()
                current = stable_json(payload)
                if current != last:
                    last = current
                    seq += 1
                    yield encode_sse(data=payload, event="bridge_state", event_id=str(seq))
                else:
                    yield encode_sse_comment("bridge_state")
                if await sleep_or_disconnect(request, 1.0):
                    break

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    @app.get("/api/sim/stream/bridge_state")
    async def sim_stream_bridge_state(request: Request) -> StreamingResponse:
        return await stream_bridge_state(request)

    @app.get("/api/stream/coverage")
    async def stream_coverage(request: Request) -> StreamingResponse:
        async def gen():
            seq = 0
            last = ""
            while True:
                if await request.is_disconnected():
                    break
                payload = _coverage_payload()
                current = stable_json(payload)
                if current != last:
                    last = current
                    seq += 1
                    yield encode_sse(data=payload, event="coverage", event_id=str(seq))
                else:
                    yield encode_sse_comment("coverage")
                if await sleep_or_disconnect(request, 1.0):
                    break

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    @app.get("/api/sim/stream/coverage")
    async def sim_stream_coverage(request: Request) -> StreamingResponse:
        return await stream_coverage(request)

    @app.get("/api/stream/track")
    async def stream_track(request: Request, limit: int = 600) -> StreamingResponse:
        async def gen():
            seq = 0
            last_t_unix: float | None = None
            while True:
                if await request.is_disconnected():
                    break
                payload, last_t_unix_next = _track_delta_payload(last_t_unix, limit=limit)
                if payload is not None:
                    last_t_unix = last_t_unix_next
                    seq += 1
                    yield encode_sse(data=payload, event="track", event_id=str(seq))
                else:
                    yield encode_sse_comment("track")
                if await sleep_or_disconnect(request, 0.5):
                    break

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    @app.get("/api/sim/stream/track")
    async def sim_stream_track(request: Request, limit: int = 600) -> StreamingResponse:
        return await stream_track(request, limit=limit)

    @app.get("/api/map_state")
    def map_state() -> dict[str, object]:
        lng, lat = _map_origin()
        half_w = max(1.0, cfg.map_bounds_w_m * 0.5)
        half_h = max(1.0, cfg.map_bounds_h_m * 0.5)
        d_lng = _meters_to_deg_lng(half_w, lat)
        d_lat = _meters_to_deg_lat(half_h)
        bounds_polygon = [
            [lng - d_lng, lat - d_lat],
            [lng + d_lng, lat - d_lat],
            [lng + d_lng, lat + d_lat],
            [lng - d_lng, lat + d_lat],
            [lng - d_lng, lat - d_lat],
        ]
        mission_payload = mission.get_path()
        sim_state = mission.get_sim_state()
        sitl_state = sitl.get_state()
        tele = mav.get_telemetry()
        vehicle = None
        if tele.get("lon") is not None and tele.get("lat") is not None:
            vehicle = {
                "lng": float(tele["lon"]),
                "lat": float(tele["lat"]),
                "yaw_deg": tele.get("yaw_deg"),
            }
        if vehicle is None:
            vehicle = mission_payload.get("sim")
        if vehicle is not None:
            origin = {"lng": float(vehicle["lng"]), "lat": float(vehicle["lat"])}
        else:
            start_ll = mission_payload.get("start_position_lng_lat")
            if isinstance(start_ll, list) and len(start_ll) >= 2:
                origin = {"lng": float(start_ll[0]), "lat": float(start_ll[1])}
            else:
                origin = {"lng": lng, "lat": lat}
        map_provider = str(cfg.map_provider or "").strip().lower()
        tile_template = "/api/map/tile/{z}/{x}/{y}.png" if map_provider == "tencent" else ""
        return {
            "map_provider": map_provider,
            "tile_url_template": tile_template,
            "center_lng_lat": [lng, lat],
            "zoom": int(cfg.map_default_zoom),
            "default_basemap_mode": cfg.map_default_mode,
            "restrict_to_bounds_default": bool(cfg.map_restrict_to_bounds_default),
            "tencent_vector_style": int(cfg.tencent_vector_style),
            "tencent_hybrid_style": int(cfg.tencent_hybrid_style),
            "supported_basemap_modes": ["vector", "satellite", "hybrid"],
            "operating_fence": _operating_fence(),
            "origin": origin,
            "bounds_polygon_lng_lat": bounds_polygon,
            "planned_path_lng_lat": mission_payload.get("waypoints_lng_lat", []),
            "scan_area_polygon_lng_lat": mission_payload.get("scan_area_polygon_lng_lat", []),
            "mission_preview": mission_payload.get("coverage_preview", {}),
            "sim_running": bool(sim_state.get("sim_running")),
            "sim_paused": bool(sim_state.get("sim_paused")),
            "sim_done": bool(sim_state.get("sim_done")),
            "sitl_executor": sitl_state,
            "vehicle": vehicle,
        }

    @app.get("/api/sim/map_state")
    def sim_map_state() -> dict[str, object]:
        return map_state()

    @app.get("/api/track")
    def track(limit: int = 500) -> dict[str, object]:
        items = mav.get_track(limit=limit)
        return {"count": len(items), "items": items}

    @app.get("/api/coverage")
    def get_coverage() -> dict[str, object]:
        return coverage.get_coverage()

    @app.get("/api/sim/coverage")
    def sim_get_coverage() -> dict[str, object]:
        return get_coverage()

    @app.post("/api/coverage/reset")
    def reset_coverage() -> dict[str, object]:
        _reset_coverage_for_mission()
        return {"ok": True}

    @app.post("/api/sim/coverage/reset")
    def sim_reset_coverage() -> dict[str, object]:
        return reset_coverage()

    @app.get("/api/scan/debug")
    def scan_debug() -> dict[str, object]:
        return coverage.get_scan_debug()

    @app.get("/api/sim/scan/debug")
    def sim_scan_debug() -> dict[str, object]:
        return scan_debug()

    @app.post("/api/mission/area")
    def mission_area(req: MissionAreaRequest) -> dict[str, object]:
        try:
            fence = _operating_fence()
            _require_points_inside_fence(req.polygon_lng_lat, "mission area")
            area_ok, area_msg, area_details = validate_mission_area_size_within_fence(req.polygon_lng_lat, fence)
            if not area_ok:
                raise ValueError(f"{area_msg}; details={area_details}")
            out = mission.set_area(req.polygon_lng_lat)
            runs.log("MISSION_AREA_SET", {"points": out.get("points")})
            return out
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/mission/orbit_center")
    def mission_orbit_center(req: MissionStartRequest) -> dict[str, object]:
        try:
            _require_points_inside_fence([[req.lng, req.lat]], "orbit center")
            out = mission.set_orbit_center(req.lng, req.lat)
            runs.log("MISSION_ORBIT_CENTER_SET", {"lng": req.lng, "lat": req.lat})
            return out
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/mission/start_position")
    def mission_start_position(req: MissionStartRequest) -> dict[str, object]:
        try:
            _require_points_inside_fence([[req.lng, req.lat]], "reference position")
            out = mission.set_reference_position(req.lng, req.lat)
            runs.log("MISSION_START_POSITION_SET", {"lng": req.lng, "lat": req.lat})
            return out
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/mission/landing_position")
    def mission_landing_position(req: MissionStartRequest) -> dict[str, object]:
        try:
            _require_points_inside_fence([[req.lng, req.lat]], "landing position")
            out = mission.set_landing_position(req.lng, req.lat)
            runs.log("MISSION_LANDING_POSITION_SET", {"lng": req.lng, "lat": req.lat})
            return out
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.delete("/api/mission/landing_position")
    def mission_landing_position_clear() -> dict[str, object]:
        out = mission.clear_landing_position()
        runs.log("MISSION_LANDING_POSITION_CLEARED", {})
        return out

    @app.post("/api/mission/generate_scan")
    def mission_generate_scan(req: MissionGenerateRequest) -> dict[str, object]:
        try:
            _sync_mission_reference_from_live_or_map(require_live=False)
            out = real_mission.generate_scan(
                spacing_m=req.spacing_m,
                speed_m_s=req.speed_m_s,
                start_scan=req.start_scan,
                auto_spacing=req.auto_spacing,
            )
            _require_mission_payload_inside_fence(out, context="generated scan mission")
            runs.log(
                "MISSION_PATH_GENERATED",
                {
                    "requested_spacing_m": req.spacing_m,
                    "resolved_spacing_m": (out.get("config") or {}).get("spacing_m"),
                    "speed_m_s": req.speed_m_s,
                    "auto_spacing": req.auto_spacing,
                },
            )
            return out
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/mission/generate_orbit_scan")
    def mission_generate_orbit_scan(req: MissionOrbitGenerateRequest) -> dict[str, object]:
        try:
            _sync_mission_reference_from_live_or_map(require_live=False)
            layer_payload = [_model_to_dict(layer) for layer in req.layers] if req.layers else None
            out = real_mission.generate_orbit_scan(
                radius_m=req.radius_m,
                altitude_m=req.altitude_m,
                laps=req.laps,
                points_per_lap=req.points_per_lap,
                clockwise=req.clockwise,
                yaw_to_center=req.yaw_to_center,
                speed_m_s=req.speed_m_s,
                start_scan=req.start_scan,
                layers=layer_payload,
            )
            _require_mission_payload_inside_fence(out, context="generated orbit mission")
            runs.log(
                "MISSION_ORBIT_GENERATED",
                {
                    "radius_m": req.radius_m,
                    "altitude_m": req.altitude_m,
                    "laps": req.laps,
                    "layers": layer_payload,
                    "points_per_lap": req.points_per_lap,
                    "clockwise": req.clockwise,
                    "yaw_to_center": req.yaw_to_center,
                    "speed_m_s": req.speed_m_s,
                },
            )
            return out
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/mission/generate_tiny")
    def mission_generate_tiny(req: TinyMissionRequest) -> dict[str, object]:
        try:
            _require_autonomy_ready("mission_generate_tiny")
            tele = mav.get_telemetry()
            start_lng = req.start_lng if req.start_lng is not None else tele.get("lon")
            start_lat = req.start_lat if req.start_lat is not None else tele.get("lat")
            if start_lng is None or start_lat is None:
                raise ValueError("start position is unavailable; provide start_lng/start_lat or wait for GPS telemetry")
            heading_deg = req.heading_deg if req.heading_deg is not None else tele.get("yaw_deg")
            out = real_mission.generate_tiny_mission(
                start_lng=float(start_lng),
                start_lat=float(start_lat),
                heading_deg=(float(heading_deg) if heading_deg is not None else None),
                mission_profile=req.mission_profile,
                takeoff_alt_m=req.takeoff_alt_m,
                hover_before_s=req.hover_before_s,
                forward_m=req.forward_m,
                hover_after_s=req.hover_after_s,
                speed_m_s=req.speed_m_s,
                start_scan=req.start_scan,
                fence_polygon_lng_lat=list(_operating_fence().get("polygon_lng_lat") or []),
            )
            _require_mission_payload_inside_fence(out, context="generated tiny mission")
            runs.log(
                "MISSION_TINY_GENERATED",
                {
                    "mission_profile": req.mission_profile,
                    "takeoff_alt_m": req.takeoff_alt_m,
                    "hover_before_s": req.hover_before_s,
                    "forward_m": req.forward_m,
                    "hover_after_s": req.hover_after_s,
                    "speed_m_s": req.speed_m_s,
                    "heading_deg": heading_deg,
                    "start_scan": req.start_scan,
                },
            )
            return out
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/mission/path")
    def mission_path() -> dict[str, object]:
        return real_mission.path()

    # --- Simulation namespace (Phase split compatibility layer) ---
    @app.post("/api/sim/mission/area")
    def sim_mission_area(req: MissionAreaRequest) -> dict[str, object]:
        return mission_area(req)

    @app.post("/api/sim/mission/orbit_center")
    def sim_mission_orbit_center(req: MissionStartRequest) -> dict[str, object]:
        return mission_orbit_center(req)

    @app.post("/api/sim/mission/start_position")
    def sim_mission_start_position(req: MissionStartRequest) -> dict[str, object]:
        return mission_start_position(req)

    @app.post("/api/sim/mission/landing_position")
    def sim_mission_landing_position(req: MissionStartRequest) -> dict[str, object]:
        return mission_landing_position(req)

    @app.delete("/api/sim/mission/landing_position")
    def sim_mission_landing_position_clear() -> dict[str, object]:
        return mission_landing_position_clear()

    @app.post("/api/sim/mission/generate_scan")
    def sim_mission_generate_scan(req: MissionGenerateRequest) -> dict[str, object]:
        return mission_generate_scan(req)

    @app.post("/api/sim/mission/generate_orbit_scan")
    def sim_mission_generate_orbit_scan(req: MissionOrbitGenerateRequest) -> dict[str, object]:
        return mission_generate_orbit_scan(req)

    @app.get("/api/sim/mission/path")
    def sim_mission_path() -> dict[str, object]:
        return mission_path()

    @app.post("/api/sim/mission/clear")
    def sim_mission_clear() -> dict[str, object]:
        return mission_clear()

    # --- Real mission namespace (field/operator surface) ---
    @app.get("/api/real/connection/ports")
    def real_connection_ports() -> dict[str, object]:
        return real_radio.list_serial_ports()

    @app.get("/api/real/connection/status")
    def real_connection_status() -> dict[str, object]:
        return real_radio.status()

    @app.post("/api/real/connection/connect")
    def real_connection_connect(req: RealRadioConnectRequest) -> dict[str, object]:
        try:
            out = real_radio.connect(serial_port=req.serial_port, serial_baud=req.serial_baud)
            runs.log("REAL_RADIO_CONNECTED", {"serial_port": req.serial_port, "serial_baud": req.serial_baud})
            return out
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.post("/api/real/connection/disconnect")
    def real_connection_disconnect() -> dict[str, object]:
        out = real_radio.disconnect()
        runs.log("REAL_RADIO_DISCONNECTED", out)
        return out

    @app.post("/api/real/connection/heartbeat_test")
    def real_connection_heartbeat_test() -> dict[str, object]:
        out = real_radio.heartbeat_test()
        if not bool(out.get("ok")):
            raise HTTPException(status_code=409, detail=out)
        return out

    def _run_real_control(fn, *, action: str | None = None):
        def _real_status_snapshot() -> dict[str, object]:
            status_payload = real_radio.status()
            mav_status = status_payload.get("mav_status") if isinstance(status_payload.get("mav_status"), dict) else {}
            mode_now = str(mav_status.get("mode") or "UNKNOWN")
            return {
                "armed": bool(mav_status.get("armed")),
                "resulting_mode": mode_now,
                "last_status_text": str(mav_status.get("last_status_text") or ""),
                "recent_status_text": list(mav_status.get("recent_status_text") or []),
            }

        try:
            out = fn()
            payload = dict(out) if isinstance(out, dict) else {}
            snap = _real_status_snapshot()
            if action and "action" not in payload:
                payload["action"] = action
            payload.setdefault("ok", True)
            payload.setdefault("armed", snap["armed"])
            payload.setdefault("resulting_mode", snap["resulting_mode"])
            payload.setdefault("last_status_text", snap["last_status_text"])
            payload.setdefault("recent_status_text", snap["recent_status_text"])
            return payload
        except RuntimeError as exc:
            snap = _real_status_snapshot()
            raise HTTPException(
                status_code=409,
                detail={
                    "ok": False,
                    "action": action or "real_control",
                    "error": str(exc),
                    "armed": snap["armed"],
                    "resulting_mode": snap["resulting_mode"],
                    "last_status_text": snap["last_status_text"],
                    "recent_status_text": snap["recent_status_text"],
                },
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/real/control/arm")
    def real_control_arm() -> dict[str, object]:
        out = _run_real_control(real_radio.arm, action="arm")
        runs.log("REAL_ARM", out)
        return out

    @app.post("/api/real/control/disarm")
    def real_control_disarm() -> dict[str, object]:
        out = _run_real_control(real_radio.disarm, action="disarm")
        runs.log("REAL_DISARM", out)
        return out

    @app.post("/api/real/control/takeoff")
    def real_control_takeoff(req: TakeoffRequest) -> dict[str, object]:
        out = _run_real_control(lambda: real_radio.takeoff(req.alt_m), action="takeoff")
        runs.log("REAL_TAKEOFF", {"alt_m": req.alt_m})
        return out

    @app.post("/api/real/control/hold")
    def real_control_hold() -> dict[str, object]:
        out = _run_real_control(lambda: real_radio.set_mode("LOITER"), action="hold")
        payload = {**out, "action": "hold", "target_mode": "LOITER"}
        runs.log("REAL_HOLD", payload)
        return payload

    @app.post("/api/real/control/rtl")
    def real_control_rtl() -> dict[str, object]:
        out = _run_real_control(real_radio.rtl, action="rtl")
        runs.log("REAL_RTL", out)
        return out

    @app.post("/api/real/control/land")
    def real_control_land() -> dict[str, object]:
        out = _run_real_control(real_radio.land, action="land")
        runs.log("REAL_LAND", out)
        return out

    @app.post("/api/real/control/land_here")
    def real_control_land_here() -> dict[str, object]:
        # LAND mode commands immediate landing at current position.
        out = _run_real_control(real_radio.land, action="land_here")
        runs.log("REAL_LAND_HERE", out)
        return out

    @app.post("/api/real/control/set_mode")
    def real_control_set_mode(req: ModeRequest) -> dict[str, object]:
        out = _run_real_control(lambda: real_radio.set_mode(req.mode), action="set_mode")
        runs.log("REAL_SET_MODE", {"mode": req.mode})
        return out

    @app.post("/api/real/control/compass_calibrate/start")
    def real_control_compass_calibrate_start() -> dict[str, object]:
        out = _run_real_control(real_radio.start_compass_calibration, action="compass_calibrate_start")
        runs.log("REAL_COMPASS_CALIBRATE_START", out)
        return out

    @app.post("/api/real/control/compass_calibrate/cancel")
    def real_control_compass_calibrate_cancel() -> dict[str, object]:
        out = _run_real_control(real_radio.cancel_compass_calibration, action="compass_calibrate_cancel")
        runs.log("REAL_COMPASS_CALIBRATE_CANCEL", out)
        return out

    @app.post("/api/real/control/level_calibrate")
    def real_control_level_calibrate() -> dict[str, object]:
        out = _run_real_control(real_radio.level_calibration, action="level_calibrate")
        runs.log("REAL_LEVEL_CALIBRATE", out)
        return out

    @app.post("/api/real/control/compass_calibrate/north_reference")
    def real_control_compass_calibrate_north_reference(req: CompassNorthReferenceRequest) -> dict[str, object]:
        out = _run_real_control(
            lambda: real_radio.set_compass_north_reference(north_heading_deg=req.north_heading_deg),
            action="compass_calibrate_north_reference",
        )
        runs.log("REAL_COMPASS_CALIBRATE_NORTH_REFERENCE", out)
        return out

    @app.post("/api/real/mission/generate_tiny")
    def real_mission_generate_tiny(req: TinyMissionRequest) -> dict[str, object]:
        try:
            _require_real_generation_ready("real_mission_generate_tiny")
            tele = real_radio.telemetry()
            start_lng = req.start_lng if req.start_lng is not None else tele.get("lon")
            start_lat = req.start_lat if req.start_lat is not None else tele.get("lat")
            if start_lng is None or start_lat is None:
                raise ValueError("start position is unavailable; provide start_lng/start_lat or wait for GPS telemetry")
            heading_deg = req.heading_deg if req.heading_deg is not None else tele.get("yaw_deg")
            out = real_mission.generate_tiny_mission(
                start_lng=float(start_lng),
                start_lat=float(start_lat),
                heading_deg=(float(heading_deg) if heading_deg is not None else None),
                mission_profile=req.mission_profile,
                takeoff_alt_m=req.takeoff_alt_m,
                hover_before_s=req.hover_before_s,
                forward_m=req.forward_m,
                hover_after_s=req.hover_after_s,
                speed_m_s=req.speed_m_s,
                start_scan=req.start_scan,
                fence_polygon_lng_lat=list(_operating_fence().get("polygon_lng_lat") or []),
            )
            _require_mission_payload_inside_fence(out, context="generated tiny mission")
            runs.log(
                "REAL_MISSION_TINY_GENERATED",
                {
                    "mission_profile": req.mission_profile,
                    "takeoff_alt_m": req.takeoff_alt_m,
                    "hover_before_s": req.hover_before_s,
                    "forward_m": req.forward_m,
                    "hover_after_s": req.hover_after_s,
                    "speed_m_s": req.speed_m_s,
                    "heading_deg": heading_deg,
                    "start_scan": req.start_scan,
                },
            )
            return out
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/real/mission/generate_scan")
    def real_mission_generate_scan(req: MissionGenerateRequest) -> dict[str, object]:
        try:
            _sync_real_planning_reference()
            out = real_mission.generate_scan(
                spacing_m=req.spacing_m,
                speed_m_s=req.speed_m_s,
                start_scan=req.start_scan,
                auto_spacing=req.auto_spacing,
            )
            _require_mission_payload_inside_fence(out, context="generated real scan mission")
            runs.log(
                "REAL_MISSION_PATH_GENERATED",
                {
                    "requested_spacing_m": req.spacing_m,
                    "resolved_spacing_m": (out.get("config") or {}).get("spacing_m"),
                    "speed_m_s": req.speed_m_s,
                    "auto_spacing": req.auto_spacing,
                },
            )
            return out
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/real/mission/generate_orbit_scan")
    def real_mission_generate_orbit_scan(req: MissionOrbitGenerateRequest) -> dict[str, object]:
        try:
            _sync_real_planning_reference()
            layer_payload = [_model_to_dict(layer) for layer in req.layers] if req.layers else None
            out = real_mission.generate_orbit_scan(
                radius_m=req.radius_m,
                altitude_m=req.altitude_m,
                laps=req.laps,
                points_per_lap=req.points_per_lap,
                clockwise=req.clockwise,
                yaw_to_center=req.yaw_to_center,
                speed_m_s=req.speed_m_s,
                start_scan=req.start_scan,
                layers=layer_payload,
            )
            _require_mission_payload_inside_fence(out, context="generated real orbit mission")
            runs.log(
                "REAL_MISSION_ORBIT_GENERATED",
                {
                    "radius_m": req.radius_m,
                    "altitude_m": req.altitude_m,
                    "laps": req.laps,
                    "layers": layer_payload,
                    "points_per_lap": req.points_per_lap,
                    "clockwise": req.clockwise,
                    "yaw_to_center": req.yaw_to_center,
                    "speed_m_s": req.speed_m_s,
                },
            )
            return out
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/real/mission/path")
    def real_mission_path() -> dict[str, object]:
        return mission_path()

    @app.post("/api/real/mission/start")
    def real_mission_start(req: SitlStartRequest) -> dict[str, object]:
        try:
            _require_real_autonomy_ready("real_mission_start")
            _require_mission_payload_inside_fence(mission.get_path(), context="real mission start")
            out = real_executor.start_scan(alt_m=req.alt_m, accept_radius_m=req.accept_radius_m)
            mav_status = real_radio.status().get("mav_status") or {}
            out = {**out, "last_status_text": str(mav_status.get("last_status_text") or "")}
            runs.log("REAL_MISSION_START", {"alt_m": req.alt_m, "accept_radius_m": req.accept_radius_m})
            return out
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.post("/api/real/mission/stop")
    def real_mission_stop() -> dict[str, object]:
        out = real_executor.stop_scan()
        mav_status = real_radio.status().get("mav_status") or {}
        out = {**out, "last_status_text": str(mav_status.get("last_status_text") or "")}
        runs.log("REAL_MISSION_STOP", out)
        return out

    @app.get("/api/real/mission/state")
    def real_mission_state() -> dict[str, object]:
        out = real_executor.get_state()
        mav_status = real_radio.status().get("mav_status") or {}
        return {**out, "last_status_text": str(mav_status.get("last_status_text") or "")}

    @app.post("/api/sim/connection/connect")
    def sim_connection_connect() -> dict[str, object]:
        return connection_connect()

    @app.post("/api/sim/connection/disconnect")
    def sim_connection_disconnect() -> dict[str, object]:
        return connection_disconnect()

    @app.post("/api/sim/control/arm")
    def sim_control_arm() -> dict[str, object]:
        return control_arm()

    @app.post("/api/sim/control/disarm")
    def sim_control_disarm() -> dict[str, object]:
        return control_disarm()

    @app.post("/api/sim/control/takeoff")
    def sim_control_takeoff(req: TakeoffRequest) -> dict[str, object]:
        return control_takeoff(req)

    @app.post("/api/sim/control/rtl")
    def sim_control_rtl() -> dict[str, object]:
        return control_rtl()

    @app.post("/api/sim/control/land")
    def sim_control_land() -> dict[str, object]:
        return control_land()

    @app.post("/api/sim/control/hold")
    def sim_control_hold() -> dict[str, object]:
        return control_hold()

    @app.post("/api/sim/control/set_mode")
    def sim_control_set_mode(req: ModeRequest) -> dict[str, object]:
        return control_set_mode(req)

    @app.post("/api/sim/control/battery_reset")
    def sim_control_battery_reset() -> dict[str, object]:
        return control_battery_reset()

    @app.post("/api/sim/control/compass_calibrate/start")
    def sim_control_compass_calibrate_start() -> dict[str, object]:
        return control_compass_calibrate_start()

    @app.post("/api/sim/control/compass_calibrate/cancel")
    def sim_control_compass_calibrate_cancel() -> dict[str, object]:
        return control_compass_calibrate_cancel()

    @app.post("/api/sim/control/level_calibrate")
    def sim_control_level_calibrate() -> dict[str, object]:
        return control_level_calibrate()

    @app.post("/api/sim/control/compass_calibrate/north_reference")
    def sim_control_compass_calibrate_north_reference(req: CompassNorthReferenceRequest) -> dict[str, object]:
        return control_compass_calibrate_north_reference(req)

    @app.post("/api/mission/clear")
    def mission_clear() -> dict[str, object]:
        try:
            sitl.stop_scan()
        except Exception:
            pass
        out = mission.clear()
        coverage.reset()
        runs.log("MISSION_CLEARED", out)
        return out

    @app.post("/api/mission/sim/start")
    def mission_sim_start() -> dict[str, object]:
        try:
            _require_autonomy_ready("mission_sim_start")
            return sim_mission.sim_start()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/mission/sim/pause")
    def mission_sim_pause() -> dict[str, object]:
        return sim_mission.sim_pause()

    @app.post("/api/mission/sim/stop")
    def mission_sim_stop() -> dict[str, object]:
        return sim_mission.sim_stop()

    @app.post("/api/mission/sim/tick")
    def mission_sim_tick(req: SimTickRequest) -> dict[str, object]:
        return sim_mission.sim_tick(req.dt)

    @app.get("/api/mission/sim/state")
    def mission_sim_state() -> dict[str, object]:
        return sim_mission.sim_state()

    @app.post("/api/sitl/start_scan")
    def sitl_start_scan(req: SitlStartRequest) -> dict[str, object]:
        try:
            _require_autonomy_ready("sitl_start_scan")
            _reset_coverage_for_mission()
            out = sim_mission.sitl_start_scan(alt_m=req.alt_m, accept_radius_m=req.accept_radius_m)
            runs.log("SITL_START_SCAN", {"alt_m": req.alt_m, "accept_radius_m": req.accept_radius_m})
            return out
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.post("/api/sitl/stop_scan")
    def sitl_stop_scan() -> dict[str, object]:
        out = sim_mission.sitl_stop_scan()
        runs.log("SITL_STOP_SCAN", out)
        return out

    @app.get("/api/sitl/state")
    def sitl_state() -> dict[str, object]:
        return sim_mission.sitl_state()

    @app.post("/api/sim/mission/sim/start")
    def sim_ns_mission_sim_start() -> dict[str, object]:
        return mission_sim_start()

    @app.post("/api/sim/mission/sim/pause")
    def sim_ns_mission_sim_pause() -> dict[str, object]:
        return mission_sim_pause()

    @app.post("/api/sim/mission/sim/stop")
    def sim_ns_mission_sim_stop() -> dict[str, object]:
        return mission_sim_stop()

    @app.post("/api/sim/mission/sim/tick")
    def sim_ns_mission_sim_tick(req: SimTickRequest) -> dict[str, object]:
        return mission_sim_tick(req)

    @app.get("/api/sim/mission/sim/state")
    def sim_ns_mission_sim_state() -> dict[str, object]:
        return mission_sim_state()

    @app.post("/api/sim/sitl/start_scan")
    def sim_ns_sitl_start_scan(req: SitlStartRequest) -> dict[str, object]:
        return sitl_start_scan(req)

    @app.post("/api/sim/sitl/stop_scan")
    def sim_ns_sitl_stop_scan() -> dict[str, object]:
        return sitl_stop_scan()

    @app.get("/api/sim/sitl/state")
    def sim_ns_sitl_state() -> dict[str, object]:
        return sitl_state()

    def _run_control(fn):
        try:
            return fn()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/control/arm")
    def control_arm() -> dict[str, object]:
        out = _run_control(mav.arm)
        runs.log("ARM", out)
        return out

    @app.post("/api/control/disarm")
    def control_disarm() -> dict[str, object]:
        out = _run_control(mav.disarm)
        runs.log("DISARM", out)
        return out

    @app.post("/api/control/takeoff")
    def control_takeoff(req: TakeoffRequest) -> dict[str, object]:
        out = _run_control(lambda: mav.takeoff(req.alt_m))
        runs.log("TAKEOFF", {"alt_m": req.alt_m})
        return out

    @app.post("/api/control/rtl")
    def control_rtl() -> dict[str, object]:
        out = _run_control(mav.rtl)
        runs.log("RTL", out)
        return out

    @app.post("/api/control/hold")
    def control_hold() -> dict[str, object]:
        out = _run_control(lambda: mav.set_mode("LOITER"))
        mode_now = str(mav.get_status().get("mode") or "UNKNOWN")
        payload = {**out, "target_mode": "LOITER", "resulting_mode": mode_now}
        runs.log("HOLD", payload)
        return payload

    @app.post("/api/control/land")
    def control_land() -> dict[str, object]:
        out = _run_control(mav.land)
        runs.log("LAND", out)
        return out

    @app.post("/api/control/set_mode")
    def control_set_mode(req: ModeRequest) -> dict[str, object]:
        out = _run_control(lambda: mav.set_mode(req.mode))
        runs.log("SET_MODE", {"mode": req.mode})
        return out

    @app.post("/api/control/battery_reset")
    def control_battery_reset() -> dict[str, object]:
        out = _run_control(mav.battery_reset)
        runs.log("BATTERY_RESET", out)
        return out

    @app.post("/api/control/compass_calibrate/start")
    def control_compass_calibrate_start() -> dict[str, object]:
        out = _run_control(mav.start_compass_calibration)
        runs.log("COMPASS_CALIBRATE_START", out)
        return out

    @app.post("/api/control/compass_calibrate/cancel")
    def control_compass_calibrate_cancel() -> dict[str, object]:
        out = _run_control(mav.cancel_compass_calibration)
        runs.log("COMPASS_CALIBRATE_CANCEL", out)
        return out

    @app.post("/api/control/level_calibrate")
    def control_level_calibrate() -> dict[str, object]:
        out = _run_control(mav.level_calibration)
        runs.log("LEVEL_CALIBRATE", out)
        return out

    @app.post("/api/control/compass_calibrate/north_reference")
    def control_compass_calibrate_north_reference(req: CompassNorthReferenceRequest) -> dict[str, object]:
        out = _run_control(lambda: mav.set_compass_north_reference(north_heading_deg=req.north_heading_deg))
        runs.log("COMPASS_CALIBRATE_NORTH_REFERENCE", out)
        return out

    @app.post("/api/runs/start")
    def runs_start(req: RunStartRequest) -> dict[str, object]:
        try:
            out = runs.start(scenario=req.scenario or {}, controller=req.controller or {}, notes=req.notes or "")
            return out
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.post("/api/runs/stop")
    def runs_stop() -> dict[str, object]:
        try:
            return runs.stop()
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.get("/api/runs/current")
    def runs_current() -> dict[str, object]:
        cur = runs.current()
        if cur is None:
            return {"run": None}
        return {"run": cur}

    @app.post("/api/runs/current/notes")
    def runs_notes(req: RunNotesRequest) -> dict[str, object]:
        try:
            return runs.set_notes(req.notes)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.get("/api/runs/current/export/json")
    def runs_export_json() -> Response:
        try:
            body = runs.export_json(_snapshot)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return Response(
            content=body,
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=run_export.json"},
        )

    @app.get("/api/runs/current/export/path.csv")
    def runs_export_path_csv() -> Response:
        try:
            body = runs.export_path_csv(_snapshot)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return Response(
            content=body,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=run_path.csv"},
        )

    @app.get("/api/runs/current/export/path.geojson")
    def runs_export_path_geojson() -> Response:
        try:
            body = runs.export_path_geojson(_snapshot)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return Response(
            content=body,
            media_type="application/geo+json",
            headers={"Content-Disposition": "attachment; filename=run_path.geojson"},
        )

    @app.get("/api/runs/current/export/coverage.csv")
    def runs_export_coverage_csv() -> Response:
        try:
            body = runs.export_coverage_csv(_snapshot)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return Response(
            content=body,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=run_coverage.csv"},
        )

    @app.get("/api/runs/current/export/report.pdf")
    def runs_export_report_pdf() -> Response:
        try:
            body = runs.export_report_pdf(_snapshot)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return Response(
            content=body,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=run_report.pdf"},
        )

    @app.get("/api/map/tile/{z}/{x}/{y}.png")
    def map_tile(z: int, x: int, y: int, mode: str = "vector", style: int | None = None) -> Response:
        if str(cfg.map_provider).lower() != "tencent":
            raise HTTPException(status_code=404, detail="map provider tile proxy not configured")
        if not cfg.tencent_key:
            raise HTTPException(status_code=404, detail="tencent key not configured")
        if z < 0 or x < 0 or y < 0:
            raise HTTPException(status_code=404, detail="invalid tile")
        n = 1 << int(z)
        if n <= 0:
            raise HTTPException(status_code=404, detail="invalid zoom")
        xw = int(x) % n
        yi = (n - 1) - int(y)
        host_idx = (int(z) + int(xw) + int(yi)) % 4
        hosts = [f"rt{host_idx}", f"rt{(host_idx + 1) % 4}", f"rt{(host_idx + 2) % 4}", f"rt{(host_idx + 3) % 4}"]
        mode_norm = str(mode or "vector").strip().lower()
        if mode_norm not in {"vector", "satellite", "hybrid"}:
            mode_norm = "vector"
        default_style = int(cfg.tencent_vector_style if mode_norm == "vector" else cfg.tencent_hybrid_style)
        style_id = int(style) if style is not None else default_style

        def build_urls(host: str) -> list[str]:
            if mode_norm == "satellite":
                # Tencent satellite tiles are served by p*.map.gtimg.com with a folder fanout by 16.
                px = host.replace("rt", "p")
                return [
                    (
                        f"https://{px}.map.gtimg.com/sateTiles/"
                        f"{int(z)}/{int(xw) // 16}/{int(yi) // 16}/{int(xw)}_{int(yi)}.jpg"
                        f"?key={cfg.tencent_key}"
                    ),
                    (
                        f"https://{host}.map.gtimg.com/realtimerender"
                        f"?z={int(z)}&x={int(xw)}&y={int(yi)}&type=sate&style=0&key={cfg.tencent_key}"
                    ),
                    (
                        f"https://{host}.map.gtimg.com/realtimerender"
                        f"?z={int(z)}&x={int(xw)}&y={int(yi)}&type=satellite&style=0&key={cfg.tencent_key}"
                    ),
                ]
            render_type = "hybrid" if mode_norm == "hybrid" else "vector"
            return [
                (
                    f"https://{host}.map.gtimg.com/realtimerender"
                    f"?z={int(z)}&x={int(xw)}&y={int(yi)}&type={render_type}&style={int(style_id)}&key={cfg.tencent_key}"
                ),
                (
                    f"https://{host}.map.gtimg.com/realtimerender"
                    f"?z={int(z)}&x={int(xw)}&y={int(yi)}&type=vector&style={int(style_id)}&key={cfg.tencent_key}"
                ),
            ]

        ctx = ssl.create_default_context()
        last_err: str | None = None
        attempts = 0
        max_tencent_attempts = 4
        for host in hosts:
            for url in build_urls(host):
                attempts += 1
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                try:
                    with urllib.request.urlopen(req, timeout=2.5, context=ctx) as resp:
                        body = resp.read()
                        if not body:
                            continue
                        data = bytes(body)
                        if data[:3] == b"\xff\xd8\xff":
                            return Response(
                                content=data,
                                media_type="image/jpeg",
                                headers={"Cache-Control": "public, max-age=3600"},
                            )
                        if data[:8] == b"\x89PNG\r\n\x1a\n":
                            return Response(
                                content=data,
                                media_type="image/png",
                                headers={"Cache-Control": "public, max-age=3600"},
                            )
                except Exception as exc:
                    last_err = str(exc)
                    if attempts >= max_tencent_attempts:
                        break
                    continue
            if attempts >= max_tencent_attempts:
                break

        # Tencent upstream can fail from some networks (TLS handshake timeout, route block).
        # Degrade gracefully to OSM vector tiles so operators still get a usable basemap.
        try:
            osm_url = f"https://tile.openstreetmap.org/{int(z)}/{int(xw)}/{int(yi)}.png"
            osm_req = urllib.request.Request(
                osm_url,
                headers={"User-Agent": "DroneThesis/1.0 (basemap-fallback)"},
            )
            with urllib.request.urlopen(osm_req, timeout=4, context=ctx) as osm_resp:
                body = osm_resp.read()
                if body and bytes(body[:8]) == b"\x89PNG\r\n\x1a\n":
                    return Response(
                        content=bytes(body),
                        media_type="image/png",
                        headers={"Cache-Control": "public, max-age=1200"},
                    )
        except Exception as exc:
            if not last_err:
                last_err = str(exc)
        raise HTTPException(status_code=404, detail=f"tile unavailable: {last_err or 'upstream empty'}")

    return app


app = create_app()
