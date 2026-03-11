from __future__ import annotations

import logging
import math
import ssl
import threading
import time
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from .config import load_config
from .coverage_service import CoverageConfig, CoverageService
from .mavlink_service import MavlinkService, MavlinkSettings
from .mission_service import MissionService
from .run_manager import RunManager
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


class MissionGenerateRequest(BaseModel):
    spacing_m: float = 8.0
    speed_m_s: float = 3.0
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


def create_app() -> FastAPI:
    cfg = load_config(Path(__file__).resolve().parents[2])
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
            connection_url=cfg.mavlink_url,
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
    sitl = SitlExecutor(mav=mav, mission=mission)
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

    def _meters_to_deg_lat(m: float) -> float:
        return float(m) / 111320.0

    def _meters_to_deg_lng(m: float, lat_deg: float) -> float:
        c = math.cos(math.radians(float(lat_deg)))
        return float(m) / (111320.0 * max(0.1, abs(c)))

    def _meters_between_lng(lng1: float, lng2: float, lat_ref: float) -> float:
        return abs(float(lng2) - float(lng1)) * 111320.0 * max(0.1, abs(math.cos(math.radians(float(lat_ref)))))

    def _meters_between_lat(lat1: float, lat2: float) -> float:
        return abs(float(lat2) - float(lat1)) * 111320.0

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
            coverage.reset(
                origin_lng=center_lng,
                origin_lat=center_lat,
                bounds_w_m=max(20.0, width_m + 2.0 * margin_m),
                bounds_h_m=max(20.0, height_m + 2.0 * margin_m),
                roi_polygon_lng_lat=[[float(p[0]), float(p[1])] for p in area_poly if isinstance(p, (list, tuple)) and len(p) >= 2],
            )
            return
        if len(waypoints) >= 1:
            coverage.reset(
                origin_lng=float(waypoints[0][0]),
                origin_lat=float(waypoints[0][1]),
                roi_polygon_lng_lat=None,
            )
            return
        coverage.reset()

    def _map_origin() -> tuple[float, float]:
        return float(cfg.map_default_center_lng), float(cfg.map_default_center_lat)

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

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {
            "ok": True,
            "service": "backend",
            "map_provider": cfg.map_provider,
            "tencent_key_configured": bool(cfg.tencent_key),
        }

    @app.get("/api/status")
    def status() -> dict[str, object]:
        return mav.get_status()

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
        start_ll = mission_payload.get("start_position_lng_lat")
        if isinstance(start_ll, list) and len(start_ll) >= 2:
            origin = {"lng": float(start_ll[0]), "lat": float(start_ll[1])}
        else:
            origin = {"lng": lng, "lat": lat}
        return {
            "map_provider": cfg.map_provider,
            "tile_url_template": "/api/map/tile/{z}/{x}/{y}.png",
            "center_lng_lat": [lng, lat],
            "zoom": int(cfg.map_default_zoom),
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

    @app.get("/api/track")
    def track(limit: int = 500) -> dict[str, object]:
        items = mav.get_track(limit=limit)
        return {"count": len(items), "items": items}

    @app.get("/api/coverage")
    def get_coverage() -> dict[str, object]:
        return coverage.get_coverage()

    @app.post("/api/coverage/reset")
    def reset_coverage() -> dict[str, object]:
        _reset_coverage_for_mission()
        return {"ok": True}

    @app.get("/api/scan/debug")
    def scan_debug() -> dict[str, object]:
        return coverage.get_scan_debug()

    @app.post("/api/mission/area")
    def mission_area(req: MissionAreaRequest) -> dict[str, object]:
        try:
            out = mission.set_area(req.polygon_lng_lat)
            runs.log("MISSION_AREA_SET", {"points": out.get("points")})
            return out
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/mission/start_position")
    def mission_start_position(req: MissionStartRequest) -> dict[str, object]:
        out = mission.set_start_position(req.lng, req.lat)
        runs.log("MISSION_START_SET", {"lng": req.lng, "lat": req.lat})
        return out

    @app.post("/api/mission/generate_scan")
    def mission_generate_scan(req: MissionGenerateRequest) -> dict[str, object]:
        try:
            out = mission.generate_scan(
                spacing_m=req.spacing_m,
                speed_m_s=req.speed_m_s,
                start_scan=req.start_scan,
            )
            runs.log("MISSION_PATH_GENERATED", {"spacing_m": req.spacing_m, "speed_m_s": req.speed_m_s})
            return out
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/mission/path")
    def mission_path() -> dict[str, object]:
        return mission.get_path()

    @app.post("/api/mission/clear")
    def mission_clear() -> dict[str, object]:
        out = mission.clear()
        runs.log("MISSION_CLEARED", out)
        return out

    @app.post("/api/mission/sim/start")
    def mission_sim_start() -> dict[str, object]:
        try:
            return mission.sim_start()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/mission/sim/pause")
    def mission_sim_pause() -> dict[str, object]:
        return mission.sim_pause()

    @app.post("/api/mission/sim/stop")
    def mission_sim_stop() -> dict[str, object]:
        return mission.sim_stop()

    @app.post("/api/mission/sim/tick")
    def mission_sim_tick(req: SimTickRequest) -> dict[str, object]:
        return mission.step(req.dt)

    @app.get("/api/mission/sim/state")
    def mission_sim_state() -> dict[str, object]:
        return mission.get_sim_state()

    @app.post("/api/sitl/start_scan")
    def sitl_start_scan(req: SitlStartRequest) -> dict[str, object]:
        try:
            _reset_coverage_for_mission()
            out = sitl.start_scan(alt_m=req.alt_m, accept_radius_m=req.accept_radius_m)
            runs.log("SITL_START_SCAN", {"alt_m": req.alt_m, "accept_radius_m": req.accept_radius_m})
            return out
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.post("/api/sitl/stop_scan")
    def sitl_stop_scan() -> dict[str, object]:
        out = sitl.stop_scan()
        runs.log("SITL_STOP_SCAN", out)
        return out

    @app.get("/api/sitl/state")
    def sitl_state() -> dict[str, object]:
        return sitl.get_state()

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
    def map_tile(z: int, x: int, y: int) -> Response:
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
        ctx = ssl.create_default_context()
        last_err: str | None = None
        for host in hosts:
            url = (
                f"https://{host}.map.gtimg.com/realtimerender"
                f"?z={int(z)}&x={int(xw)}&y={int(yi)}&type=vector&style=0&key={cfg.tencent_key}"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            try:
                with urllib.request.urlopen(req, timeout=6, context=ctx) as resp:
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
                continue
        raise HTTPException(status_code=404, detail=f"tile unavailable: {last_err or 'upstream empty'}")

    return app


app = create_app()
