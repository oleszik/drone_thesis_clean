from __future__ import annotations

import math
import threading
import time
from typing import Any

from .mavlink_service import MavlinkService
from .mission_service import MissionService


def _distance_m(lng1: float, lat1: float, lng2: float, lat2: float) -> float:
    d_lat = (float(lat2) - float(lat1)) * 111320.0
    c = math.cos(math.radians((float(lat1) + float(lat2)) * 0.5))
    d_lng = (float(lng2) - float(lng1)) * 111320.0 * max(0.1, abs(c))
    return math.hypot(d_lng, d_lat)


class SitlExecutor:
    STATES = {"IDLE", "ARMING", "TAKEOFF", "RUN_PATH", "COMPLETE", "STOPPED", "ERROR"}
    MAX_START_DISTANCE_M = 5000.0

    def __init__(self, mav: MavlinkService, mission: MissionService) -> None:
        self._mav = mav
        self._mission = mission
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()

        self._state = "IDLE"
        self._scan_active = False
        self._last_error = ""
        self._alt_m = 10.0
        self._accept_radius_m = 3.0
        self._speed_m_s = 3.0
        self._waypoints: list[list[float]] = []
        self._waypoint_meta: list[dict[str, Any]] = []
        self._wp_idx = 0

    def _set_state(self, state: str) -> None:
        if state not in self.STATES:
            state = "ERROR"
        self._state = state

    def _wait_until(self, pred, timeout_s: float, step_s: float = 0.2) -> bool:
        t0 = time.monotonic()
        while not self._stop_evt.is_set():
            if pred():
                return True
            if (time.monotonic() - t0) >= timeout_s:
                return False
            time.sleep(step_s)
        return False

    def start_scan(self, alt_m: float, accept_radius_m: float) -> dict[str, Any]:
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise ValueError("scan executor already running")
            if not bool(self._mav.get_status().get("connected")):
                raise ValueError("MAVLink not connected")
            payload = self._mission.get_path()
            waypoints = payload.get("waypoints_lng_lat") or []
            waypoint_meta = payload.get("waypoint_meta") or []
            cfg = payload.get("config") or {}
            if len(waypoints) < 2:
                raise ValueError("no generated scan path")
            tele = self._mav.get_telemetry()
            cur_lng = tele.get("lon")
            cur_lat = tele.get("lat")
            if cur_lng is not None and cur_lat is not None:
                start_lng = float(waypoints[0][0])
                start_lat = float(waypoints[0][1])
                d0 = _distance_m(float(cur_lng), float(cur_lat), start_lng, start_lat)
                if d0 > self.MAX_START_DISTANCE_M:
                    raise ValueError(
                        f"vehicle too far from mission start ({d0:.0f} m). "
                        "Move the vehicle or update the mission start near the current position."
                    )
            self._waypoints = [list(w) for w in waypoints]
            self._waypoint_meta = [dict(item) for item in waypoint_meta]
            initial_alt = alt_m
            if self._waypoint_meta:
                meta_alt = self._waypoint_meta[0].get("altitude_m")
                if meta_alt is not None:
                    try:
                        initial_alt = float(meta_alt)
                    except Exception:
                        initial_alt = alt_m
            self._alt_m = max(1.0, float(initial_alt))
            self._accept_radius_m = max(0.5, float(accept_radius_m))
            self._speed_m_s = max(0.2, float(cfg.get("speed_m_s") or 3.0))
            self._last_error = ""
            # Start from the explicit mission start waypoint so ArduPilot
            # execution respects the start point selected in the dashboard.
            self._wp_idx = 0
            self._scan_active = True
            self._set_state("ARMING")
            self._stop_evt.clear()
            self._thread = threading.Thread(target=self._run, name="sitl-executor", daemon=True)
            self._thread.start()
        return self.get_state()

    def stop_scan(self) -> dict[str, Any]:
        with self._lock:
            self._stop_evt.set()
        try:
            self._mav.rtl()
        except Exception:
            try:
                self._mav.land()
            except Exception:
                pass
        with self._lock:
            self._scan_active = False
            self._set_state("STOPPED")
        return self.get_state()

    def is_scan_active(self) -> bool:
        with self._lock:
            return bool(self._scan_active and self._state == "RUN_PATH")

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "scan_active": bool(self._scan_active),
                "waypoint_index": int(self._wp_idx),
                "waypoint_count": int(len(self._waypoints)),
                "alt_m": float(self._alt_m),
                "accept_radius_m": float(self._accept_radius_m),
                "speed_m_s": float(self._speed_m_s),
                "last_error": self._last_error,
            }

    def _run(self) -> None:
        try:
            start_lng = float(self._waypoints[0][0])
            start_lat = float(self._waypoints[0][1])
            tele0 = self._mav.get_telemetry()
            home_alt = tele0.get("alt_m")
            if home_alt is None and tele0.get("rel_alt_m") is not None:
                home_alt = float(tele0.get("rel_alt_m") or 0.0)
            try:
                self._mav.set_home(lng=start_lng, lat=start_lat, alt_m=home_alt)
            except Exception:
                # SITL start still works without a home update; this is best-effort.
                pass

            with self._lock:
                self._set_state("ARMING")
            try:
                self._mav.set_mode("GUIDED")
            except Exception as exc:
                raise RuntimeError(f"guided mode switch failed: {exc}") from exc

            try:
                self._mav.arm()
            except Exception as exc:
                raise RuntimeError(f"arming failed: {exc}") from exc

            with self._lock:
                self._set_state("TAKEOFF")

            try:
                self._mav.takeoff(self._alt_m)
            except Exception as exc:
                raise RuntimeError(f"takeoff failed: {exc}") from exc

            try:
                self._mav.set_speed(self._speed_m_s)
            except Exception:
                pass

            with self._lock:
                self._set_state("RUN_PATH")

            rate_hz = 5.0
            dt = 1.0 / rate_hz
            while not self._stop_evt.is_set():
                with self._lock:
                    if self._wp_idx >= len(self._waypoints):
                        break
                    tgt = self._waypoints[self._wp_idx]
                    meta = self._waypoint_meta[self._wp_idx] if self._wp_idx < len(self._waypoint_meta) else {}
                    accept_r = self._accept_radius_m
                    alt = max(1.0, float(meta.get("altitude_m") or self._alt_m))

                lng_t, lat_t = float(tgt[0]), float(tgt[1])
                yaw_deg = meta.get("yaw_deg")
                try:
                    self._mav.goto_location(
                        lng=lng_t,
                        lat=lat_t,
                        alt_rel_m=alt,
                        yaw_deg=(float(yaw_deg) if yaw_deg is not None else None),
                    )
                except Exception as exc:
                    if self._wp_idx == 0:
                        raise RuntimeError(f"first waypoint execution failed: {exc}") from exc
                    raise RuntimeError(f"waypoint execution failed (index={self._wp_idx}): {exc}") from exc

                tele = self._mav.get_telemetry()
                lng = tele.get("lon")
                lat = tele.get("lat")
                if lng is not None and lat is not None:
                    d = _distance_m(float(lng), float(lat), lng_t, lat_t)
                    if d <= accept_r:
                        with self._lock:
                            self._wp_idx += 1
                time.sleep(dt)

            if not self._stop_evt.is_set():
                with self._lock:
                    self._set_state("COMPLETE")
                    self._scan_active = False
                try:
                    self._mav.land()
                except Exception:
                    try:
                        self._mav.rtl()
                    except Exception:
                        pass
            else:
                with self._lock:
                    self._scan_active = False
                    self._set_state("STOPPED")
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
                self._scan_active = False
                self._set_state("ERROR")
            try:
                self._mav.rtl()
            except Exception:
                pass
