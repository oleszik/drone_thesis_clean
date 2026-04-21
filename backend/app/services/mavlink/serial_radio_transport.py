from __future__ import annotations

import threading
import time
import importlib
from dataclasses import dataclass
from typing import Any

from ...mavlink_service import MavlinkService, MavlinkSettings


@dataclass(frozen=True)
class SerialRadioTransport:
    connection_url: str

    @property
    def transport_kind(self) -> str:
        return "serial_radio"


class RealRadioService:
    def __init__(
        self,
        *,
        reconnect_sec: float,
        heartbeat_timeout_sec: float,
        default_takeoff_alt_m: float,
        track_max_points: int,
        default_serial_port: str = "",
        default_serial_baud: int = 57600,
    ) -> None:
        self._lock = threading.Lock()
        self._reconnect_sec = float(reconnect_sec)
        self._heartbeat_timeout_sec = float(heartbeat_timeout_sec)
        self._default_takeoff_alt_m = float(default_takeoff_alt_m)
        self._track_max_points = int(track_max_points)
        self._serial_port = str(default_serial_port or "")
        self._serial_baud = int(default_serial_baud)
        self._mav = self._new_mav(self._serial_port, self._serial_baud)

    def _new_mav(self, serial_port: str, serial_baud: int) -> MavlinkService:
        conn_url = f"serial:{serial_port}:{int(serial_baud)}" if serial_port else f"serial::{int(serial_baud)}"
        return MavlinkService(
            MavlinkSettings(
                connection_url=conn_url,
                reconnect_sec=self._reconnect_sec,
                heartbeat_timeout_sec=self._heartbeat_timeout_sec,
                default_takeoff_alt_m=self._default_takeoff_alt_m,
                track_max_points=self._track_max_points,
                serial_baud=int(serial_baud),
            )
        )

    def list_serial_ports(self) -> dict[str, Any]:
        try:
            list_ports = importlib.import_module("serial.tools.list_ports")
        except Exception:
            return {"ports": [], "error_message": "pyserial is not installed"}
        out: list[dict[str, str]] = []
        try:
            for p in list_ports.comports():
                out.append(
                    {
                        "port": str(getattr(p, "device", "")),
                        "description": str(getattr(p, "description", "")),
                        "hwid": str(getattr(p, "hwid", "")),
                    }
                )
        except Exception as exc:
            return {"ports": [], "error_message": str(exc)}
        return {"ports": out, "error_message": ""}

    def connect(self, *, serial_port: str, serial_baud: int) -> dict[str, Any]:
        port = str(serial_port or "").strip()
        if not port:
            raise ValueError("serial_port is required")
        baud = max(1200, int(serial_baud))
        with self._lock:
            try:
                self._mav.stop()
            except Exception:
                pass
            self._serial_port = port
            self._serial_baud = baud
            self._mav = self._new_mav(port, baud)
        out = self._mav.connect()
        return {
            **out,
            "serial_port": self._serial_port,
            "serial_baud": self._serial_baud,
        }

    def disconnect(self) -> dict[str, Any]:
        out = self._mav.disconnect()
        return {**out, "serial_port": self._serial_port, "serial_baud": self._serial_baud}

    def telemetry(self) -> dict[str, Any]:
        return self._mav.get_telemetry()

    def get_telemetry(self) -> dict[str, Any]:
        return self._mav.get_telemetry()

    def status(self) -> dict[str, Any]:
        st = self._mav.get_status()
        tele = self._mav.get_telemetry()
        hb_age = st.get("last_heartbeat_age_s")
        updated_at = tele.get("updated_at_unix")
        telemetry_age_s: float | None = None
        if updated_at is not None:
            try:
                telemetry_age_s = max(0.0, time.time() - float(updated_at))
            except Exception:
                telemetry_age_s = None

        stale = False
        lost = not bool(st.get("connected"))
        if hb_age is not None:
            stale = stale or float(hb_age) > max(2.0, self._heartbeat_timeout_sec)
            lost = lost or float(hb_age) > max(6.0, self._heartbeat_timeout_sec * 2.0)
        if telemetry_age_s is not None:
            stale = stale or telemetry_age_s > 2.5
            lost = lost or telemetry_age_s > 6.0
        state = "healthy"
        if lost:
            state = "lost"
        elif stale:
            state = "stale"
        elif not bool(st.get("connected")):
            state = "disconnected"

        return {
            "serial_port": self._serial_port,
            "serial_baud": self._serial_baud,
            "connected": bool(st.get("connected")),
            "last_heartbeat_age_s": hb_age,
            "last_telemetry_age_s": telemetry_age_s,
            "stale": bool(stale),
            "lost": bool(lost),
            "state": state,
            "error_message": str(st.get("last_error") or ""),
            "mav_status": st,
        }

    def get_status(self) -> dict[str, Any]:
        return self._mav.get_status()

    def heartbeat_test(self) -> dict[str, Any]:
        st = self.status()
        hb = st.get("last_heartbeat_age_s")
        ok = bool(st.get("connected")) and hb is not None and float(hb) <= max(2.0, self._heartbeat_timeout_sec)
        return {
            "ok": bool(ok),
            "heartbeat_age_s": hb,
            "state": st.get("state"),
            "serial_port": st.get("serial_port"),
            "serial_baud": st.get("serial_baud"),
            "error_message": "" if ok else (st.get("error_message") or "heartbeat is stale or missing"),
        }

    def get_track(self, limit: int = 500) -> list[dict[str, Any]]:
        return self._mav.get_track(limit=limit)

    def arm(self) -> dict[str, Any]:
        return self._mav.arm()

    def disarm(self) -> dict[str, Any]:
        return self._mav.disarm()

    def takeoff(self, alt_m: float | None = None) -> dict[str, Any]:
        return self._mav.takeoff(alt_m)

    def rtl(self) -> dict[str, Any]:
        return self._mav.rtl()

    def land(self) -> dict[str, Any]:
        return self._mav.land()

    def set_mode(self, mode: str) -> dict[str, Any]:
        return self._mav.set_mode(mode)

    def start_compass_calibration(self, *, retry_on_failure: bool = True, autosave: bool = True) -> dict[str, Any]:
        return self._mav.start_compass_calibration(retry_on_failure=retry_on_failure, autosave=autosave)

    def cancel_compass_calibration(self) -> dict[str, Any]:
        return self._mav.cancel_compass_calibration()

    def set_speed(self, speed_m_s: float) -> dict[str, Any]:
        return self._mav.set_speed(speed_m_s)

    def set_home(self, lng: float, lat: float, alt_m: float | None = None) -> dict[str, Any]:
        return self._mav.set_home(lng=lng, lat=lat, alt_m=alt_m)

    def goto_location(self, lng: float, lat: float, alt_rel_m: float, yaw_deg: float | None = None) -> dict[str, Any]:
        return self._mav.goto_location(lng=lng, lat=lat, alt_rel_m=alt_rel_m, yaw_deg=yaw_deg)
