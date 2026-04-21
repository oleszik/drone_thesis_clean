from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

try:
    from pymavlink import mavutil
except Exception:  # pragma: no cover
    mavutil = None


@dataclass(frozen=True)
class MavlinkSettings:
    connection_url: str
    reconnect_sec: float
    heartbeat_timeout_sec: float
    default_takeoff_alt_m: float
    track_max_points: int = 2000
    serial_baud: int | None = None


class MavlinkService:
    def __init__(self, settings: MavlinkSettings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._master: Any = None
        self._last_heartbeat_ts: float | None = None

        self._status: dict[str, Any] = {
            "connected": False,
            "armed": False,
            "mode": "UNKNOWN",
            "failsafes": {"gps_ok": False, "ekf_ok": False, "battery_low": False},
            "compass_calibration": self._new_compass_calibration_status(),
            "connection_url": settings.connection_url,
            "last_error": "",
            "last_heartbeat_age_s": None,
        }
        self._telemetry: dict[str, Any] = {
            "lat": None,
            "lon": None,
            "alt_m": None,
            "rel_alt_m": None,
            "speed_m_s": None,
            "vx_m_s": None,
            "vy_m_s": None,
            "vz_m_s": None,
            "yaw_deg": None,
            "roll_deg": None,
            "pitch_deg": None,
            "battery_percent": None,
            "gps_fix": None,
            "satellites": None,
            "ekf_ok": None,
            "updated_at_unix": None,
        }
        self._track: deque[dict[str, Any]] = deque(maxlen=max(100, int(settings.track_max_points)))

    @staticmethod
    def _new_compass_calibration_status() -> dict[str, Any]:
        return {
            "state": "idle",
            "message": "idle",
            "completion_pct": None,
            "attempt": None,
            "compass_id": None,
            "cal_status": None,
            "cal_status_label": None,
            "started_at_unix": None,
            "updated_at_unix": None,
            "report": None,
        }

    @staticmethod
    def _safe_int(v: Any) -> int | None:
        try:
            return int(v)
        except Exception:
            return None

    @staticmethod
    def _safe_float(v: Any) -> float | None:
        try:
            out = float(v)
        except Exception:
            return None
        if not math.isfinite(out):
            return None
        return out

    @staticmethod
    def _command_result_label(result_code: int | None) -> str:
        if result_code is None:
            return "unknown"
        if mavutil is not None:
            try:
                enum_tbl = mavutil.mavlink.enums.get("MAV_RESULT")
                if isinstance(enum_tbl, dict):
                    enum_value = enum_tbl.get(int(result_code))
                    name = str(getattr(enum_value, "name", "") or "").strip()
                    if name:
                        return name.lower()
            except Exception:
                pass
        return f"result_{int(result_code)}"

    @staticmethod
    def _mag_cal_status_label(status_code: int | None) -> str:
        if status_code is None:
            return "unknown"
        if mavutil is not None:
            try:
                enum_tbl = mavutil.mavlink.enums.get("MAV_MAG_CAL_STATUS")
                if isinstance(enum_tbl, dict):
                    enum_value = enum_tbl.get(int(status_code))
                    name = str(getattr(enum_value, "name", "") or "").strip()
                    if name.startswith("MAV_MAG_CAL_"):
                        name = name[len("MAV_MAG_CAL_") :]
                    if name:
                        return name.lower()
            except Exception:
                pass
        fallback = {
            0: "not_started",
            1: "waiting_to_start",
            2: "running_step_one",
            3: "running_step_two",
            4: "success",
            5: "failed",
            6: "bad_orientation",
            7: "bad_radius",
        }
        return fallback.get(int(status_code), f"status_{int(status_code)}")

    @staticmethod
    def _mag_cal_success_status_code() -> int:
        if mavutil is None:
            return 4
        return int(getattr(mavutil.mavlink, "MAV_MAG_CAL_SUCCESS", 4))

    @staticmethod
    def _mag_cal_failure_status_codes() -> set[int]:
        fallback = {5, 6, 7}
        if mavutil is None:
            return fallback
        out = set()
        for name in ("MAV_MAG_CAL_FAILED", "MAV_MAG_CAL_BAD_ORIENTATION", "MAV_MAG_CAL_BAD_RADIUS"):
            val = getattr(mavutil.mavlink, name, None)
            if isinstance(val, int):
                out.add(int(val))
        return out or fallback

    @staticmethod
    def _mag_cal_start_cmd() -> int:
        if mavutil is None:
            return 42424
        return int(getattr(mavutil.mavlink, "MAV_CMD_DO_START_MAG_CAL", 42424))

    @staticmethod
    def _mag_cal_cancel_cmd() -> int:
        if mavutil is None:
            return 42425
        return int(getattr(mavutil.mavlink, "MAV_CMD_DO_CANCEL_MAG_CAL", 42425))

    def _set_compass_calibration(self, **updates: Any) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            cur = self._status.get("compass_calibration")
            if not isinstance(cur, dict):
                cur = self._new_compass_calibration_status()
            cur = {**cur, **updates, "updated_at_unix": now}
            self._status["compass_calibration"] = cur
            return dict(cur)

    @staticmethod
    def _is_autopilot_heartbeat(msg: Any) -> bool:
        if msg is None or msg.get_type() != "HEARTBEAT":
            return False
        try:
            hb_type = int(getattr(msg, "type", -1))
            autopilot = int(getattr(msg, "autopilot", -1))
        except Exception:
            return False
        if hb_type == mavutil.mavlink.MAV_TYPE_GCS:
            return False
        return autopilot != mavutil.mavlink.MAV_AUTOPILOT_INVALID

    def _wait_for_autopilot_heartbeat(self, master: Any, timeout_s: float) -> Any:
        deadline = time.monotonic() + max(0.5, float(timeout_s))
        while time.monotonic() < deadline:
            remain = max(0.1, deadline - time.monotonic())
            msg = master.recv_match(type="HEARTBEAT", blocking=True, timeout=remain)
            if msg is None:
                continue
            if self._is_autopilot_heartbeat(msg):
                return msg
        raise TimeoutError("no autopilot heartbeat received")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, name="mavlink-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._close_master()

    def connect(self) -> dict[str, Any]:
        self.start()
        return {"ok": True, "action": "connect"}

    def disconnect(self) -> dict[str, Any]:
        self.stop()
        return {"ok": True, "action": "disconnect"}

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            out = dict(self._status)
            if self._last_heartbeat_ts is not None and out.get("connected"):
                out["last_heartbeat_age_s"] = max(0.0, time.monotonic() - self._last_heartbeat_ts)
            return out

    def get_telemetry(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._telemetry)

    def get_track(self, limit: int = 500) -> list[dict[str, Any]]:
        n = max(1, min(int(limit), self._track.maxlen))
        with self._lock:
            if n >= len(self._track):
                return list(self._track)
            return list(self._track)[-n:]

    def arm(self) -> dict[str, Any]:
        master = self._require_master()
        master.arducopter_arm()
        return {"ok": True, "action": "arm"}

    def disarm(self) -> dict[str, Any]:
        master = self._require_master()
        master.arducopter_disarm()
        return {"ok": True, "action": "disarm"}

    def set_mode(self, mode: str) -> dict[str, Any]:
        mode_name = str(mode or "").upper().strip()
        if not mode_name:
            raise RuntimeError("mode is required")
        master = self._require_master()
        mapping = master.mode_mapping() or {}
        if mode_name not in mapping:
            raise RuntimeError(f"unsupported mode: {mode_name}")
        master.set_mode(mode_name)
        return {"ok": True, "action": "set_mode", "mode": mode_name}

    def takeoff(self, alt_m: float | None = None) -> dict[str, Any]:
        master = self._require_master()
        alt = float(alt_m if alt_m is not None else self._settings.default_takeoff_alt_m)
        alt = max(1.0, alt)
        # Use GUIDED for copter takeoff command.
        self.set_mode("GUIDED")
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            alt,
        )
        return {"ok": True, "action": "takeoff", "alt_m": alt}

    def land(self) -> dict[str, Any]:
        self.set_mode("LAND")
        return {"ok": True, "action": "land"}

    def rtl(self) -> dict[str, Any]:
        self.set_mode("RTL")
        return {"ok": True, "action": "rtl"}

    def start_compass_calibration(self, *, retry_on_failure: bool = True, autosave: bool = True) -> dict[str, Any]:
        if mavutil is None:
            raise RuntimeError("pymavlink is not installed")
        st = self.get_status()
        if bool(st.get("armed")):
            raise RuntimeError("vehicle must be disarmed before compass calibration")
        master = self._require_master()
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            self._mag_cal_start_cmd(),
            0,
            0,  # all available compasses
            1 if bool(retry_on_failure) else 0,
            1 if bool(autosave) else 0,
            0,  # no start delay
            0,  # no auto reboot
            0,
            0,
        )
        cal = self._set_compass_calibration(
            state="starting",
            message="start command sent; keep disarmed and rotate vehicle on all axes",
            completion_pct=0,
            attempt=1,
            started_at_unix=time.time(),
            report=None,
        )
        return {
            "ok": True,
            "action": "start_compass_calibration",
            "retry_on_failure": bool(retry_on_failure),
            "autosave": bool(autosave),
            "compass_calibration": cal,
        }

    def cancel_compass_calibration(self) -> dict[str, Any]:
        if mavutil is None:
            raise RuntimeError("pymavlink is not installed")
        master = self._require_master()
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            self._mag_cal_cancel_cmd(),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        cal = self._set_compass_calibration(
            state="cancel_requested",
            message="cancel command sent",
        )
        return {
            "ok": True,
            "action": "cancel_compass_calibration",
            "compass_calibration": cal,
        }

    def battery_reset(self) -> dict[str, Any]:
        if mavutil is None:
            raise RuntimeError("pymavlink is not installed")
        master = self._require_master()
        # In SITL this reboots the autopilot process state, which resets simulated battery.
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
            0,
            1,  # reboot autopilot
            0,
            0,
            0,
            0,
            0,
            0,
        )
        # Force a reconnect cycle so status/UI updates promptly after reboot.
        self._close_master()
        return {"ok": True, "action": "battery_reset", "method": "autopilot_reboot"}

    def set_speed(self, speed_m_s: float) -> dict[str, Any]:
        if mavutil is None:
            raise RuntimeError("pymavlink is not installed")
        master = self._require_master()
        speed = max(0.2, float(speed_m_s))
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
            0,
            1,
            speed,
            -1,
            0,
            0,
            0,
            0,
        )
        return {"ok": True, "action": "set_speed", "speed_m_s": speed}

    def set_yaw(self, yaw_deg: float, yaw_rate_deg_s: float = 30.0, clockwise: bool | None = None) -> dict[str, Any]:
        if mavutil is None:
            raise RuntimeError("pymavlink is not installed")
        master = self._require_master()
        yaw = float(yaw_deg) % 360.0
        yaw_rate = max(1.0, float(yaw_rate_deg_s))
        direction = 0
        if clockwise is True:
            direction = 1
        elif clockwise is False:
            direction = -1
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            0,
            yaw,
            yaw_rate,
            direction,
            0,
            0,
            0,
            0,
        )
        return {"ok": True, "action": "set_yaw", "yaw_deg": yaw, "yaw_rate_deg_s": yaw_rate}

    def set_home(self, lng: float, lat: float, alt_m: float | None = None) -> dict[str, Any]:
        if mavutil is None:
            raise RuntimeError("pymavlink is not installed")
        master = self._require_master()
        alt = max(0.0, float(alt_m or 0.0))
        master.mav.command_int_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL,
            mavutil.mavlink.MAV_CMD_DO_SET_HOME,
            0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            int(float(lat) * 1e7),
            int(float(lng) * 1e7),
            alt,
        )
        return {"ok": True, "action": "set_home", "lng": float(lng), "lat": float(lat), "alt_m": alt}

    def goto_location(self, lng: float, lat: float, alt_rel_m: float, yaw_deg: float | None = None) -> dict[str, Any]:
        if mavutil is None:
            raise RuntimeError("pymavlink is not installed")
        master = self._require_master()
        lat_i = int(float(lat) * 1e7)
        lon_i = int(float(lng) * 1e7)
        alt = max(0.0, float(alt_rel_m))
        # Position control only: ignore velocity/accel/yaw fields.
        type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        )
        yaw_rad = 0.0
        if yaw_deg is None:
            type_mask |= mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
        else:
            yaw_rad = math.radians(float(yaw_deg) % 360.0)
        master.mav.set_position_target_global_int_send(
            0,
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            type_mask,
            lat_i,
            lon_i,
            alt,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            yaw_rad,
            0.0,
        )
        return {
            "ok": True,
            "action": "goto",
            "lng": float(lng),
            "lat": float(lat),
            "alt_rel_m": alt,
            "yaw_deg": None if yaw_deg is None else float(yaw_deg) % 360.0,
        }

    def _require_master(self) -> Any:
        with self._lock:
            master = self._master
            connected = bool(self._status.get("connected"))
        if not connected or master is None:
            raise RuntimeError("MAVLink not connected")
        return master

    def _close_master(self) -> None:
        with self._lock:
            master = self._master
            self._master = None
            self._status["connected"] = False
            cur = self._status.get("compass_calibration")
            if isinstance(cur, dict):
                state = str(cur.get("state") or "").strip().lower()
                if state in {"starting", "running", "waiting_to_start", "cancel_requested"}:
                    self._status["compass_calibration"] = {
                        **cur,
                        "state": "link_lost",
                        "message": "mavlink disconnected during compass calibration",
                        "updated_at_unix": time.time(),
                    }
        if master is None:
            return
        try:
            master.close()
        except Exception:
            pass

    def _run(self) -> None:
        if mavutil is None:
            with self._lock:
                self._status["last_error"] = "pymavlink is not installed"
            return

        while not self._stop_evt.is_set():
            if not self._connect():
                time.sleep(self._settings.reconnect_sec)
                continue

            while not self._stop_evt.is_set():
                master = self._master
                if master is None:
                    break
                try:
                    msg = master.recv_match(blocking=True, timeout=0.5)
                except Exception as exc:
                    with self._lock:
                        self._status["last_error"] = f"recv failed: {exc}"
                    self._close_master()
                    break
                if msg is None:
                    if self._heartbeat_stale():
                        with self._lock:
                            self._status["last_error"] = "heartbeat timeout"
                        self._close_master()
                        break
                    continue
                self._handle_msg(msg)

            time.sleep(self._settings.reconnect_sec)

    def _connect(self) -> bool:
        conn_url = self._settings.connection_url
        kwargs: dict[str, Any] = {"autoreconnect": True}
        if str(conn_url).startswith("udp:"):
            # For SITL local telemetry, we need input socket semantics.
            conn_url = "udpin:" + str(conn_url)[len("udp:") :]
        elif str(conn_url).startswith("serial:"):
            # Format: serial:<port>:<baud>
            raw = str(conn_url)[len("serial:") :]
            port = raw
            baud = self._settings.serial_baud
            if ":" in raw:
                maybe_port, maybe_baud = raw.rsplit(":", 1)
                if maybe_port:
                    port = maybe_port
                try:
                    baud = int(maybe_baud)
                except Exception:
                    pass
            conn_url = port
            if baud is not None:
                kwargs["baud"] = int(baud)
        try:
            master = mavutil.mavlink_connection(conn_url, **kwargs)
            self._wait_for_autopilot_heartbeat(master, self._settings.heartbeat_timeout_sec)
            self._request_data_streams(master)
        except Exception as exc:
            with self._lock:
                self._status["connected"] = False
                self._status["last_error"] = f"connect failed: {exc}"
            return False

        with self._lock:
            self._master = master
            self._last_heartbeat_ts = time.monotonic()
            self._status["connected"] = True
            self._status["connection_url"] = str(conn_url)
            self._status["last_error"] = ""
            self._status["compass_calibration"] = self._new_compass_calibration_status()
        return True

    def _request_data_streams(self, master: Any) -> None:
        # Ask autopilot to push standard telemetry groups.
        try:
            master.mav.request_data_stream_send(
                master.target_system,
                master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                10,
                1,
            )
        except Exception:
            pass

        def _msg_interval(msg_id: int, hz: float) -> None:
            try:
                interval_us = int(1_000_000 / max(0.1, float(hz)))
                master.mav.command_long_send(
                    master.target_system,
                    master.target_component,
                    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                    0,
                    msg_id,
                    interval_us,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
            except Exception:
                return

        _msg_interval(mavutil.mavlink.MAVLINK_MSG_ID_HEARTBEAT, 1.0)
        _msg_interval(mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 10.0)
        _msg_interval(mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 10.0)
        _msg_interval(mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS, 2.0)
        _msg_interval(mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT, 2.0)
        _msg_interval(mavutil.mavlink.MAVLINK_MSG_ID_EKF_STATUS_REPORT, 2.0)

    def _heartbeat_stale(self) -> bool:
        with self._lock:
            hb = self._last_heartbeat_ts
        if hb is None:
            return True
        return (time.monotonic() - hb) > (self._settings.heartbeat_timeout_sec * 2.0)

    def _handle_msg(self, msg: Any) -> None:
        msg_type = msg.get_type()
        now = time.time()

        if msg_type == "HEARTBEAT":
            if not self._is_autopilot_heartbeat(msg):
                return
            armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            mode_name = mavutil.mode_string_v10(msg)
            with self._lock:
                self._last_heartbeat_ts = time.monotonic()
                self._status["armed"] = armed
                self._status["mode"] = mode_name
                self._status["connected"] = True
            return

        if msg_type == "COMMAND_ACK":
            command = self._safe_int(getattr(msg, "command", None))
            result = self._safe_int(getattr(msg, "result", None))
            if command in {self._mag_cal_start_cmd(), self._mag_cal_cancel_cmd()}:
                result_label = self._command_result_label(result)
                accepted = False
                if mavutil is not None:
                    accepted = result in {
                        int(getattr(mavutil.mavlink, "MAV_RESULT_ACCEPTED", 0)),
                        int(getattr(mavutil.mavlink, "MAV_RESULT_IN_PROGRESS", 5)),
                    }
                else:
                    accepted = result in {0, 5}
                if command == self._mag_cal_start_cmd() and not accepted:
                    self._set_compass_calibration(
                        state="failed",
                        message=f"start rejected: {result_label}",
                        cal_status=result,
                        cal_status_label=result_label,
                    )
                if command == self._mag_cal_cancel_cmd():
                    self._set_compass_calibration(
                        state="cancelled" if accepted else "cancel_failed",
                        message=("calibration cancelled" if accepted else f"cancel rejected: {result_label}"),
                        cal_status=result,
                        cal_status_label=result_label,
                    )
            return

        if msg_type == "MAG_CAL_PROGRESS":
            completion_pct = self._safe_int(getattr(msg, "completion_pct", None))
            cal_status = self._safe_int(getattr(msg, "cal_status", None))
            cal_status_label = self._mag_cal_status_label(cal_status)
            step_state = "waiting_to_start" if "waiting" in cal_status_label else "running"
            message = (
                "waiting to start compass calibration"
                if step_state == "waiting_to_start"
                else f"compass calibration running ({completion_pct if completion_pct is not None else '--'}%)"
            )
            current = self._set_compass_calibration(
                state=step_state,
                message=message,
                completion_pct=completion_pct,
                attempt=self._safe_int(getattr(msg, "attempt", None)),
                compass_id=self._safe_int(getattr(msg, "compass_id", None)),
                cal_status=cal_status,
                cal_status_label=cal_status_label,
            )
            if current.get("started_at_unix") is None:
                self._set_compass_calibration(started_at_unix=time.time())
            return

        if msg_type == "MAG_CAL_REPORT":
            cal_status = self._safe_int(getattr(msg, "cal_status", None))
            cal_status_label = self._mag_cal_status_label(cal_status)
            success = cal_status == self._mag_cal_success_status_code()
            failure = cal_status in self._mag_cal_failure_status_codes()
            state = "succeeded" if success else ("failed" if failure else "reported")
            report = {
                "compass_id": self._safe_int(getattr(msg, "compass_id", None)),
                "cal_status": cal_status,
                "cal_status_label": cal_status_label,
                "autosaved": bool(self._safe_int(getattr(msg, "autosaved", None)) or 0),
                "fitness": self._safe_float(getattr(msg, "fitness", None)),
                "ofs_x": self._safe_float(getattr(msg, "ofs_x", None)),
                "ofs_y": self._safe_float(getattr(msg, "ofs_y", None)),
                "ofs_z": self._safe_float(getattr(msg, "ofs_z", None)),
                "diag_x": self._safe_float(getattr(msg, "diag_x", None)),
                "diag_y": self._safe_float(getattr(msg, "diag_y", None)),
                "diag_z": self._safe_float(getattr(msg, "diag_z", None)),
            }
            message = (
                "compass calibration complete and saved"
                if success
                else f"compass calibration ended with status {cal_status_label}"
            )
            self._set_compass_calibration(
                state=state,
                message=message,
                completion_pct=100 if success else self._safe_int(getattr(msg, "completion_pct", None)),
                compass_id=report["compass_id"],
                cal_status=cal_status,
                cal_status_label=cal_status_label,
                report=report,
            )
            return

        if msg_type == "GLOBAL_POSITION_INT":
            lat = float(msg.lat) / 1e7
            lon = float(msg.lon) / 1e7
            alt_m = float(msg.alt) / 1000.0
            rel_alt_m = float(msg.relative_alt) / 1000.0
            vx = float(msg.vx) / 100.0
            vy = float(msg.vy) / 100.0
            vz = float(msg.vz) / 100.0
            speed = math.sqrt(vx * vx + vy * vy)
            with self._lock:
                self._telemetry.update(
                    {
                        "lat": lat,
                        "lon": lon,
                        "alt_m": alt_m,
                        "rel_alt_m": rel_alt_m,
                        "vx_m_s": vx,
                        "vy_m_s": vy,
                        "vz_m_s": vz,
                        "speed_m_s": speed,
                        "updated_at_unix": now,
                    }
                )
                self._track.append(
                    {
                        "t_unix": now,
                        "lat": lat,
                        "lon": lon,
                        "rel_alt_m": rel_alt_m,
                        "speed_m_s": speed,
                    }
                )
            return

        if msg_type == "ATTITUDE":
            roll = math.degrees(float(msg.roll))
            pitch = math.degrees(float(msg.pitch))
            yaw = math.degrees(float(msg.yaw))
            with self._lock:
                self._telemetry["roll_deg"] = roll
                self._telemetry["pitch_deg"] = pitch
                self._telemetry["yaw_deg"] = yaw
                self._telemetry["updated_at_unix"] = now
            return

        if msg_type == "SYS_STATUS":
            battery = float(msg.battery_remaining)
            with self._lock:
                self._telemetry["battery_percent"] = battery if battery >= 0 else None
                self._telemetry["updated_at_unix"] = now
                self._status["failsafes"]["battery_low"] = bool(battery >= 0 and battery < 20.0)
            return

        if msg_type == "GPS_RAW_INT":
            gps_fix = int(msg.fix_type)
            sats = int(msg.satellites_visible)
            with self._lock:
                self._telemetry["gps_fix"] = gps_fix
                self._telemetry["satellites"] = sats
                self._telemetry["updated_at_unix"] = now
                self._status["failsafes"]["gps_ok"] = gps_fix >= 3
            return

        if msg_type == "EKF_STATUS_REPORT":
            flags = int(msg.flags)
            with self._lock:
                self._telemetry["ekf_ok"] = bool(flags != 0)
                self._telemetry["updated_at_unix"] = now
                self._status["failsafes"]["ekf_ok"] = bool(flags != 0)
