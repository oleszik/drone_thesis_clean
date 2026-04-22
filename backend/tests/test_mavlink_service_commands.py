from __future__ import annotations

import threading

import pytest

import backend.app.mavlink_service as mav_module
from backend.app.mavlink_service import MavlinkService, MavlinkSettings


class _FakeMav:
    def __init__(self) -> None:
        self.command_long_calls: list[tuple[object, ...]] = []

    def command_long_send(self, *args) -> None:
        self.command_long_calls.append(args)


class _FakeMaster:
    def __init__(self) -> None:
        self.target_system = 1
        self.target_component = 1
        self.mav = _FakeMav()
        self.last_set_mode = ""
        self.arm_calls = 0
        self.disarm_calls = 0

    def arducopter_arm(self) -> None:
        self.arm_calls += 1

    def arducopter_disarm(self) -> None:
        self.disarm_calls += 1

    def mode_mapping(self) -> dict[str, int]:
        return {
            "STABILIZE": 0,
            "GUIDED": 4,
            "LOITER": 5,
            "RTL": 6,
            "LAND": 9,
        }

    def set_mode(self, mode: str) -> None:
        self.last_set_mode = str(mode)


class _FakeCommandAckMsg:
    def __init__(self, command: int, result: int) -> None:
        self.command = int(command)
        self.result = int(result)

    def get_type(self) -> str:
        return "COMMAND_ACK"


def _build_service() -> tuple[MavlinkService, _FakeMaster]:
    svc = MavlinkService(
        MavlinkSettings(
            connection_url="udp:127.0.0.1:14550",
            reconnect_sec=1.0,
            heartbeat_timeout_sec=2.0,
            default_takeoff_alt_m=3.0,
        )
    )
    fake_master = _FakeMaster()
    with svc._lock:
        svc._master = fake_master
        svc._status["connected"] = True
        svc._status["armed"] = False
        svc._status["mode"] = "STABILIZE"
    return svc, fake_master


def test_arm_success_with_state_transition() -> None:
    svc, _ = _build_service()

    def _arm_transition() -> None:
        with svc._lock:
            svc._status["armed"] = True

    threading.Timer(0.05, _arm_transition).start()
    out = svc.arm(timeout_s=0.6)
    assert out["ok"] is True
    assert out["action"] == "arm"
    assert out["armed"] is True


def test_arm_timeout_failure_includes_reason() -> None:
    svc, _ = _build_service()
    with svc._lock:
        svc._status["last_status_text"] = "[warning] prearm: GPS fix required"

    with pytest.raises(RuntimeError, match="arm failed or was rejected by autopilot"):
        svc.arm(timeout_s=0.1)


def test_set_mode_success_after_reported_mode_change() -> None:
    svc, master = _build_service()

    def _mode_transition() -> None:
        with svc._lock:
            svc._status["mode"] = "GUIDED"

    threading.Timer(0.05, _mode_transition).start()
    out = svc.set_mode("GUIDED", timeout_s=0.6)
    assert out["ok"] is True
    assert master.last_set_mode == "GUIDED"
    assert out["resulting_mode"] == "GUIDED"


def test_set_mode_timeout_failure() -> None:
    svc, _ = _build_service()
    with svc._lock:
        svc._status["last_status_text"] = "[error] mode change rejected"

    with pytest.raises(RuntimeError, match="requested=LOITER"):
        svc.set_mode("LOITER", timeout_s=0.1)


def test_takeoff_success_after_altitude_rise(monkeypatch: pytest.MonkeyPatch) -> None:
    svc, master = _build_service()
    with svc._lock:
        svc._telemetry["rel_alt_m"] = 0.0

    class _FakeMavlinkConsts:
        MAV_CMD_NAV_TAKEOFF = 22

    class _FakeMavutil:
        mavlink = _FakeMavlinkConsts()

    monkeypatch.setattr(mav_module, "mavutil", _FakeMavutil())

    def _mode_transition() -> None:
        with svc._lock:
            svc._status["mode"] = "GUIDED"

    def _altitude_transition() -> None:
        with svc._lock:
            svc._telemetry["rel_alt_m"] = 1.3

    threading.Timer(0.03, _mode_transition).start()
    threading.Timer(0.35, _altitude_transition).start()

    out = svc.takeoff(alt_m=3.0, rise_timeout_s=1.0)
    assert out["ok"] is True
    assert out["action"] == "takeoff"
    assert master.mav.command_long_calls
    sent_command = master.mav.command_long_calls[0][2]
    assert int(sent_command) == 22


def test_takeoff_failure_when_altitude_does_not_rise(monkeypatch: pytest.MonkeyPatch) -> None:
    svc, _ = _build_service()
    with svc._lock:
        svc._telemetry["rel_alt_m"] = 0.0

    class _FakeMavlinkConsts:
        MAV_CMD_NAV_TAKEOFF = 22

    class _FakeMavutil:
        mavlink = _FakeMavlinkConsts()

    monkeypatch.setattr(mav_module, "mavutil", _FakeMavutil())

    def _mode_transition() -> None:
        with svc._lock:
            svc._status["mode"] = "GUIDED"
            svc._status["last_status_text"] = "[warning] takeoff denied: throttle failsafe"

    threading.Timer(0.03, _mode_transition).start()

    with pytest.raises(RuntimeError, match="no altitude rise observed"):
        svc.takeoff(alt_m=3.0, rise_timeout_s=0.25)


def test_level_calibration_success_after_ack(monkeypatch: pytest.MonkeyPatch) -> None:
    svc, master = _build_service()

    class _FakeMavlinkConsts:
        MAV_CMD_PREFLIGHT_CALIBRATION = 241
        MAV_RESULT_ACCEPTED = 0
        MAV_RESULT_IN_PROGRESS = 5

    class _FakeMavutil:
        mavlink = _FakeMavlinkConsts()

    monkeypatch.setattr(mav_module, "mavutil", _FakeMavutil())

    def _ack() -> None:
        svc._handle_msg(_FakeCommandAckMsg(command=241, result=0))

    threading.Timer(0.04, _ack).start()

    out = svc.level_calibration(timeout_s=0.5)
    assert out["ok"] is True
    assert out["action"] == "level_calibration"
    assert out["ack_result"] in {"result_0", "accepted"}
    assert master.mav.command_long_calls
    sent = master.mav.command_long_calls[0]
    assert int(sent[2]) == 241
    assert int(sent[8]) == 2


def test_level_calibration_timeout_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    svc, _ = _build_service()
    with svc._lock:
        svc._status["last_status_text"] = "[critical] prearm: accel cal required"

    class _FakeMavlinkConsts:
        MAV_CMD_PREFLIGHT_CALIBRATION = 241
        MAV_RESULT_ACCEPTED = 0
        MAV_RESULT_IN_PROGRESS = 5

    class _FakeMavutil:
        mavlink = _FakeMavlinkConsts()

    monkeypatch.setattr(mav_module, "mavutil", _FakeMavutil())

    with pytest.raises(RuntimeError, match="timed out waiting for COMMAND_ACK"):
        svc.level_calibration(timeout_s=0.12)


def test_set_compass_north_reference_success() -> None:
    svc, _ = _build_service()
    with svc._lock:
        svc._telemetry["yaw_deg"] = 37.5

    out = svc.set_compass_north_reference(north_heading_deg=0.0)
    assert out["ok"] is True
    assert out["action"] == "compass_north_reference"
    assert abs(float(out["north_heading_deg"]) - 0.0) < 1e-6
    assert abs(float(out["measured_yaw_deg"]) - 37.5) < 1e-6

    tele = svc.get_telemetry()
    aligned = tele.get("yaw_aligned_deg")
    assert aligned is not None
    assert abs(float(aligned) - 0.0) < 1e-6


def test_set_compass_north_reference_requires_yaw() -> None:
    svc, _ = _build_service()
    with svc._lock:
        svc._telemetry["yaw_deg"] = None
    with pytest.raises(RuntimeError, match="current yaw is unavailable"):
        svc.set_compass_north_reference(north_heading_deg=0.0)
