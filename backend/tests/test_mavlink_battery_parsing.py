from __future__ import annotations

from backend.app.mavlink_service import MavlinkService, MavlinkSettings


class _Msg:
    def __init__(self, msg_type: str, **fields) -> None:
        self._msg_type = msg_type
        for k, v in fields.items():
            setattr(self, k, v)

    def get_type(self) -> str:
        return self._msg_type


def _build_service() -> MavlinkService:
    return MavlinkService(
        MavlinkSettings(
            connection_url="udp:127.0.0.1:14550",
            reconnect_sec=1.0,
            heartbeat_timeout_sec=2.0,
            default_takeoff_alt_m=3.0,
        )
    )


def test_parse_sys_status_battery_fields() -> None:
    msg = _Msg(
        "SYS_STATUS",
        voltage_battery=22500,
        current_battery=1350,
        battery_remaining=72,
    )
    out = MavlinkService._parse_sys_status_battery(msg, now_unix=1000.0)
    assert out["battery_source"] == "SYS_STATUS"
    assert out["battery_voltage_v"] == 22.5
    assert out["battery_current_a"] == 13.5
    assert out["battery_percent"] == 72.0
    assert out["battery_remaining_percent"] == 72.0
    assert out["battery_updated_at_unix"] == 1000.0


def test_parse_battery_status_fields() -> None:
    msg = _Msg(
        "BATTERY_STATUS",
        voltages=[3800, 3810, 3790, 3820, 3805, 3815, 65535, 0, 0, 0],
        current_battery=420,
        battery_remaining=68,
        current_consumed=1540,
    )
    out = MavlinkService._parse_battery_status_battery(msg, now_unix=2000.0)
    assert out["battery_source"] == "BATTERY_STATUS"
    assert abs(float(out["battery_voltage_v"]) - 22.84) < 1e-6
    assert out["battery_current_a"] == 4.2
    assert out["battery_percent"] == 68.0
    assert out["battery_remaining_percent"] == 68.0
    assert out["battery_consumed_mah"] == 1540.0
    assert out["battery_updated_at_unix"] == 2000.0


def test_handle_msg_updates_battery_telemetry_and_debug_log() -> None:
    svc = _build_service()

    svc._handle_msg(
        _Msg(
            "SYS_STATUS",
            voltage_battery=23000,
            current_battery=300,
            battery_remaining=80,
        )
    )

    svc._handle_msg(
        _Msg(
            "BATTERY_STATUS",
            voltages=[3900, 3900, 3900, 3900, 3900, 3900, 65535, 65535, 65535, 65535],
            current_battery=500,
            battery_remaining=78,
            current_consumed=220,
        )
    )

    tele = svc.get_telemetry()
    assert abs(float(tele["battery_voltage_v"]) - 23.4) < 1e-6
    assert tele["battery_current_a"] == 5.0
    assert tele["battery_percent"] == 78.0
    assert tele["battery_remaining_percent"] == 78.0
    assert tele["battery_consumed_mah"] == 220.0
    assert tele["battery_source"] == "BATTERY_STATUS"
    assert tele["battery_updated_at_unix"] is not None

    recent = svc.get_recent_battery_messages(limit=5)
    assert len(recent) == 2
    assert recent[0]["source"] == "SYS_STATUS"
    assert recent[1]["source"] == "BATTERY_STATUS"


def test_message_debug_counts_sys_and_battery_status() -> None:
    svc = _build_service()
    svc._handle_msg(_Msg("SYS_STATUS", voltage_battery=22100, current_battery=120, battery_remaining=77))
    svc._handle_msg(
        _Msg(
            "BATTERY_STATUS",
            voltages=[3700, 3700, 3700, 3700, 3700, 3700, 65535, 65535, 65535, 65535],
            current_battery=210,
            battery_remaining=76,
            current_consumed=100,
        )
    )

    dbg = svc.get_message_debug()
    assert int(dbg["total_messages"]) >= 2
    counts = dbg["message_counts"]
    assert counts["SYS_STATUS"] >= 1
    assert counts["BATTERY_STATUS"] >= 1
    assert "message_last_seen_unix" in dbg
    assert dbg["message_last_seen_unix"]["SYS_STATUS"] is not None
    assert dbg["message_last_seen_unix"]["BATTERY_STATUS"] is not None
    assert "message_age_s" in dbg
    assert dbg["message_age_s"]["SYS_STATUS"] is not None
    assert dbg["message_age_s"]["BATTERY_STATUS"] is not None
