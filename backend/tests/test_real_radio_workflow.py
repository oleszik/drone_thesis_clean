from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_real_connection_status_shape() -> None:
    resp = client.get("/api/real/connection/status")
    assert resp.status_code == 200
    payload = resp.json()
    for key in (
        "serial_port",
        "serial_baud",
        "connected",
        "last_heartbeat_age_s",
        "last_telemetry_age_s",
        "stale",
        "lost",
        "state",
        "error_message",
    ):
        assert key in payload


def test_real_connection_ports_endpoint_available() -> None:
    resp = client.get("/api/real/connection/ports")
    assert resp.status_code == 200
    payload = resp.json()
    assert "ports" in payload


def test_real_heartbeat_test_reports_not_ready_when_disconnected() -> None:
    resp = client.post("/api/real/connection/heartbeat_test")
    assert resp.status_code in (200, 409)
    if resp.status_code == 409:
        detail = resp.json().get("detail")
        assert isinstance(detail, dict)
        assert "ok" in detail


def test_real_readiness_includes_radio_checks() -> None:
    resp = client.get("/api/real/readiness")
    assert resp.status_code == 200
    payload = resp.json()
    checks = payload.get("checks") or []
    keys = {c.get("key") for c in checks if isinstance(c, dict)}
    assert "radio_connected" in keys
    assert "radio_heartbeat_fresh" in keys
    assert "radio_telemetry_fresh" in keys


def test_real_debug_battery_includes_message_rate_status() -> None:
    resp = client.get("/api/real/debug/battery")
    assert resp.status_code == 200
    payload = resp.json()
    assert "message_debug" in payload
    assert "battery_message_status" in payload

    message_debug = payload["message_debug"]
    assert "total_messages" in message_debug
    assert "message_counts" in message_debug
    assert "message_last_seen_unix" in message_debug
    assert "message_age_s" in message_debug

    battery_message_status = payload["battery_message_status"]
    for key in (
        "sys_status_seen",
        "battery_status_seen",
        "sys_status_count",
        "battery_status_count",
        "sys_status_age_s",
        "battery_status_age_s",
    ):
        assert key in battery_message_status
