from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def _assert_autonomy_blocked(resp) -> None:
    assert resp.status_code == 409
    payload = resp.json()
    detail = payload.get("detail")
    assert isinstance(detail, dict)
    assert "error" in detail
    assert "blocking_reasons" in detail
    assert "readiness_snapshot" in detail


def test_guard_blocks_generate_scan_when_not_ready() -> None:
    resp = client.post(
        "/api/mission/generate_scan",
        json={"spacing_m": 8.0, "speed_m_s": 3.0, "start_scan": False, "auto_spacing": False},
    )
    _assert_autonomy_blocked(resp)


def test_guard_blocks_sitl_start_when_not_ready() -> None:
    resp = client.post("/api/sitl/start_scan", json={"alt_m": 10.0, "accept_radius_m": 3.0})
    _assert_autonomy_blocked(resp)


def test_guard_blocks_generate_tiny_mission_when_not_ready() -> None:
    resp = client.post("/api/mission/generate_tiny", json={})
    _assert_autonomy_blocked(resp)


def test_control_hold_endpoint_available() -> None:
    resp = client.post("/api/control/hold")
    # In test env MAVLink is usually disconnected (409), but endpoint should exist and
    # can return 200 when connected.
    assert resp.status_code in (200, 409)
    payload = resp.json()
    if resp.status_code == 200:
        assert payload.get("target_mode") == "LOITER"
        assert "resulting_mode" in payload
