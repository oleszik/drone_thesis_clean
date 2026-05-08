from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def _extract_payload(resp):
    body = resp.json()
    if resp.status_code == 200:
        return body
    return body.get("detail") if isinstance(body, dict) else None


def test_real_validate_start_endpoint_shape() -> None:
    resp = client.post("/api/real/mission/validate_start", json={"alt_m": 10.0, "accept_radius_m": 3.0})
    assert resp.status_code in (200, 409)
    payload = _extract_payload(resp)
    assert isinstance(payload, dict)
    assert payload.get("action") == "validate_start"
    assert "can_start" in payload
    assert "blocking_reasons" in payload
    assert "readiness_snapshot" in payload
    assert "mission_summary" in payload
    assert "timestamp" in payload


def test_validate_start_updates_last_command_debug() -> None:
    client.post("/api/real/mission/validate_start", json={"alt_m": 9.0, "accept_radius_m": 2.5})
    dbg = client.get("/api/real/debug/last_command")
    assert dbg.status_code == 200
    payload = dbg.json()
    assert payload.get("action") == "validate_start"
    assert payload.get("endpoint") == "/api/real/mission/validate_start"
