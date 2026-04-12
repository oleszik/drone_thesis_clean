from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_health_exposes_runtime_mode() -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload.get("runtime_mode") in ("simulation", "real_mission")


def test_real_namespace_readiness_available() -> None:
    resp = client.get("/api/real/readiness")
    assert resp.status_code == 200
    payload = resp.json()
    assert "can_autonomous" in payload


def test_sim_namespace_state_endpoints_available() -> None:
    sim_state = client.get("/api/sim/sitl/state")
    assert sim_state.status_code == 200
    mission_state = client.get("/api/sim/mission/sim/state")
    assert mission_state.status_code == 200


def test_real_namespace_control_routes_exposed() -> None:
    # Endpoint should exist (200 if connected, 409 when no mavlink link)
    resp = client.post("/api/real/control/hold")
    assert resp.status_code in (200, 409)
