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


def test_real_namespace_mission_executor_routes_exposed() -> None:
    state = client.get("/api/real/mission/state")
    assert state.status_code == 200
    payload = state.json()
    assert "state" in payload
    assert "scan_active" in payload

    # Endpoint should exist (200 if ready, 409 while autonomy is blocked)
    resp = client.post("/api/real/mission/start", json={"alt_m": 10.0, "accept_radius_m": 3.0})
    assert resp.status_code in (200, 409)


def test_real_namespace_mission_generation_routes_exposed() -> None:
    scan = client.post("/api/real/mission/generate_scan", json={"spacing_m": 8.0, "speed_m_s": 3.0})
    assert scan.status_code in (200, 409)

    orbit = client.post(
        "/api/real/mission/generate_orbit_scan",
        json={"radius_m": 12.0, "altitude_m": 10.0, "laps": 1, "points_per_lap": 24},
    )
    assert orbit.status_code in (200, 409)
