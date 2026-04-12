from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_health_endpoint_smoke() -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload.get("ok") is True
    assert payload.get("service") == "backend"


def test_readiness_endpoint_smoke() -> None:
    resp = client.get("/api/readiness")
    assert resp.status_code == 200
    payload = resp.json()

    assert "overall_ready" in payload
    assert "can_manual" in payload
    assert "can_autonomous" in payload
    assert "checks" in payload
    assert "blocking_reasons" in payload
    assert "timestamp" in payload

    checks = payload.get("checks")
    assert isinstance(checks, list)
    assert len(checks) >= 1

    first = checks[0]
    assert isinstance(first, dict)
    for key in ("key", "ok", "severity", "message", "value"):
        assert key in first
