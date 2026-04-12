from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_start_position_rejects_outside_operating_fence() -> None:
    resp = client.post("/api/mission/start_position", json={"lng": 0.0, "lat": 0.0})
    assert resp.status_code == 400
    detail = str(resp.json().get("detail", ""))
    assert "outside configured fence" in detail


def test_area_rejects_polygon_outside_operating_fence() -> None:
    resp = client.post(
        "/api/mission/area",
        json={
            "polygon_lng_lat": [
                [0.0, 0.0],
                [0.001, 0.0],
                [0.001, 0.001],
                [0.0, 0.001],
            ]
        },
    )
    assert resp.status_code == 400
    detail = str(resp.json().get("detail", ""))
    assert "outside configured fence" in detail
