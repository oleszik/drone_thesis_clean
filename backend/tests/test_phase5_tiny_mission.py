from __future__ import annotations

from backend.app.mission_service import MissionService


def test_tiny_mission_clamps_and_profile_shape() -> None:
    mission = MissionService(footprint_radius_m=6.0)
    out = mission.generate_tiny_mission(
        start_lng=116.397428,
        start_lat=39.90923,
        heading_deg=90.0,
        takeoff_alt_m=99.0,
        hover_before_s=0.2,
        forward_m=42.0,
        hover_after_s=120.0,
        speed_m_s=9.0,
    )

    assert out["mission_type"] == "tiny_mission"
    assert len(out["waypoints_lng_lat"]) == 3
    cfg = out["config"]
    assert cfg["preset"] == "tiny_mission"
    assert cfg["takeoff_alt_m"] == 6.0
    assert cfg["forward_m"] == 6.0
    assert cfg["hover_before_s"] == 2.0
    assert cfg["hover_after_s"] == 20.0
    assert cfg["speed_m_s"] == 2.0

    profile = cfg["command_profile"]
    assert [step["action"] for step in profile] == ["takeoff", "hold", "move_forward", "hold", "rtl"]


def test_tiny_mission_rejects_outside_fence() -> None:
    mission = MissionService(footprint_radius_m=6.0)
    try:
        mission.generate_tiny_mission(
            start_lng=116.397428,
            start_lat=39.90923,
            heading_deg=0.0,
            takeoff_alt_m=3.0,
            forward_m=3.0,
            fence_polygon_lng_lat=[
                [116.3973, 39.9092],
                [116.3974, 39.9092],
                [116.3974, 39.90922],
                [116.3973, 39.90922],
            ],
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "outside configured fence" in str(exc)


def test_tiny_vertical_profile_shape() -> None:
    mission = MissionService(footprint_radius_m=6.0)
    out = mission.generate_tiny_mission(
        start_lng=116.397428,
        start_lat=39.90923,
        mission_profile="vertical_hop",
        takeoff_alt_m=3.0,
        hover_before_s=4.0,
    )

    assert out["mission_type"] == "tiny_mission"
    assert len(out["waypoints_lng_lat"]) == 2
    assert out["waypoints_lng_lat"][0] == out["waypoints_lng_lat"][1]

    cfg = out["config"]
    assert cfg["preset"] == "tiny_vertical_mission"
    assert cfg["mission_profile"] == "vertical_hop"
    assert cfg["forward_m"] == 0.0
    assert cfg["hover_after_s"] == 0.0
    profile = cfg["command_profile"]
    assert [step["action"] for step in profile] == ["takeoff", "hold", "land", "disarm"]
