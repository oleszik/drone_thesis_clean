from __future__ import annotations

from typing import Any

from ...mission_service import MissionService


class RealMissionService:
    def __init__(self, mission: MissionService) -> None:
        self._mission = mission

    def generate_scan(self, **kwargs: Any) -> dict[str, Any]:
        return self._mission.generate_scan(**kwargs)

    def generate_orbit_scan(self, **kwargs: Any) -> dict[str, Any]:
        return self._mission.generate_orbit_scan(**kwargs)

    def generate_tiny_mission(self, **kwargs: Any) -> dict[str, Any]:
        return self._mission.generate_tiny_mission(**kwargs)

    def path(self) -> dict[str, Any]:
        return self._mission.get_path()
