from __future__ import annotations

from typing import Any

from ...mission_service import MissionService
from ...sitl_executor import SitlExecutor


class SimulationMissionService:
    def __init__(self, mission: MissionService, sitl: SitlExecutor) -> None:
        self._mission = mission
        self._sitl = sitl

    def sim_start(self) -> dict[str, Any]:
        return self._mission.sim_start()

    def sim_pause(self) -> dict[str, Any]:
        return self._mission.sim_pause()

    def sim_stop(self) -> dict[str, Any]:
        return self._mission.sim_stop()

    def sim_tick(self, dt: float) -> dict[str, Any]:
        return self._mission.step(dt)

    def sim_state(self) -> dict[str, Any]:
        return self._mission.get_sim_state()

    def sitl_start_scan(self, *, alt_m: float, accept_radius_m: float) -> dict[str, Any]:
        return self._sitl.start_scan(alt_m=alt_m, accept_radius_m=accept_radius_m)

    def sitl_stop_scan(self) -> dict[str, Any]:
        return self._sitl.stop_scan()

    def sitl_state(self) -> dict[str, Any]:
        return self._sitl.get_state()
