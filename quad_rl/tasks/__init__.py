from __future__ import annotations

from quad_rl.tasks.base_task import BaseTask, TaskStep
from quad_rl.tasks.hover_task import HoverTask
from quad_rl.tasks.landing_task import LandingTask
from quad_rl.tasks.mission_task import MissionTask
from quad_rl.tasks.scan_task import ScanTask
from quad_rl.tasks.waypoint_sequence_task import WaypointSequenceTask
from quad_rl.tasks.waypoint_task import WaypointTask
from quad_rl.tasks.yaw_task import YawTask


class StageAMixTask(BaseTask):
    name = "stage_a"

    def __init__(self, cfg=None) -> None:
        super().__init__(cfg)
        self._rng = None
        self._active = HoverTask(cfg)
        self._pool = [HoverTask(cfg), YawTask(cfg), LandingTask(cfg)]

    def reset(self, env, rng):
        self._rng = rng
        idx = int(rng.integers(0, len(self._pool)))
        self._active = self._pool[idx]
        self._active.reset(env, rng)

    def step(self, env) -> TaskStep:
        result = self._active.step(env)
        result.info["primitive"] = self._active.name
        return result

    def get_target(self, env):
        return self._active.get_target(env)


def build_task(name: str, cfg=None) -> BaseTask:
    key = (name or "hover").strip().lower()
    if key == "hover":
        return HoverTask(cfg)
    if key == "yaw":
        return YawTask(cfg)
    if key in ("landing", "land"):
        return LandingTask(cfg)
    if key == "waypoint":
        return WaypointTask(cfg)
    if key == "sequence":
        return WaypointSequenceTask(cfg)
    if key == "scan":
        return ScanTask(cfg)
    if key == "mission":
        return MissionTask(cfg)
    if key == "stage_a":
        return StageAMixTask(cfg)
    raise ValueError(f"Unknown task '{name}'.")


__all__ = [
    "BaseTask",
    "TaskStep",
    "HoverTask",
    "YawTask",
    "LandingTask",
    "WaypointTask",
    "WaypointSequenceTask",
    "ScanTask",
    "MissionTask",
    "StageAMixTask",
    "build_task",
]
