from __future__ import annotations

import numpy as np

from quad_rl.tasks.base_task import BaseTask, TaskStep


class MissionTask(BaseTask):
    """Stage D mission: takeoff -> transit -> scan -> return -> land."""

    name = "mission"

    def __init__(self, cfg=None) -> None:
        super().__init__(cfg)
        self.phase = "takeoff"
        self.home = np.zeros(3, dtype=np.float32)
        self.takeoff_target = np.zeros(3, dtype=np.float32)
        self.transit_target = np.zeros(3, dtype=np.float32)
        self.return_target = np.zeros(3, dtype=np.float32)
        self.land_target = np.zeros(3, dtype=np.float32)
        self.scan_points = np.zeros((1, 3), dtype=np.float32)
        self.scan_idx = 0
        self.target_yaw = 0.0
        self.prev_dist = 0.0
        self.hold_count = 0
        self.done = False

    def _build_scan_points(self, center: np.ndarray) -> np.ndarray:
        spacing = float(self.cfg.scan_spacing)
        pts = [
            center + np.array([0.0, 0.0, 0.0], dtype=np.float32),
            center + np.array([spacing, 0.0, 0.0], dtype=np.float32),
            center + np.array([spacing, spacing, 0.0], dtype=np.float32),
            center + np.array([0.0, spacing, 0.0], dtype=np.float32),
        ]
        return np.asarray(pts, dtype=np.float32)

    def reset(self, env, rng: np.random.Generator) -> None:
        super().reset(env, rng)
        self.home = env.state.pos.copy()
        self.target_yaw = float(env.state.yaw)
        self.done = False
        self.hold_count = 0
        self.scan_idx = 0

        self.takeoff_target = self.home.copy()
        self.takeoff_target[2] = float(self.home[2] + self.cfg.mission_takeoff_height)

        hop = float(self.cfg.mission_transit_hop)
        theta = float(rng.uniform(-np.pi, np.pi))
        self.transit_target = self.takeoff_target + np.array(
            [hop * np.cos(theta), hop * np.sin(theta), 0.0], dtype=np.float32
        )

        self.scan_points = self._build_scan_points(self.transit_target)

        self.return_target = self.home.copy()
        self.return_target[2] = float(self.takeoff_target[2])

        self.land_target = self.home.copy()
        self.land_target[2] = float(self.cfg.landing_target_z)

        self.phase = "takeoff"
        self.prev_dist = float(np.linalg.norm(self.takeoff_target - env.state.pos))

    def _current_target(self) -> np.ndarray:
        if self.phase == "takeoff":
            return self.takeoff_target
        if self.phase == "transit":
            return self.transit_target
        if self.phase == "scan":
            idx = int(np.clip(self.scan_idx, 0, len(self.scan_points) - 1))
            return self.scan_points[idx]
        if self.phase == "return":
            return self.return_target
        return self.land_target

    def step(self, env) -> TaskStep:
        if self.done:
            return TaskStep(reward=0.0, success=True, done=True, info={"phase": self.phase})

        target = self._current_target()
        dist = float(np.linalg.norm(target - env.state.pos))
        progress = self.prev_dist - dist
        self.prev_dist = dist

        reward = float(self.cfg.seq_k_prog) * progress - 0.1 * dist - float(self.cfg.step_penalty)
        radius = float(self.cfg.seq_success_radius)

        transition = False
        if self.phase == "land":
            touchdown = (
                float(env.state.pos[2]) <= float(self.cfg.landing_target_z) + 0.05
                and abs(float(env.state.vel[2])) <= float(self.cfg.landing_vz_thresh)
            )
            if touchdown:
                self.hold_count += 1
            else:
                self.hold_count = 0
            if self.hold_count >= int(self.cfg.landing_hold_steps):
                self.done = True
                return TaskStep(reward=reward, success=True, done=True, info={"phase": self.phase})
        else:
            if dist <= radius:
                self.hold_count += 1
            else:
                self.hold_count = 0

            if self.hold_count >= int(self.cfg.seq_hold_steps):
                transition = True
                self.hold_count = 0

        if transition:
            if self.phase == "takeoff":
                self.phase = "transit"
            elif self.phase == "transit":
                self.phase = "scan"
                self.scan_idx = 0
            elif self.phase == "scan":
                self.scan_idx += 1
                if self.scan_idx >= len(self.scan_points):
                    self.phase = "return"
            elif self.phase == "return":
                self.phase = "land"

            self.prev_dist = float(np.linalg.norm(self._current_target() - env.state.pos))

        info = {"phase": self.phase, "dist": dist, "scan_idx": int(self.scan_idx)}
        return TaskStep(reward=reward, success=False, done=False, info=info)

    def get_target(self, env):
        return self._current_target(), self.target_yaw
