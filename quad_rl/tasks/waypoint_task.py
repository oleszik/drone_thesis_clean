from __future__ import annotations

import numpy as np

from quad_rl.tasks.base_task import BaseTask, TaskStep


class WaypointTask(BaseTask):
    name = "waypoint"

    def __init__(self, cfg=None) -> None:
        super().__init__(cfg)
        self.target_pos = np.zeros(3, dtype=np.float32)
        self.target_yaw = 0.0
        self.hold_count = 0.0
        self.prev_dist = 0.0

    def reset(self, env, rng: np.random.Generator) -> None:
        super().reset(env, rng)
        start = env.state.pos.copy()
        hop = float(rng.uniform(self.cfg.waypoint_hop_min, self.cfg.waypoint_hop_max))
        theta = float(rng.uniform(-np.pi, np.pi))
        delta = np.array([hop * np.cos(theta), hop * np.sin(theta), 0.0], dtype=np.float32)

        self.target_pos = start + delta
        if not bool(self.cfg.waypoint_lock_altitude):
            self.target_pos[2] = float(
                np.clip(
                    start[2] + rng.uniform(-0.5, 0.5),
                    self.cfg.world_z_min + 0.4,
                    self.cfg.world_z_max - 0.4,
                )
            )
        else:
            self.target_pos[2] = float(start[2])

        self.target_yaw = float(env.state.yaw)
        self.hold_count = 0.0
        self.prev_dist = float(np.linalg.norm(self.target_pos - env.state.pos))

    def step(self, env) -> TaskStep:
        dist = float(np.linalg.norm(self.target_pos - env.state.pos))
        progress = self.prev_dist - dist
        self.prev_dist = dist

        radius = float(self.cfg.waypoint_success_radius)
        reward = (
            float(self.cfg.waypoint_k_prog) * progress
            - 0.1 * dist
            - float(self.cfg.step_penalty)
        )

        if dist <= radius:
            self.hold_count = min(float(self.cfg.waypoint_hold_steps), self.hold_count + 1.0)
            # Stay-shaping: reward settling in-target and reduce lateral churning.
            reward += float(self.cfg.waypoint_k_stay) * (radius - dist)
            reward -= float(self.cfg.waypoint_k_v_in) * float(np.linalg.norm(env.state.vel[:2]))
        else:
            margin = float(self.cfg.waypoint_hold_reset_margin)
            decay = float(self.cfg.waypoint_hold_decay)
            if dist <= radius + margin:
                self.hold_count = max(0.0, self.hold_count - 0.5 * decay)
            else:
                self.hold_count = max(0.0, self.hold_count - decay)

        success = self.hold_count >= float(self.cfg.waypoint_hold_steps)
        return TaskStep(
            reward=reward,
            success=success,
            info={"dist": dist, "hold_count": self.hold_count},
        )

    def get_target(self, env):
        return self.target_pos, self.target_yaw
