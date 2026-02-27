from __future__ import annotations

import numpy as np

from quad_rl.tasks.base_task import BaseTask, TaskStep


class LandingTask(BaseTask):
    name = "landing"

    def __init__(self, cfg=None) -> None:
        super().__init__(cfg)
        self.target_pos = np.zeros(3, dtype=np.float32)
        self.target_yaw = 0.0
        self.hold_count = 0
        self.prev_z = 0.0

    def reset(self, env, rng: np.random.Generator) -> None:
        super().reset(env, rng)
        self.target_pos = env.state.pos.copy()
        self.target_pos[2] = float(self.cfg.landing_target_z)
        self.target_yaw = float(env.state.yaw)
        self.hold_count = 0
        self.prev_z = float(env.state.pos[2])

    def step(self, env) -> TaskStep:
        xy_dist = float(np.linalg.norm(self.target_pos[:2] - env.state.pos[:2]))
        z_err = abs(float(env.state.pos[2] - self.target_pos[2]))
        vz = float(env.state.vel[2])
        down_progress = max(0.0, self.prev_z - float(env.state.pos[2]))
        self.prev_z = float(env.state.pos[2])

        reward = (
            -xy_dist
            - z_err
            - 0.2 * abs(vz)
            + 0.2 * down_progress
            - float(self.cfg.step_penalty)
        )

        touchdown = (
            float(env.state.pos[2]) <= float(self.cfg.landing_target_z) + 0.05
            and abs(vz) <= float(self.cfg.landing_vz_thresh)
            and xy_dist <= float(self.cfg.landing_xy_radius)
        )

        if touchdown:
            self.hold_count += 1
        else:
            self.hold_count = 0

        success = self.hold_count >= int(self.cfg.landing_hold_steps)
        return TaskStep(
            reward=reward,
            success=success,
            info={"xy_dist": xy_dist, "z_err": z_err, "vz": vz, "hold_count": self.hold_count},
        )

    def get_target(self, env):
        return self.target_pos, self.target_yaw
