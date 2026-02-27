from __future__ import annotations

import numpy as np

from quad_rl.tasks.base_task import BaseTask, TaskStep, wrap_angle


class YawTask(BaseTask):
    name = "yaw"

    def __init__(self, cfg=None) -> None:
        super().__init__(cfg)
        self.target_pos = np.zeros(3, dtype=np.float32)
        self.target_yaw = 0.0
        self.hold_count = 0

    def reset(self, env, rng: np.random.Generator) -> None:
        super().reset(env, rng)
        self.target_pos = env.state.pos.copy()
        yaw_delta = float(rng.uniform(-self.cfg.yaw_target_range, self.cfg.yaw_target_range))
        self.target_yaw = wrap_angle(float(env.state.yaw) + yaw_delta)
        self.hold_count = 0

    def step(self, env) -> TaskStep:
        pos_dist = float(np.linalg.norm(self.target_pos - env.state.pos))
        yaw_err = abs(wrap_angle(self.target_yaw - float(env.state.yaw)))
        speed = float(np.linalg.norm(env.state.vel))

        reward = -pos_dist - 0.5 * yaw_err - 0.05 * speed - float(self.cfg.step_penalty)

        pos_ok = pos_dist <= float(self.cfg.yaw_success_radius)
        yaw_ok = yaw_err <= float(self.cfg.yaw_tolerance)

        if pos_ok and yaw_ok:
            self.hold_count += 1
        else:
            self.hold_count = 0

        success = self.hold_count >= int(self.cfg.yaw_hold_steps)
        return TaskStep(
            reward=reward,
            success=success,
            info={"pos_dist": pos_dist, "yaw_err": yaw_err, "hold_count": self.hold_count},
        )

    def get_target(self, env):
        return self.target_pos, self.target_yaw
