from __future__ import annotations

import numpy as np

from quad_rl.tasks.base_task import BaseTask, TaskStep


class HoverTask(BaseTask):
    name = "hover"

    def __init__(self, cfg=None) -> None:
        super().__init__(cfg)
        self.target_pos = np.zeros(3, dtype=np.float32)
        self.target_yaw = 0.0
        self.hold_count = 0

    def reset(self, env, rng: np.random.Generator) -> None:
        super().reset(env, rng)
        self.target_pos = env.state.pos.copy()
        self.target_yaw = float(env.state.yaw)
        self.hold_count = 0

    def step(self, env) -> TaskStep:
        dist = float(np.linalg.norm(self.target_pos - env.state.pos))
        speed = float(np.linalg.norm(env.state.vel))
        reward = -dist - 0.05 * speed - float(self.cfg.step_penalty)

        if dist <= float(self.cfg.hover_success_radius):
            self.hold_count += 1
        else:
            self.hold_count = 0

        success = self.hold_count >= int(self.cfg.hover_hold_steps)
        return TaskStep(
            reward=reward,
            success=success,
            info={"dist": dist, "hold_count": self.hold_count},
        )

    def get_target(self, env):
        return self.target_pos, self.target_yaw
 