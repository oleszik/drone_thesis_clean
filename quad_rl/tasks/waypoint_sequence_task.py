from __future__ import annotations

import numpy as np

from quad_rl.tasks.base_task import BaseTask, TaskStep


class WaypointSequenceTask(BaseTask):
    name = "sequence"

    def __init__(self, cfg=None) -> None:
        super().__init__(cfg)
        self.waypoints = np.zeros((1, 3), dtype=np.float32)
        self.target_yaw = 0.0
        self.wp_idx = 0
        self.hold_count = 0.0
        self.prev_dist = 0.0
        self.stall_steps = 0
        self.done = False

    def reset(self, env, rng: np.random.Generator) -> None:
        super().reset(env, rng)
        n = int(self.cfg.seq_n_waypoints)
        start = env.state.pos.copy()
        points = []
        cursor = start.copy()

        for _ in range(n):
            hop = float(rng.uniform(self.cfg.seq_hop_min, self.cfg.seq_hop_max))
            theta = float(rng.uniform(-np.pi, np.pi))
            cursor = cursor + np.array([hop * np.cos(theta), hop * np.sin(theta), 0.0], dtype=np.float32)
            if bool(self.cfg.seq_lock_altitude):
                cursor[2] = float(start[2])
            else:
                cursor[2] = float(
                    np.clip(
                        cursor[2] + rng.uniform(-0.3, 0.3),
                        self.cfg.world_z_min + 0.4,
                        self.cfg.world_z_max - 0.4,
                    )
                )
            points.append(cursor.copy())

        self.waypoints = np.asarray(points, dtype=np.float32)
        self.target_yaw = float(env.state.yaw)
        self.wp_idx = 0
        self.hold_count = 0.0
        self.done = False
        self.stall_steps = 0
        self.prev_dist = float(np.linalg.norm(self.waypoints[0] - env.state.pos))

    def _current_target(self) -> np.ndarray:
        idx = int(np.clip(self.wp_idx, 0, len(self.waypoints) - 1))
        return self.waypoints[idx]

    def step(self, env) -> TaskStep:
        if self.done:
            return TaskStep(reward=0.0, success=True, done=True, info={"wp_idx": self.wp_idx})

        target = self._current_target()
        dist = float(np.linalg.norm(target - env.state.pos))
        progress = self.prev_dist - dist
        self.prev_dist = dist
        radius = float(self.cfg.seq_success_radius)

        if abs(progress) < 1e-3 and dist > radius:
            self.stall_steps += 1
        else:
            self.stall_steps = 0

        reward = (
            float(self.cfg.seq_k_prog) * progress
            - 0.1 * dist
            - 0.001 * self.stall_steps
            - float(self.cfg.step_penalty)
        )

        # Soft wall near XY boundary to reduce out-of-bounds crashes.
        wall_margin = float(getattr(self.cfg, "seq_wall_margin", 0.0))
        if wall_margin > 0.0:
            x = float(env.state.pos[0])
            y = float(env.state.pos[1])
            vx = float(env.state.vel[0])
            vy = float(env.state.vel[1])
            d_edge = float(self.cfg.world_xy_bound) - max(abs(x), abs(y))
            if d_edge < wall_margin:
                frac = 1.0 - d_edge / max(wall_margin, 1e-6)
                reward -= float(getattr(self.cfg, "seq_k_wall", 0.0)) * (frac**2)

                v_out = (x * vx + y * vy) / (abs(x) + abs(y) + 1e-6)
                reward -= float(getattr(self.cfg, "seq_k_wall_vel", 0.0)) * (max(0.0, v_out) ** 2)

        if dist <= radius:
            self.hold_count = min(float(self.cfg.seq_hold_steps), self.hold_count + 1.0)
            # Stay-shaping: make near-target behavior sticky and discourage lateral oscillation.
            reward += float(self.cfg.seq_k_stay) * (radius - dist)
            reward -= float(self.cfg.seq_k_v_in) * float(np.linalg.norm(env.state.vel[:2]))
        else:
            margin = float(self.cfg.seq_hold_reset_margin)
            decay = float(self.cfg.seq_hold_decay)
            if dist <= radius + margin:
                self.hold_count = max(0.0, self.hold_count - 0.5 * decay)
            else:
                self.hold_count = max(0.0, self.hold_count - decay)

        success = False
        if self.hold_count >= float(self.cfg.seq_hold_steps):
            self.wp_idx += 1
            self.hold_count = 0.0
            if self.wp_idx >= len(self.waypoints):
                self.done = True
                success = True
            else:
                next_target = self._current_target()
                self.prev_dist = float(np.linalg.norm(next_target - env.state.pos))

        info = {
            "wp_idx": int(min(self.wp_idx, len(self.waypoints) - 1)),
            "n_waypoints": int(len(self.waypoints)),
            "dist": dist,
            "hold_count": self.hold_count,
            "stall_steps": self.stall_steps,
        }
        return TaskStep(reward=reward, success=success, done=self.done, info=info)

    def get_target(self, env):
        return self._current_target(), self.target_yaw
