from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from quad_rl.curriculum.presets import get_preset
from quad_rl.envs.dynamics import PointMassDynamics
from quad_rl.envs.safety import apply_safety, check_crash, clamp_xy_inside_bounds
from quad_rl.tasks.base_task import wrap_angle
from quad_rl.tasks.hover_task import HoverTask


class Quad15DEnv(gym.Env):
    """
    Unified 15D observation, 4D continuous action environment.
    Default task is HoverTask if not provided.
    """

    metadata = {"render_modes": []}
    _BASE_OBS_DIM = 15

    def _allow_oob_touch_for_task(self) -> bool:
        task_key = (self.task.name or "").strip().lower()
        allow_default = bool(getattr(self.cfg, "allow_oob_touch", False))
        if task_key == "scan":
            override = getattr(self.cfg, "allow_oob_touch_scan", None)
            return allow_default if override is None else bool(override)
        if task_key == "sequence":
            override = getattr(self.cfg, "allow_oob_touch_sequence", None)
            return allow_default if override is None else bool(override)
        return allow_default

    def _xy_action_limit(self) -> float:
        task_key = (self.task.name or "").strip().lower()
        if task_key == "scan":
            scan_v_xy = getattr(self.cfg, "scan_v_xy_max", None)
            if scan_v_xy is not None:
                return float(scan_v_xy)
        return float(self.cfg.v_xy_max)

    def _yaw_rate_action_limit(self) -> float:
        task_key = (self.task.name or "").strip().lower()
        if task_key == "scan":
            scan_yaw_rate = getattr(self.cfg, "scan_yaw_rate_max", None)
            if scan_yaw_rate is not None:
                return float(scan_yaw_rate)
        return float(self.cfg.yaw_rate_max)

    def _scan_speed_action_enabled(self) -> bool:
        task_key = (self.task.name or "").strip().lower()
        return bool(task_key == "scan" and getattr(self.cfg, "scan_speed_action_enable", False))

    def _action_dim(self) -> int:
        return 5 if self._scan_speed_action_enabled() else 4

    def _action_physical_limits(self) -> np.ndarray:
        v_xy_limit = self._xy_action_limit()
        yaw_rate_limit = self._yaw_rate_action_limit()
        return np.array([v_xy_limit, v_xy_limit, self.cfg.v_z_max, yaw_rate_limit], dtype=np.float32)

    def _scan_obs_aug_enabled(self) -> bool:
        task_key = (self.task.name or "").strip().lower()
        return bool(task_key == "scan" and getattr(self.cfg, "scan_obs_aug_enable", False))

    def _scan_obs_aug_dim(self) -> int:
        if not self._scan_obs_aug_enabled():
            return 0
        patch_size = int(getattr(self.cfg, "scan_obs_patch_size", 5))
        patch_size = max(1, patch_size)
        if patch_size % 2 == 0:
            patch_size += 1
        dim = int(patch_size * patch_size)
        if bool(getattr(self.cfg, "scan_obs_boundary_feat", True)):
            dim += 2
        if bool(getattr(self.cfg, "scan_obs_global_coverage_enable", False)):
            g = int(getattr(self.cfg, "scan_obs_global_size", 8))
            g = max(1, g)
            dim += int(g * g)
        return dim

    def _meas_obs_aug_dim(self) -> int:
        if not bool(getattr(self.cfg, "obs_meas_aug_enable", False)):
            return 0
        dim = 4  # vx, vy, yaw_rate, yaw_err
        if bool(getattr(self.cfg, "obs_ekf_quality_enable", False)):
            dim += 1
        return dim

    def _scan_z_low_limit(self) -> float:
        scan_z_min = getattr(self.cfg, "scan_z_min", None)
        if scan_z_min is not None:
            return float(scan_z_min) - 0.2
        return float(self.cfg.world_z_min) - 0.05

    def _scan_z_high_limit(self) -> float:
        scan_z_max = getattr(self.cfg, "scan_z_max", None)
        if scan_z_max is not None:
            return float(scan_z_max) + 0.2
        return float(self.cfg.world_z_max)

    def _oob_reason(self, task_key: str | None = None) -> str:
        x = float(self.state.pos[0])
        y = float(self.state.pos[1])
        z = float(self.state.pos[2])
        key = (task_key or "").strip().lower()
        z_low_limit = self._scan_z_low_limit() if key == "scan" else float(self.cfg.world_z_min) - 0.05
        z_high_limit = self._scan_z_high_limit() if key == "scan" else float(self.cfg.world_z_max)
        reasons = []
        if abs(x) > float(self.cfg.world_xy_bound):
            reasons.append("x")
        if abs(y) > float(self.cfg.world_xy_bound):
            reasons.append("y")
        if z > z_high_limit:
            reasons.append("z_high")
        if z < z_low_limit:
            reasons.append("z_low")
        if not reasons:
            return "out_of_bounds"
        return "+".join(reasons)

    def __init__(
        self,
        task=None,
        cfg=None,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
        is_eval: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else get_preset("A0")
        self.task = task if task is not None else HoverTask(self.cfg)
        self.max_steps = int(max_steps if max_steps is not None else self.cfg.max_steps)
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self.is_eval = bool(is_eval)

        obs_dim = int(self._BASE_OBS_DIM + self._scan_obs_aug_dim() + self._meas_obs_aug_dim())
        obs_high = np.full((obs_dim,), np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        act_high = np.ones((self._action_dim(),), dtype=np.float32)
        self.action_space = spaces.Box(low=-act_high, high=act_high, dtype=np.float32)

        self.dynamics = PointMassDynamics(self.cfg)
        self.state = self.dynamics.state
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.last_action_norm = np.zeros(self._action_dim(), dtype=np.float32)
        self.step_count = 0
        self.scan_oob_consec = 0
        self.ekf_quality = float(getattr(self.cfg, "obs_ekf_quality_default", 1.0))

    def _initial_state(self) -> tuple[np.ndarray, float]:
        z0 = max(self.cfg.world_z_min + 1.0, 1.0)
        pos = np.array(
            [
                float(self.rng.uniform(-0.2, 0.2)),
                float(self.rng.uniform(-0.2, 0.2)),
                float(z0),
            ],
            dtype=np.float32,
        )
        yaw = float(self.rng.uniform(-np.pi, np.pi))
        return pos, yaw

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self._seed is not None and self.step_count == 0:
            self.rng = np.random.default_rng(self._seed)

        pos0, yaw0 = self._initial_state()
        self.state = self.dynamics.reset(pos0, yaw0)
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.last_action_norm = np.zeros(self._action_dim(), dtype=np.float32)
        self.step_count = 0
        self.scan_oob_consec = 0

        self.task.reset(self, self.rng)

        obs = self._build_obs()
        info = {"task": self.task.name}
        return obs, info

    def _build_obs(self) -> np.ndarray:
        target_pos, target_yaw = self.task.get_target(self)
        err = np.asarray(target_pos, dtype=np.float32) - self.state.pos
        yaw_err = wrap_angle(float(target_yaw) - float(self.state.yaw))

        obs = np.array(
            [
                err[0],
                err[1],
                err[2],
                self.state.vel[0],
                self.state.vel[1],
                self.state.vel[2],
                self.state.roll,
                self.state.pitch,
                yaw_err,
                self.state.p,
                self.state.q,
                self.state.r,
                self.prev_action[0],
                self.prev_action[1],
                self.prev_action[2],
            ],
            dtype=np.float32,
        )
        if self._scan_obs_aug_enabled() and hasattr(self.task, "get_obs_aug_features"):
            aug = np.asarray(self.task.get_obs_aug_features(self), dtype=np.float32).reshape(-1)
            obs = np.concatenate([obs, aug], axis=0).astype(np.float32, copy=False)
        if bool(getattr(self.cfg, "obs_meas_aug_enable", False)):
            meas = [
                float(self.state.vel[0]),
                float(self.state.vel[1]),
                float(self.state.r),
                float(yaw_err),
            ]
            if bool(getattr(self.cfg, "obs_ekf_quality_enable", False)):
                meas.append(float(getattr(self, "ekf_quality", getattr(self.cfg, "obs_ekf_quality_default", 1.0))))
            obs = np.concatenate([obs, np.asarray(meas, dtype=np.float32)], axis=0).astype(np.float32, copy=False)
        return obs

    def step(self, action):
        action_dim = int(self._action_dim())
        raw_action = np.asarray(action, dtype=np.float32).reshape(-1)
        raw_action = raw_action[:action_dim] if raw_action.size >= action_dim else np.pad(raw_action, (0, max(0, action_dim - raw_action.size)))
        action_norm = np.clip(raw_action, -1.0, 1.0).astype(np.float32, copy=False)
        self.last_action_norm = action_norm.copy()
        allow_oob_touch = self._allow_oob_touch_for_task()
        task_key = (self.task.name or "").strip().lower()
        action_phys = action_norm[:4] * self._action_physical_limits()
        if task_key == "scan" and self._scan_speed_action_enabled():
            a_speed_raw = float(action_norm[4]) if action_norm.size >= 5 else 1.0
            a_speed = float(np.clip(0.5 * (a_speed_raw + 1.0), 0.0, 1.0))
            dir_xy = np.asarray(action_norm[:2], dtype=np.float32)
            n = float(np.linalg.norm(dir_xy))
            if n > 1e-6:
                dir_xy = dir_xy / n
            else:
                dir_xy = np.zeros((2,), dtype=np.float32)
            vxy_cap_max = float(self._xy_action_limit())
            speed_cmd = float(a_speed * vxy_cap_max)
            action_phys[0] = float(dir_xy[0] * speed_cmd)
            action_phys[1] = float(dir_xy[1] * speed_cmd)

        safe_action, shield_info = apply_safety(
            action_phys,
            self.cfg,
            self.task.name,
            state_pos=self.state.pos,
        )
        if task_key == "scan":
            vz_cmd = float(safe_action[2])
            z_now = float(self.state.pos[2])
            scan_z_cmd_min = getattr(self.cfg, "scan_z_cmd_min", None)
            scan_z_cmd_max = getattr(self.cfg, "scan_z_cmd_max", None)
            cmd_min = float(scan_z_cmd_min) if scan_z_cmd_min is not None else -float(self.cfg.v_z_max)
            cmd_max = float(scan_z_cmd_max) if scan_z_cmd_max is not None else float(self.cfg.v_z_max)
            if cmd_max < cmd_min:
                cmd_min, cmd_max = cmd_max, cmd_min
            vz_cmd = float(np.clip(vz_cmd, cmd_min, cmd_max))

            scan_z_target = getattr(self.cfg, "scan_z_target", None)
            scan_k_z_p = float(getattr(self.cfg, "scan_k_z_p", 0.0))
            if scan_z_target is not None and abs(scan_k_z_p) > 0.0:
                err = float(scan_z_target) - z_now
                vz_cmd = float(np.clip(vz_cmd + scan_k_z_p * err, cmd_min, cmd_max))
            safe_action[2] = vz_cmd
        self.state = self.dynamics.step(safe_action, self.cfg)
        skip_xy_position_clamp = bool(task_key == "scan" and not allow_oob_touch)
        if skip_xy_position_clamp:
            touch_info = {"oob_touch": False, "xy_clamp_applied": False}
        else:
            touch_info = clamp_xy_inside_bounds(self.state, self.cfg, self.task.name)
        self.prev_action = safe_action[:3].copy()
        self.step_count += 1

        task_step = self.task.step(self)
        crash, crash_info = check_crash(self.state, self.cfg)
        if task_key == "scan":
            x = float(self.state.pos[0])
            y = float(self.state.pos[1])
            z = float(self.state.pos[2])
            z_low_limit = self._scan_z_low_limit()
            z_high_limit = self._scan_z_high_limit()
            scan_out_of_bounds_raw = (
                abs(x) > float(self.cfg.world_xy_bound)
                or abs(y) > float(self.cfg.world_xy_bound)
                or z > z_high_limit
                or z < z_low_limit
            )
            if scan_out_of_bounds_raw:
                self.scan_oob_consec = int(self.scan_oob_consec) + 1
            else:
                self.scan_oob_consec = 0
            grace_steps = max(0, int(getattr(self.cfg, "scan_oob_grace_steps", 0)))
            scan_out_of_bounds_term = bool(scan_out_of_bounds_raw and self.scan_oob_consec > grace_steps)
            crash_info["scan_out_of_bounds_raw"] = bool(scan_out_of_bounds_raw)
            crash_info["out_of_bounds"] = bool(scan_out_of_bounds_term)
            crash_info["scan_z_low_limit"] = float(z_low_limit)
            crash_info["scan_z_high_limit"] = float(z_high_limit)
            crash_info["scan_oob_consec"] = int(self.scan_oob_consec)
            crash = bool(crash_info["out_of_bounds"] or crash_info.get("tilt_exceeded", False))
        touched_xy = bool(touch_info.get("oob_touch", False))
        if touched_xy and not allow_oob_touch:
            crash = True
            crash_info["out_of_bounds"] = True
        if allow_oob_touch and bool(crash_info.get("out_of_bounds", False)):
            # With touch-guard enabled, OOB events are handled by clamp/penalty instead of termination.
            crash = bool(crash_info.get("tilt_exceeded", False))

        success = bool(task_step.success)
        terminated = bool(success or task_step.done or crash)
        truncated = bool(self.step_count >= self.max_steps and not terminated)

        reward = float(task_step.reward)
        if touched_xy:
            reward -= float(getattr(self.cfg, "k_oob_touch", 0.0))
            reward -= float(getattr(self.cfg, "k_oob_touch_step", 0.0))
        if crash:
            reward -= float(self.cfg.crash_penalty)

        info = {}
        info.update(task_step.info)
        info.update(shield_info)
        info.update(touch_info)
        info.update(crash_info)
        info["success"] = success
        info["crash"] = bool(crash)
        info["task"] = self.task.name
        info["vx_cmd"] = float(safe_action[0])
        info["vy_cmd"] = float(safe_action[1])
        info["v_xy_limit_eff"] = shield_info.get("v_xy_limit_eff")
        info["action_raw"] = np.asarray(action_norm[:2], dtype=np.float32).copy()
        if task_key == "scan" and self._scan_speed_action_enabled():
            info["a_speed"] = float(np.clip(0.5 * (float(action_norm[4]) + 1.0), 0.0, 1.0))
        if task_key == "scan":
            info["scan_z_low_limit"] = float(self._scan_z_low_limit())
            info["scan_z_high_limit"] = float(self._scan_z_high_limit())
            info["scan_oob_consec"] = int(self.scan_oob_consec)
        if bool(crash_info.get("out_of_bounds", False)):
            info["oob_reason"] = self._oob_reason(task_key)

        if (
            task_key == "scan"
            and bool(getattr(self.cfg, "scan_debug_oob", False))
            and terminated
            and bool(crash_info.get("out_of_bounds", False))
        ):
            reason = str(info.get("oob_reason", "out_of_bounds"))
            print(
                f"[scan oob debug] terminate step_idx={int(self.step_count)} "
                f"x={float(self.state.pos[0]):.3f} y={float(self.state.pos[1]):.3f} "
                f"bound={float(self.cfg.world_xy_bound):.3f} reason={reason}"
            )

        obs = self._build_obs()
        return obs, reward, terminated, truncated, info

    def render(self):
        return {
            "pos": self.state.pos.copy(),
            "vel": self.state.vel.copy(),
            "rpy": np.array([self.state.roll, self.state.pitch, self.state.yaw], dtype=np.float32),
            "task": self.task.name,
        }

    def close(self):
        return None
