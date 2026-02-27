from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class DynamicsState:
    pos: np.ndarray
    vel: np.ndarray
    roll: float
    pitch: float
    yaw: float
    p: float
    q: float
    r: float


class PointMassDynamics:
    """Lightweight 3D point-mass + yaw dynamics with gravity compensation."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.state = DynamicsState(
            pos=np.zeros(3, dtype=np.float32),
            vel=np.zeros(3, dtype=np.float32),
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            p=0.0,
            q=0.0,
            r=0.0,
        )

    def reset(self, pos: np.ndarray, yaw: float) -> DynamicsState:
        self.state.pos = np.asarray(pos, dtype=np.float32).copy()
        self.state.vel = np.zeros(3, dtype=np.float32)
        self.state.roll = 0.0
        self.state.pitch = 0.0
        self.state.yaw = float(yaw)
        self.state.p = 0.0
        self.state.q = 0.0
        self.state.r = 0.0
        return self.state

    def step(self, action: np.ndarray, cfg=None) -> DynamicsState:
        cfg = cfg or self.cfg
        dt = float(cfg.dt)

        v_cmd = np.asarray(action[:3], dtype=np.float32)
        yaw_rate_cmd = float(action[3])

        vel_err = v_cmd - self.state.vel
        acc_cmd = float(cfg.dynamics_k_vel) * vel_err

        g = float(cfg.dynamics_g)
        acc = acc_cmd.copy()
        # az = control - gravity + compensation. compensation=1.0 means neutral hover at vz_cmd=0.
        acc[2] = acc_cmd[2] - g + float(cfg.gravity_compensation) * g

        self.state.vel = self.state.vel + acc * dt
        self.state.pos = self.state.pos + self.state.vel * dt

        roll_des = float(np.clip(-acc_cmd[1] / max(g, 1e-6), -0.9, 0.9))
        pitch_des = float(np.clip(acc_cmd[0] / max(g, 1e-6), -0.9, 0.9))

        prev_roll = self.state.roll
        prev_pitch = self.state.pitch
        self.state.roll = roll_des
        self.state.pitch = pitch_des
        self.state.p = (self.state.roll - prev_roll) / max(dt, 1e-6)
        self.state.q = (self.state.pitch - prev_pitch) / max(dt, 1e-6)

        tau_yaw = max(float(cfg.dynamics_tau_yaw), 1e-3)
        self.state.r = self.state.r + ((yaw_rate_cmd - self.state.r) / tau_yaw) * dt
        self.state.yaw = wrap_angle(self.state.yaw + self.state.r * dt)

        return self.state
