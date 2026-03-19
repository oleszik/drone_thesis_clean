from __future__ import annotations

import numpy as np


def compute_next_target(current_pos: np.ndarray, coverage_tracker, policy_action: np.ndarray) -> tuple[float, float]:
    _ = coverage_tracker
    action = np.asarray(policy_action, dtype=np.float32).reshape(-1)
    if action.size >= 2:
        return float(current_pos[0] + action[0]), float(current_pos[1] + action[1])
    return float(current_pos[0]), float(current_pos[1])


def clamp_norm_xy(vec_xy: np.ndarray, cap_m: float) -> np.ndarray:
    v = np.asarray(vec_xy, dtype=np.float32).reshape(2)
    cap = max(0.0, float(cap_m))
    n = float(np.linalg.norm(v))
    if n <= 1e-6 or cap <= 1e-9:
        return np.zeros(2, dtype=np.float32)
    if n <= cap:
        return v.astype(np.float32)
    return (v * (cap / n)).astype(np.float32)


def unit_xy(vec_xy: np.ndarray, default_xy: np.ndarray | None = None) -> np.ndarray:
    v = np.asarray(vec_xy, dtype=np.float32).reshape(2)
    n = float(np.linalg.norm(v))
    if n > 1e-6:
        return (v / n).astype(np.float32)
    if default_xy is not None:
        d = np.asarray(default_xy, dtype=np.float32).reshape(2)
        dn = float(np.linalg.norm(d))
        if dn > 1e-6:
            return (d / dn).astype(np.float32)
    return np.asarray([1.0, 0.0], dtype=np.float32)
