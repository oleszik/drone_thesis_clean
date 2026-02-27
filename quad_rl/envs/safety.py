from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def _clamp_xy(vx: float, vy: float, limit: float) -> Tuple[float, float]:
    mag = float(np.hypot(vx, vy))
    if mag <= limit or mag <= 1e-8:
        return vx, vy
    scale = limit / mag
    return vx * scale, vy * scale


def apply_safety(
    action: np.ndarray,
    cfg,
    task_name: str,
    state_pos: np.ndarray | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    safe = np.asarray(action, dtype=np.float32).copy()
    task_key = (task_name or "").strip().lower()
    scan_disable_wall_clamp = bool(getattr(cfg, "scan_disable_wall_clamp", False))

    wall_scale = 1.0
    wall_d_edge = None
    scan_d_edge = None
    scan_wall_active = False
    apply_v_xy_clamp = True
    v_xy_limit = float(cfg.v_xy_max)
    if task_key == "scan":
        scan_v_xy = getattr(cfg, "scan_v_xy_max", None)
        if scan_v_xy is not None:
            v_xy_limit = float(scan_v_xy)
        scan_base_v_xy_limit = float(v_xy_limit)
        if scan_disable_wall_clamp:
            apply_v_xy_clamp = False
        else:
            wall_margin = getattr(cfg, "scan_wall_margin", None)
            if wall_margin is None:
                wall_margin = getattr(cfg, "seq_wall_margin", 0.0)
            wall_margin = float(wall_margin)
            if state_pos is None or wall_margin <= 0.0:
                apply_v_xy_clamp = False
            else:
                x, y = float(state_pos[0]), float(state_pos[1])
                scan_d_edge = float(cfg.world_xy_bound) - max(abs(x), abs(y))
                scan_wall_active = bool(scan_d_edge < wall_margin)
                apply_v_xy_clamp = bool(scan_wall_active)
                if scan_wall_active:
                    t = float(np.clip(scan_d_edge / wall_margin, 0.0, 1.0))
                    scan_vmin = getattr(cfg, "scan_wall_vmin_scale", None)
                    if scan_vmin is None:
                        scan_vmin = getattr(cfg, "seq_wall_vmin_scale", 1.0)
                    scan_power = getattr(cfg, "scan_wall_power", None)
                    if scan_power is None:
                        scan_power = getattr(cfg, "seq_wall_power", 1.0)
                    vmin_scale = float(np.clip(scan_vmin, 0.0, 1.0))
                    wall_power = max(float(scan_power), 1e-6)
                    wall_scale = vmin_scale + (1.0 - vmin_scale) * (t**wall_power)
                    v_xy_limit = scan_base_v_xy_limit * wall_scale
    if task_key == "sequence" and state_pos is not None:
        wall_margin = float(getattr(cfg, "seq_wall_margin", 0.0))
        if wall_margin > 0.0:
            x, y = float(state_pos[0]), float(state_pos[1])
            wall_d_edge = float(cfg.world_xy_bound) - max(abs(x), abs(y))
            if wall_d_edge < wall_margin:
                t = float(np.clip(wall_d_edge / wall_margin, 0.0, 1.0))
                vmin_scale = float(np.clip(getattr(cfg, "seq_wall_vmin_scale", 1.0), 0.0, 1.0))
                wall_power = max(float(getattr(cfg, "seq_wall_power", 1.0)), 1e-6)
                wall_scale = vmin_scale + (1.0 - vmin_scale) * (t**wall_power)
                v_xy_limit = float(cfg.v_xy_max) * wall_scale

    vx_raw = float(safe[0])
    vy_raw = float(safe[1])
    if task_key == "scan":
        vx, vy = vx_raw, vy_raw
        if apply_v_xy_clamp:
            v = np.array([vx_raw, vy_raw], dtype=float)
            spd = float(np.linalg.norm(v) + 1e-9)
            if spd > v_xy_limit:
                v *= float(v_xy_limit / spd)
            vx, vy = float(v[0]), float(v[1])
    elif apply_v_xy_clamp:
        vx, vy = _clamp_xy(vx_raw, vy_raw, v_xy_limit)
    else:
        vx, vy = vx_raw, vy_raw
    safe[0] = vx
    safe[1] = vy
    safe[2] = float(np.clip(safe[2], -float(cfg.v_z_max), float(cfg.v_z_max)))
    safe[3] = float(np.clip(safe[3], -float(cfg.yaw_rate_max), float(cfg.yaw_rate_max)))

    z_locked = False
    if task_key == "waypoint" and bool(cfg.waypoint_lock_altitude):
        safe[2] = 0.0
        z_locked = True
    if task_key == "sequence" and bool(cfg.seq_lock_altitude):
        safe[2] = 0.0
        z_locked = True

    return safe, {
        "safety_z_locked": z_locked,
        "seq_wall_scale": wall_scale,
        "seq_wall_d_edge": wall_d_edge,
        "scan_d_edge": scan_d_edge,
        "scan_wall_active": scan_wall_active,
        "v_xy_limit_eff": float(v_xy_limit),
        "scan_wall_clamp_disabled": bool(task_key == "scan" and scan_disable_wall_clamp),
    }


def check_crash(state, cfg) -> Tuple[bool, Dict[str, Any]]:
    x, y, z = float(state.pos[0]), float(state.pos[1]), float(state.pos[2])
    out_of_bounds = (
        abs(x) > float(cfg.world_xy_bound)
        or abs(y) > float(cfg.world_xy_bound)
        or z > float(cfg.world_z_max)
        or z < float(cfg.world_z_min) - 0.05
    )
    tilt_exceeded = (
        abs(float(state.roll)) > float(cfg.crash_tilt_limit)
        or abs(float(state.pitch)) > float(cfg.crash_tilt_limit)
    )
    crash = bool(out_of_bounds or tilt_exceeded)
    return crash, {"out_of_bounds": out_of_bounds, "tilt_exceeded": tilt_exceeded}


def clamp_xy_inside_bounds(state, cfg, task_name: str) -> Dict[str, Any]:
    """Clamp XY position just inside boundary and report whether boundary was touched."""
    touched = False
    eps = 1e-3
    limit = max(0.0, float(cfg.world_xy_bound) - eps)
    x = float(state.pos[0])
    y = float(state.pos[1])
    x_clamped = float(np.clip(x, -limit, limit))
    y_clamped = float(np.clip(y, -limit, limit))
    if x_clamped != x:
        touched = True
        state.pos[0] = x_clamped
    if y_clamped != y:
        touched = True
        state.pos[1] = y_clamped
    return {"oob_touch": touched, "xy_clamp_applied": True}
