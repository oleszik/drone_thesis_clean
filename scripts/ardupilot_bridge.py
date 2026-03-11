from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from quad_rl.curriculum.presets import get_preset, list_presets
from quad_rl.utils.config_overrides import apply_overrides, parse_override_pairs
from quad_rl.utils.paths import normalize_model_path
from quad_rl.utils.scan_scale_profile import (
    apply_scan_obs_profile,
    assert_scan_obs_profile,
    effective_scan_max_steps,
    get_scan_path_scale_upper,
    resolve_scan_production_model_path,
)


def clamp_target_xy(x: float, y: float, half_w: float, half_h: float, margin: float) -> tuple[float, float, bool]:
    lim_x = max(0.0, float(half_w) - max(0.0, float(margin)))
    lim_y = max(0.0, float(half_h) - max(0.0, float(margin)))
    xc = float(np.clip(float(x), -lim_x, lim_x))
    yc = float(np.clip(float(y), -lim_y, lim_y))
    return xc, yc, bool(abs(xc - float(x)) > 1e-9 or abs(yc - float(y)) > 1e-9)


def bounds_minmax_xy(
    origin_x: float,
    origin_y: float,
    bounds_w: float,
    bounds_h: float,
    margin: float,
) -> tuple[float, float, float, float]:
    half_w = 0.5 * float(bounds_w)
    half_h = 0.5 * float(bounds_h)
    m = max(0.0, float(margin))
    x_min = float(origin_x) - half_w + m
    x_max = float(origin_x) + half_w - m
    y_min = float(origin_y) - half_h + m
    y_max = float(origin_y) + half_h - m
    if x_min > x_max:
        mid = 0.5 * (x_min + x_max)
        x_min = mid
        x_max = mid
    if y_min > y_max:
        mid = 0.5 * (y_min + y_max)
        y_min = mid
        y_max = mid
    return float(x_min), float(x_max), float(y_min), float(y_max)


def scan_center_offset_for_reference(
    reference: str,
    bounds_w: float,
    bounds_h: float,
    margin: float,
) -> tuple[float, float]:
    ref = str(reference or "center").strip().lower()
    half_w = 0.5 * float(bounds_w)
    half_h = 0.5 * float(bounds_h)
    dx = max(0.0, half_w - max(0.0, float(margin)))
    dy = max(0.0, half_h - max(0.0, float(margin)))
    # Offset from a reference point to area center.
    # Example: if reference is SW corner, center is +dx,+dy from that point.
    if ref == "sw":
        return float(dx), float(dy)
    if ref == "se":
        return float(-dx), float(dy)
    if ref == "nw":
        return float(dx), float(-dy)
    if ref == "ne":
        return float(-dx), float(-dy)
    return 0.0, 0.0


def clamp_target_xy_minmax(
    x: float,
    y: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[float, float, bool]:
    xc = float(np.clip(float(x), float(x_min), float(x_max)))
    yc = float(np.clip(float(y), float(y_min), float(y_max)))
    return xc, yc, bool(abs(xc - float(x)) > 1e-9 or abs(yc - float(y)) > 1e-9)


def _safe_slug(text: str) -> str:
    raw = (text or "").strip()
    slug = "".join(ch if (ch.isalnum() or ch in "-_") else "-" for ch in raw)
    slug = slug.strip("-_")
    return slug or "run"


def _next_free_dir(path: Path) -> Path:
    if not path.exists():
        return path
    i = 1
    while True:
        cand = path.with_name(f"{path.name}_{i}")
        if not cand.exists():
            return cand
        i += 1


def _flag_was_set(argv: list[str], flag: str) -> bool:
    return any((tok == flag) or str(tok).startswith(f"{flag}=") for tok in argv)


def _selected_profile_for_scale(scale: float) -> str:
    return "patch7_scale2" if float(scale) >= 2.0 else "patch5_default"


def _iter_opt_summary_paths() -> list[Path]:
    root = Path("runs") / "ardupilot_scan_opt_suite"
    if not root.exists():
        return []
    return sorted(root.glob("*/opt_summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def _pinned_opt_summary_path_for_profile(profile_name: str) -> Path | None:
    root = Path("runs") / "production_ardupilot_defaults"
    if not root.exists():
        return None
    bucket = "scale2" if str(profile_name) == "patch7_scale2" else "scale1"
    path = root / f"{bucket}_opt_summary.json"
    return path if path.exists() else None


def _winner_from_opt_summary(data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    defaults: dict[str, Any] = {
        "lookahead_enable": 1,
        "step_sched_enable": 1,
        "lane_keep_enable": 0,
        "adaptive_tracking": 0,
        "lane_kp": 0.4,
        "corner_vxy_cap": 0.8,
    }
    winner_arm = "A_baseline"
    ranked = data.get("ranked", [])
    if not isinstance(ranked, list) or len(ranked) <= 0:
        return winner_arm, defaults
    top = ranked[0]
    if not isinstance(top, dict):
        return winner_arm, defaults
    winner_arm = str(top.get("arm", winner_arm))
    cfg = top.get("config", {})
    if isinstance(cfg, dict):
        for k in ("lookahead_enable", "step_sched_enable", "lane_keep_enable", "adaptive_tracking"):
            if k in cfg:
                defaults[k] = int(cfg[k])
        if "lane_kp" in cfg:
            defaults["lane_kp"] = float(cfg["lane_kp"])
        if "corner_vxy_cap" in cfg:
            defaults["corner_vxy_cap"] = float(cfg["corner_vxy_cap"])
    return winner_arm, defaults


def _summary_selected_profile(data: dict[str, Any]) -> str | None:
    meta = data.get("metadata", {})
    if isinstance(meta, dict):
        sel = str(meta.get("selected_profile", "")).strip()
        if sel:
            return sel
        scale_meta = meta.get("scan_path_len_scale", None)
        if scale_meta is not None:
            try:
                return _selected_profile_for_scale(float(scale_meta))
            except Exception:
                pass
    cfg = data.get("config", {})
    if isinstance(cfg, dict):
        scale_cfg = cfg.get("scan_path_len_scale", None)
        if scale_cfg is not None:
            try:
                return _selected_profile_for_scale(float(scale_cfg))
            except Exception:
                return None
    return None


def _resolve_sitl_recommended_profile(source: str, scale_upper: float) -> tuple[str, dict[str, Any], str, bool]:
    expected_profile = _selected_profile_for_scale(float(scale_upper))
    src_raw = str(source or "latest").strip()
    src = src_raw.lower()

    if src.startswith("path:"):
        raw_path = src_raw[5:].strip()
        p = Path(raw_path).expanduser()
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                arm, cfg = _winner_from_opt_summary(data)
                prof = _summary_selected_profile(data)
                return arm, cfg, f"path:{p}", bool(prof == expected_profile)
            except Exception:
                pass
        arm, cfg = _winner_from_opt_summary({})
        return arm, cfg, f"path:{p}", False

    if src not in {"latest", ""}:
        src = "latest"

    pinned = _pinned_opt_summary_path_for_profile(expected_profile)
    if pinned is not None:
        try:
            data = json.loads(pinned.read_text(encoding="utf-8"))
            prof = _summary_selected_profile(data)
            if prof == expected_profile:
                arm, cfg = _winner_from_opt_summary(data)
                return arm, cfg, f"latest:pinned:{pinned}", True
        except Exception:
            pass

    for p in _iter_opt_summary_paths():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            prof = _summary_selected_profile(data)
            if prof != expected_profile:
                continue
            arm, cfg = _winner_from_opt_summary(data)
            return arm, cfg, f"latest:{p}", True
        except Exception:
            continue

    arm, cfg = _winner_from_opt_summary({})
    return arm, cfg, "latest:fallback", False


def _auto_step_len(scale: float) -> float:
    return 5.0 if float(scale) >= 2.0 else 3.0


def _auto_accept_radius(scale: float) -> float:
    return 1.25 if float(scale) >= 2.0 else 0.75


def _auto_vxy_cap(scale: float) -> float:
    return 1.5 if float(scale) >= 2.0 else 1.2


@dataclass
class TelemetryState:
    x: float = 0.0
    y: float = 0.0
    z: float = -10.0
    local_pos_known: bool = False
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw: float = 0.0
    mode: str = "UNKNOWN"
    custom_mode: int | None = None
    system_status: int | None = None
    armed: bool = False
    relative_alt_m: float | None = None
    relative_alt_source: str = "none"
    home_alt_m: float | None = None
    vfr_alt_m: float | None = None
    gps_ok: bool = False
    ekf_ok: bool = False
    gps_known: bool = False
    ekf_known: bool = False
    last_heartbeat_time_s: float = 0.0
    hb_base_mode: int | None = None
    hb_custom_mode: int | None = None
    hb_system_status: int | None = None
    ack_verbosity: str = "important"
    ack_smi_suppressed_count: int = 0
    ack_smi_last_report_s: float = 0.0
    last_update_s: float = 0.0


class PreflightError(RuntimeError):
    pass


def _heartbeat_snapshot(state: TelemetryState) -> dict[str, Any]:
    return {
        "mode": str(state.mode),
        "custom_mode": None if state.custom_mode is None else int(state.custom_mode),
        "armed": int(bool(state.armed)),
        "system_status": None if state.system_status is None else int(state.system_status),
        "last_heartbeat_time_s": float(state.last_heartbeat_time_s),
        "hb_base_mode": None if state.hb_base_mode is None else int(state.hb_base_mode),
        "hb_custom_mode": None if state.hb_custom_mode is None else int(state.hb_custom_mode),
        "hb_system_status": None if state.hb_system_status is None else int(state.hb_system_status),
        "relative_alt_m": None if state.relative_alt_m is None else float(state.relative_alt_m),
        "relative_alt_source": str(state.relative_alt_source or "none"),
    }


def _mav_enum_name(enum_table, key: int, fallback: str) -> str:
    try:
        table = getattr(enum_table, "__getitem__", None)
        if table is None:
            return fallback
        item = enum_table[key]
        name = str(getattr(item, "name", "")).strip()
        return name if name else fallback
    except Exception:
        return fallback


def _mav_cmd_name(mavutil, cmd_id: int) -> str:
    try:
        enums = getattr(mavutil.mavlink, "enums", {})
        if isinstance(enums, dict) and "MAV_CMD" in enums:
            return _mav_enum_name(enums["MAV_CMD"], int(cmd_id), f"CMD_{int(cmd_id)}")
    except Exception:
        pass
    return f"CMD_{int(cmd_id)}"


def _mav_result_name(mavutil, result: int) -> str:
    try:
        enums = getattr(mavutil.mavlink, "enums", {})
        if isinstance(enums, dict) and "MAV_RESULT" in enums:
            return _mav_enum_name(enums["MAV_RESULT"], int(result), f"RESULT_{int(result)}")
    except Exception:
        pass
    return f"RESULT_{int(result)}"


def _critical_ack_commands(mavutil) -> set[int]:
    out: set[int] = set()
    for name in ("MAV_CMD_DO_SET_MODE", "MAV_CMD_COMPONENT_ARM_DISARM", "MAV_CMD_NAV_TAKEOFF", "MAV_CMD_NAV_LAND"):
        try:
            out.add(int(getattr(mavutil.mavlink, name)))
        except Exception:
            continue
    return out


def _request_message_intervals_once(
    mav,
    mavutil,
    rate_hz: float,
    already_sent: bool,
) -> tuple[bool, list[str]]:
    if bool(already_sent):
        return True, []
    hz = max(1e-3, float(rate_hz))
    interval_us = int(round(1_000_000.0 / hz))
    requested: list[str] = []
    targets = [
        ("GLOBAL_POSITION_INT", 33),
        ("VFR_HUD", 74),
    ]
    try:
        cmd_set_interval = int(getattr(mavutil.mavlink, "MAV_CMD_SET_MESSAGE_INTERVAL"))
    except Exception:
        cmd_set_interval = 511
    for name, msg_id in targets:
        try:
            mav.mav.command_long_send(
                mav.target_system,
                mav.target_component,
                cmd_set_interval,
                0,
                float(msg_id),
                float(interval_us),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            requested.append(str(name))
        except Exception:
            continue
    if requested:
        print(f"[bridge] requested message intervals @ {hz:.2f}Hz: {', '.join(requested)}")
    return True, requested


class CoverageGridTracker:
    def __init__(
        self,
        width_m: float,
        height_m: float,
        cell_size_m: float,
        radius_m: float,
        boundary_cells: int = 2,
        track_pass_counts: bool = False,
    ) -> None:
        self.width_m = max(1e-3, float(width_m))
        self.height_m = max(1e-3, float(height_m))
        self.cell_size = max(1e-3, float(cell_size_m))
        self.radius = max(0.0, float(radius_m))
        self.x_min = -0.5 * self.width_m
        self.y_min = -0.5 * self.height_m
        self.nx = max(1, int(np.ceil(self.width_m / self.cell_size)))
        self.ny = max(1, int(np.ceil(self.height_m / self.cell_size)))
        self.covered = np.zeros((self.nx, self.ny), dtype=bool)
        self.pass_counts = np.zeros((self.nx, self.ny), dtype=np.int32) if bool(track_pass_counts) else None
        self.covered_cells = 0
        self.revisit_steps = 0
        ix = np.arange(self.nx, dtype=np.int32)[:, None]
        iy = np.arange(self.ny, dtype=np.int32)[None, :]
        edge = max(1, int(boundary_cells))
        self.boundary_mask = (ix < edge) | (ix >= (self.nx - edge)) | (iy < edge) | (iy >= (self.ny - edge))

    @property
    def total_cells(self) -> int:
        return int(max(1, self.nx * self.ny))

    def _mark_disk(self, x: float, y: float) -> tuple[int, int]:
        return self._mark_disk_radius(float(x), float(y), float(self.radius))

    def _mark_disk_radius(self, x: float, y: float, radius_m: float, count_hits: bool = False) -> tuple[int, int]:
        cs = self.cell_size
        r = max(0.0, float(radius_m))
        r2 = r * r
        ix0 = int(np.floor((x - r - self.x_min) / cs))
        ix1 = int(np.floor((x + r - self.x_min) / cs))
        iy0 = int(np.floor((y - r - self.y_min) / cs))
        iy1 = int(np.floor((y + r - self.y_min) / cs))
        ix0 = int(np.clip(ix0, 0, self.nx - 1))
        ix1 = int(np.clip(ix1, 0, self.nx - 1))
        iy0 = int(np.clip(iy0, 0, self.ny - 1))
        iy1 = int(np.clip(iy1, 0, self.ny - 1))
        new_hits = 0
        revisit_hits = 0
        for ix in range(ix0, ix1 + 1):
            cx = self.x_min + (float(ix) + 0.5) * cs
            dx = cx - x
            for iy in range(iy0, iy1 + 1):
                cy = self.y_min + (float(iy) + 0.5) * cs
                dy = cy - y
                if (dx * dx + dy * dy) > r2:
                    continue
                if bool(count_hits) and (self.pass_counts is not None):
                    self.pass_counts[ix, iy] = int(self.pass_counts[ix, iy] + 1)
                if bool(self.covered[ix, iy]):
                    revisit_hits += 1
                else:
                    self.covered[ix, iy] = True
                    new_hits += 1
        return int(new_hits), int(revisit_hits)

    def _mark_ellipse(self, x: float, y: float, rx_m: float, ry_m: float, count_hits: bool = False) -> tuple[int, int]:
        cs = self.cell_size
        rx = max(1e-6, float(rx_m))
        ry = max(1e-6, float(ry_m))
        ix0 = int(np.floor((x - rx - self.x_min) / cs))
        ix1 = int(np.floor((x + rx - self.x_min) / cs))
        iy0 = int(np.floor((y - ry - self.y_min) / cs))
        iy1 = int(np.floor((y + ry - self.y_min) / cs))
        ix0 = int(np.clip(ix0, 0, self.nx - 1))
        ix1 = int(np.clip(ix1, 0, self.nx - 1))
        iy0 = int(np.clip(iy0, 0, self.ny - 1))
        iy1 = int(np.clip(iy1, 0, self.ny - 1))
        new_hits = 0
        revisit_hits = 0
        inv_rx2 = 1.0 / max(1e-9, rx * rx)
        inv_ry2 = 1.0 / max(1e-9, ry * ry)
        for ix in range(ix0, ix1 + 1):
            cx = self.x_min + (float(ix) + 0.5) * cs
            dx = cx - x
            for iy in range(iy0, iy1 + 1):
                cy = self.y_min + (float(iy) + 0.5) * cs
                dy = cy - y
                if ((dx * dx) * inv_rx2 + (dy * dy) * inv_ry2) > 1.0:
                    continue
                if bool(count_hits) and (self.pass_counts is not None):
                    self.pass_counts[ix, iy] = int(self.pass_counts[ix, iy] + 1)
                if bool(self.covered[ix, iy]):
                    revisit_hits += 1
                else:
                    self.covered[ix, iy] = True
                    new_hits += 1
        return int(new_hits), int(revisit_hits)

    def update(self, x: float, y: float) -> tuple[int, int]:
        new_hits, revisit_hits = self._mark_disk(float(x), float(y))
        if new_hits > 0:
            self.covered_cells += int(new_hits)
        elif revisit_hits > 0:
            self.revisit_steps += 1
        return int(new_hits), int(revisit_hits)

    def update_disk_radius(self, x: float, y: float, radius_m: float, count_hits: bool = False) -> tuple[int, int]:
        new_hits, revisit_hits = self._mark_disk_radius(float(x), float(y), float(radius_m), count_hits=bool(count_hits))
        if new_hits > 0:
            self.covered_cells += int(new_hits)
        elif revisit_hits > 0:
            self.revisit_steps += 1
        return int(new_hits), int(revisit_hits)

    def update_ellipse(self, x: float, y: float, rx_m: float, ry_m: float, count_hits: bool = False) -> tuple[int, int]:
        # Assumption: footprint axes are aligned with LOCAL_NED XY (yaw orientation ignored).
        new_hits, revisit_hits = self._mark_ellipse(
            float(x),
            float(y),
            float(rx_m),
            float(ry_m),
            count_hits=bool(count_hits),
        )
        if new_hits > 0:
            self.covered_cells += int(new_hits)
        elif revisit_hits > 0:
            self.revisit_steps += 1
        return int(new_hits), int(revisit_hits)

    def coverage_frac(self) -> float:
        return float(self.covered_cells / max(1, self.total_cells))

    def coverage_frac_at_least(self, min_passes: int) -> float:
        k = max(1, int(min_passes))
        if self.pass_counts is None:
            if k <= 1:
                return self.coverage_frac()
            return 0.0
        return float(np.count_nonzero(self.pass_counts >= k) / max(1, self.total_cells))

    def boundary_covered_frac(self) -> float:
        mask = self.boundary_mask
        total = int(np.count_nonzero(mask))
        if total <= 0:
            return 0.0
        return float(np.count_nonzero(self.covered & mask) / max(1, total))

    def interior_covered_frac(self) -> float:
        mask = ~self.boundary_mask
        total = int(np.count_nonzero(mask))
        if total <= 0:
            return 0.0
        return float(np.count_nonzero(self.covered & mask) / max(1, total))

    def local_patch(self, x: float, y: float, patch_size: int) -> np.ndarray:
        p = max(1, int(patch_size))
        if p % 2 == 0:
            p += 1
        half = p // 2
        ix_center = int(np.floor((x - self.x_min) / self.cell_size))
        iy_center = int(np.floor((y - self.y_min) / self.cell_size))
        patch = np.ones((p, p), dtype=np.float32)
        for dx in range(-half, half + 1):
            gx = ix_center + dx
            if gx < 0 or gx >= self.nx:
                continue
            for dy in range(-half, half + 1):
                gy = iy_center + dy
                if gy < 0 or gy >= self.ny:
                    continue
                patch[dx + half, dy + half] = 1.0 if bool(self.covered[gx, gy]) else 0.0
        return patch.reshape(-1)

    def boundary_features(self, x: float, y: float) -> tuple[float, float]:
        x_max = self.x_min + float(self.nx) * self.cell_size
        y_max = self.y_min + float(self.ny) * self.cell_size
        d_x = min(float(x) - self.x_min, x_max - float(x))
        d_y = min(float(y) - self.y_min, y_max - float(y))
        span_x = max(1e-6, x_max - self.x_min)
        span_y = max(1e-6, y_max - self.y_min)
        return float(np.clip(d_x / span_x, 0.0, 1.0)), float(np.clip(d_y / span_y, 0.0, 1.0))


def _append_dense_line(points: list[np.ndarray], a: np.ndarray, b: np.ndarray, ds: float) -> None:
    dxy = b[:2] - a[:2]
    seg_len = float(np.linalg.norm(dxy))
    if seg_len <= 1e-6:
        return
    n = max(1, int(np.ceil(seg_len / max(1e-3, float(ds)))))
    for k in range(1, n + 1):
        t = float(k / n)
        p = (1.0 - t) * a + t * b
        points.append(np.asarray(p, dtype=np.float32))


def _append_dense_arc(
    points: list[np.ndarray],
    p_start: np.ndarray,
    p_end: np.ndarray,
    center_xy: np.ndarray,
    z: float,
    ccw: bool,
    ds: float,
) -> None:
    cx, cy = float(center_xy[0]), float(center_xy[1])
    a0 = float(math.atan2(float(p_start[1]) - cy, float(p_start[0]) - cx))
    a1 = float(math.atan2(float(p_end[1]) - cy, float(p_end[0]) - cx))
    if ccw:
        while a1 <= a0:
            a1 += 2.0 * math.pi
        dtheta = a1 - a0
    else:
        while a1 >= a0:
            a1 -= 2.0 * math.pi
        dtheta = a1 - a0
    r = float(np.linalg.norm(np.asarray([float(p_start[0]) - cx, float(p_start[1]) - cy], dtype=np.float64)))
    arc_len = abs(dtheta) * max(1e-6, r)
    n = max(1, int(np.ceil(arc_len / max(1e-3, float(ds)))))
    for k in range(1, n + 1):
        t = float(k / n)
        a = a0 + dtheta * t
        x = cx + r * math.cos(a)
        y = cy + r * math.sin(a)
        points.append(np.asarray([x, y, z], dtype=np.float32))


def _build_lawnmower_targets(
    bounds_w: float,
    bounds_h: float,
    margin: float,
    step_len: float,
    alt_m: float,
    turn_style: str = "sharp",
    turn_radius_m: float = 2.0,
) -> list[np.ndarray]:
    half_w = 0.5 * float(bounds_w)
    half_h = 0.5 * float(bounds_h)
    x_min = -half_w + float(margin)
    x_max = half_w - float(margin)
    y_min = -half_h + float(margin)
    y_max = half_h - float(margin)
    z = -abs(float(alt_m))
    if x_min >= x_max or y_min >= y_max:
        return [np.asarray([0.0, 0.0, z], dtype=np.float32)]

    lane_spacing = max(0.5, float(step_len))
    x_vals = list(np.arange(x_min, x_max + 1e-6, lane_spacing))
    if len(x_vals) == 0 or abs(float(x_vals[-1]) - x_max) > 1e-6:
        x_vals.append(float(x_max))
    poly = []
    for i, x in enumerate(x_vals):
        if i % 2 == 0:
            poly.append(np.asarray([x, y_min, z], dtype=np.float32))
            poly.append(np.asarray([x, y_max, z], dtype=np.float32))
        else:
            poly.append(np.asarray([x, y_max, z], dtype=np.float32))
            poly.append(np.asarray([x, y_min, z], dtype=np.float32))
    if len(poly) <= 1:
        return poly

    dense: list[np.ndarray] = [poly[0]]
    ds = max(0.2, float(step_len))
    turn_style_eff = str(turn_style).strip().lower()
    if turn_style_eff != "arc":
        for i in range(len(poly) - 1):
            _append_dense_line(dense, poly[i], poly[i + 1], ds)
        return dense

    # Arc style: fillet each 90-degree corner to keep speed/yaw smoother through lane changes.
    cursor = np.asarray(poly[0], dtype=np.float32)
    for i in range(1, len(poly) - 1):
        prev_p = np.asarray(poly[i - 1], dtype=np.float32)
        corner_p = np.asarray(poly[i], dtype=np.float32)
        next_p = np.asarray(poly[i + 1], dtype=np.float32)
        vin = np.asarray([float(corner_p[0] - prev_p[0]), float(corner_p[1] - prev_p[1])], dtype=np.float64)
        vout = np.asarray([float(next_p[0] - corner_p[0]), float(next_p[1] - corner_p[1])], dtype=np.float64)
        len_in = float(np.linalg.norm(vin))
        len_out = float(np.linalg.norm(vout))
        if len_in <= 1e-6 or len_out <= 1e-6:
            _append_dense_line(dense, cursor, corner_p, ds)
            cursor = corner_p
            continue
        din = vin / len_in
        dout = vout / len_out
        dot = float(np.clip(np.dot(din, dout), -1.0, 1.0))
        is_corner = bool(abs(dot) < 0.25)  # near-orthogonal turn
        r = min(max(0.0, float(turn_radius_m)), 0.45 * len_in, 0.45 * len_out)
        if (not is_corner) or r <= 1e-4:
            _append_dense_line(dense, cursor, corner_p, ds)
            cursor = corner_p
            continue
        p1 = np.asarray(
            [float(corner_p[0] - din[0] * r), float(corner_p[1] - din[1] * r), float(corner_p[2])],
            dtype=np.float32,
        )
        p2 = np.asarray(
            [float(corner_p[0] + dout[0] * r), float(corner_p[1] + dout[1] * r), float(corner_p[2])],
            dtype=np.float32,
        )
        center_xy = np.asarray(
            [float(corner_p[0] - din[0] * r + dout[0] * r), float(corner_p[1] - din[1] * r + dout[1] * r)],
            dtype=np.float64,
        )
        _append_dense_line(dense, cursor, p1, ds)
        cross_z = float(din[0] * dout[1] - din[1] * dout[0])
        _append_dense_arc(
            dense,
            p_start=p1,
            p_end=p2,
            center_xy=center_xy,
            z=float(corner_p[2]),
            ccw=bool(cross_z > 0.0),
            ds=ds,
        )
        cursor = p2

    _append_dense_line(dense, cursor, np.asarray(poly[-1], dtype=np.float32), ds)
    return dense


def compute_next_target(current_pos: np.ndarray, coverage_tracker: CoverageGridTracker, policy_action: np.ndarray) -> tuple[float, float]:
    _ = coverage_tracker
    action = np.asarray(policy_action, dtype=np.float32).reshape(-1)
    if action.size >= 2:
        return float(current_pos[0] + action[0]), float(current_pos[1] + action[1])
    return float(current_pos[0]), float(current_pos[1])


def _clamp_norm_xy(vec_xy: np.ndarray, cap_m: float) -> np.ndarray:
    v = np.asarray(vec_xy, dtype=np.float32).reshape(2)
    cap = max(0.0, float(cap_m))
    n = float(np.linalg.norm(v))
    if n <= 1e-6 or cap <= 1e-9:
        return np.zeros(2, dtype=np.float32)
    if n <= cap:
        return v.astype(np.float32)
    return (v * (cap / n)).astype(np.float32)


def _unit_xy(vec_xy: np.ndarray, default_xy: np.ndarray | None = None) -> np.ndarray:
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


def _connect_mavlink(connection: str):
    from pymavlink import mavutil

    mav = mavutil.mavlink_connection(connection)
    return mav, mavutil


def _maybe_switch_to_sitl_master(connection: str, prefer_sitl_tcp: int, probe_timeout_s: float = 1.0) -> str:
    requested = str(connection).strip()
    if int(prefer_sitl_tcp) != 1:
        return requested
    if requested.lower() != "udp:127.0.0.1:14550":
        return requested
    sitl_master = "tcp:127.0.0.1:5760"
    print(f"[bridge] probing SITL master {sitl_master} for heartbeat (timeout={float(probe_timeout_s):.1f}s)")
    try:
        from pymavlink import mavutil

        probe = mavutil.mavlink_connection(sitl_master)
        try:
            probe.wait_heartbeat(timeout=float(probe_timeout_s))
            print("[bridge] switching to SITL master tcp:127.0.0.1:5760 for reliable mode/arming")
            return sitl_master
        except Exception:
            return requested
        finally:
            try:
                probe.close()
            except Exception:
                pass
    except Exception:
        return requested


def _mode_string(msg, mavutil) -> str:
    try:
        return str(mavutil.mode_string_v10(msg))
    except Exception:
        return "UNKNOWN"


def _set_mode(mav, mavutil, mode_name: str) -> bool:
    mode = str(mode_name).upper()
    try:
        mapping = mav.mode_mapping() or {}
        if mode in mapping:
            mav.mav.set_mode_send(
                mav.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                int(mapping[mode]),
            )
            return True
    except Exception:
        pass
    try:
        mav.set_mode_apm(mode)
        return True
    except Exception:
        return False


def _log_preflight_event(events: list[dict[str, Any]], event: str, status: str, **extra: Any) -> None:
    row = {
        "t_wall_s": float(time.time()),
        "event": str(event),
        "status": str(status),
    }
    row.update(extra)
    events.append(row)


def _handle_mavlink_message(msg, mavutil, state: TelemetryState) -> None:
    mtype = msg.get_type()
    state.last_update_s = time.monotonic()
    if mtype == "LOCAL_POSITION_NED":
        state.x = float(msg.x)
        state.y = float(msg.y)
        state.z = float(msg.z)
        state.local_pos_known = True
        state.vx = float(msg.vx)
        state.vy = float(msg.vy)
        state.vz = float(msg.vz)
    elif mtype == "ATTITUDE":
        state.yaw = float(msg.yaw)
    elif mtype == "HEARTBEAT":
        state.mode = _mode_string(msg, mavutil)
        base_mode = int(getattr(msg, "base_mode", 0))
        state.hb_base_mode = base_mode
        state.hb_custom_mode = int(getattr(msg, "custom_mode", 0))
        state.hb_system_status = int(getattr(msg, "system_status", 0))
        state.custom_mode = int(getattr(msg, "custom_mode", 0))
        state.system_status = int(getattr(msg, "system_status", 0))
        state.armed = bool(base_mode & int(mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED))
        state.last_heartbeat_time_s = time.monotonic()
    elif mtype == "GPS_RAW_INT":
        state.gps_known = True
        state.gps_ok = bool(int(getattr(msg, "fix_type", 0)) >= 3)
    elif mtype == "EKF_STATUS_REPORT":
        state.ekf_known = True
        flags = int(getattr(msg, "flags", 0))
        state.ekf_ok = bool(flags != 0)
    elif mtype == "GLOBAL_POSITION_INT":
        rel_mm = getattr(msg, "relative_alt", None)
        if rel_mm is not None:
            state.relative_alt_m = float(rel_mm) / 1000.0
            state.relative_alt_source = "GLOBAL_POSITION_INT"
        alt_mm = getattr(msg, "alt", None)
        if alt_mm is not None and state.home_alt_m is None and rel_mm is not None:
            state.home_alt_m = (float(alt_mm) - float(rel_mm)) / 1000.0
    elif mtype == "VFR_HUD":
        alt_vfr = float(getattr(msg, "alt", 0.0))
        state.vfr_alt_m = alt_vfr
        if str(state.relative_alt_source) != "GLOBAL_POSITION_INT":
            state.relative_alt_m = float(alt_vfr)
            state.relative_alt_source = "VFR_HUD"
    elif mtype == "COMMAND_ACK":
        cmd_id = int(getattr(msg, "command", -1))
        result = int(getattr(msg, "result", -1))
        accepted = bool(result == int(getattr(mavutil.mavlink, "MAV_RESULT_ACCEPTED", 0)))
        try:
            smi_id = int(getattr(mavutil.mavlink, "MAV_CMD_SET_MESSAGE_INTERVAL"))
        except Exception:
            smi_id = 511
        is_set_msg_interval = bool(cmd_id == 511 or cmd_id == smi_id)
        critical_cmds = _critical_ack_commands(mavutil)
        cmd_name = _mav_cmd_name(mavutil, cmd_id)
        res_name = _mav_result_name(mavutil, result)
        verbosity = str(state.ack_verbosity or "important").strip().lower()
        if verbosity == "all":
            print(f"[bridge] COMMAND_ACK {cmd_name} ({cmd_id}): {res_name} ({result})")
            return
        if not accepted:
            print(f"[bridge] COMMAND_ACK {cmd_name} ({cmd_id}): {res_name} ({result})")
            return
        if cmd_id in critical_cmds:
            print(f"[bridge] COMMAND_ACK {cmd_name} ({cmd_id}): {res_name} ({result})")
            return
        if is_set_msg_interval:
            state.ack_smi_suppressed_count += 1
            now = time.monotonic()
            if (now - float(state.ack_smi_last_report_s)) >= 5.0:
                print(f"[bridge] SET_MESSAGE_INTERVAL ACK spam suppressed ({int(state.ack_smi_suppressed_count)})")
                state.ack_smi_last_report_s = float(now)
                state.ack_smi_suppressed_count = 0


def _recv_one_blocking_update(mav, mavutil, state: TelemetryState, timeout_s: float = 1.0):
    msg = mav.recv_match(blocking=True, timeout=float(timeout_s))
    if msg is not None:
        _handle_mavlink_message(msg, mavutil, state)
    return msg


def _best_altitude_estimate_m(state: TelemetryState) -> tuple[float | None, str]:
    if state.relative_alt_m is not None:
        return float(state.relative_alt_m), str(state.relative_alt_source or "relative_alt")
    if state.vfr_alt_m is not None:
        return float(state.vfr_alt_m), "VFR_HUD"
    if bool(state.local_pos_known):
        return max(0.0, -float(state.z)), "LOCAL_POSITION_NED"
    return None, "none"


def _wait_for_heartbeat(mav, mavutil, state: TelemetryState, timeout_s: float) -> None:
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    while time.monotonic() < deadline:
        _ = _recv_one_blocking_update(mav, mavutil, state, timeout_s=1.0)
        if float(state.last_heartbeat_time_s) > 0.0:
            return
    raise PreflightError("timeout waiting heartbeat")


def _wait_command_ack(mav, mavutil, state: TelemetryState, command_id: int, timeout_s: float) -> tuple[bool, int | None]:
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    while time.monotonic() < deadline:
        msg = _recv_one_blocking_update(mav, mavutil, state, timeout_s=1.0)
        if msg is None or msg.get_type() != "COMMAND_ACK":
            continue
        if int(getattr(msg, "command", -1)) != int(command_id):
            continue
        result = int(getattr(msg, "result", -1))
        accepted = {
            int(mavutil.mavlink.MAV_RESULT_ACCEPTED),
            int(mavutil.mavlink.MAV_RESULT_IN_PROGRESS),
        }
        return bool(result in accepted), result
    return False, None


def _guided_mode_confirmed(mav, state: TelemetryState) -> bool:
    if "GUIDED" in str(state.mode).upper():
        return True
    try:
        mapping = mav.mode_mapping() or {}
        guided_custom = mapping.get("GUIDED", None)
        if guided_custom is not None and state.custom_mode is not None:
            return int(state.custom_mode) == int(guided_custom)
    except Exception:
        pass
    return False


def _wait_until_armed(mav, mavutil, state: TelemetryState, timeout_s: float) -> bool:
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    while time.monotonic() < deadline:
        _ = _recv_one_blocking_update(mav, mavutil, state, timeout_s=1.0)
        if bool(state.armed):
            return True
    return False


def _cmd_arm(mav, mavutil) -> None:
    mav.mav.command_long_send(
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def _cmd_takeoff(mav, mavutil, alt_m: float) -> None:
    mav.mav.command_long_send(
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        float(max(0.0, alt_m)),
    )


def _wait_takeoff_relative_alt(
    mav,
    mavutil,
    state: TelemetryState,
    target_alt_m: float,
    tolerance_m: float,
    timeout_s: float,
) -> bool:
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    tgt = max(0.0, float(target_alt_m))
    tol = max(0.05, float(tolerance_m))
    while time.monotonic() < deadline:
        _ = _recv_one_blocking_update(mav, mavutil, state, timeout_s=1.0)
        alt_up_m = 0.0 if state.relative_alt_m is None else max(0.0, float(state.relative_alt_m))
        if alt_up_m >= (tgt - tol):
            return True
    return False


def _preflight(
    *,
    args: argparse.Namespace,
    mav,
    mavutil,
    telem: TelemetryState,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    in_dry_run = bool(int(args.dry_run) == 1)
    min_air_alt_m = 1.0
    preflight_timeout_s = max(1.0, float(args.preflight_timeout_s))
    require_mode_known = bool(int(args.require_mode_known) == 1)
    auto_guided = bool(int(args.auto_guided))
    auto_arm = bool(int(args.auto_arm))
    auto_takeoff = bool(float(args.auto_takeoff_m) > 0.0)
    result: dict[str, Any] = {
        "takeoff_skipped": 0,
        "takeoff_skip_reason": "",
    }

    if in_dry_run:
        if auto_takeoff and (not auto_guided):
            auto_guided = True
            print("[bridge][preflight][dry] auto_takeoff requires GUIDED; forcing auto_guided=1.")
        print("[bridge][preflight][dry] heartbeat check would run.")
        _log_preflight_event(events, "heartbeat_wait", "planned", timeout_s=preflight_timeout_s)
        print("[bridge][preflight][dry] sequencing: heartbeat -> guided -> arm -> takeoff")
        if auto_guided:
            print("[bridge][preflight][dry] would set mode GUIDED and wait COMMAND_ACK.")
            _log_preflight_event(events, "set_guided", "planned")
        if auto_arm:
            print("[bridge][preflight][dry] would arm and wait COMMAND_ACK.")
            _log_preflight_event(events, "arm", "planned")
        if auto_takeoff:
            print(
                f"[bridge][preflight][dry] would command takeoff to {float(args.auto_takeoff_m):.2f} m and wait relative_alt."
            )
            _log_preflight_event(
                events,
                "takeoff",
                "planned",
                target_alt_m=float(args.auto_takeoff_m),
            )
        return result

    t_deadline = time.monotonic() + preflight_timeout_s
    print("[bridge][preflight] waiting for heartbeat...")
    _log_preflight_event(events, "heartbeat_wait", "start", timeout_s=preflight_timeout_s)
    remaining = max(0.1, t_deadline - time.monotonic())
    _wait_for_heartbeat(mav, mavutil, telem, timeout_s=remaining)
    _log_preflight_event(events, "heartbeat_wait", "ok", **_heartbeat_snapshot(telem))

    if auto_takeoff and (not _guided_mode_confirmed(mav, telem)) and (not auto_guided):
        auto_guided = True
        print("[bridge][preflight] auto_takeoff requires GUIDED; forcing auto_guided=1.")
        _log_preflight_event(events, "set_guided", "forced", reason="auto_takeoff_requires_guided")

    if auto_guided:
        _log_preflight_event(events, "set_guided", "start")
        if not _set_mode(mav, mavutil, "GUIDED"):
            _log_preflight_event(events, "set_guided", "fail")
            raise PreflightError("SET_MODE send failed")
        set_mode_cmd = int(mavutil.mavlink.MAV_CMD_DO_SET_MODE)
        ack_result: int | None = None
        ack_failed = False
        confirmed = False
        check_deadline = time.monotonic() + min(5.0, max(0.1, t_deadline - time.monotonic()))
        while time.monotonic() < check_deadline:
            msg = _recv_one_blocking_update(mav, mavutil, telem, timeout_s=1.0)
            if msg is not None and msg.get_type() == "COMMAND_ACK" and int(getattr(msg, "command", -1)) == set_mode_cmd:
                ack_result = int(getattr(msg, "result", -1))
                accepted = {
                    int(mavutil.mavlink.MAV_RESULT_ACCEPTED),
                    int(mavutil.mavlink.MAV_RESULT_IN_PROGRESS),
                }
                if ack_result not in accepted:
                    ack_failed = True
            if _guided_mode_confirmed(mav, telem):
                confirmed = True
                break
        if not confirmed:
            _log_preflight_event(
                events,
                "set_guided",
                "fail",
                ack_result=ack_result,
                ack_failed=int(bool(ack_failed)),
                **_heartbeat_snapshot(telem),
            )
            raise PreflightError(f"Failed to enter GUIDED within 5s (mode={telem.mode}, custom_mode={telem.custom_mode})")
        _log_preflight_event(
            events,
            "set_guided",
            "ok",
            ack_result=ack_result,
            mode_confirmed=int(bool(confirmed)),
            require_mode_known=int(require_mode_known),
            **_heartbeat_snapshot(telem),
        )

    if auto_arm:
        _log_preflight_event(events, "arm", "start")
        _cmd_arm(mav, mavutil)
        remaining = max(0.1, t_deadline - time.monotonic())
        ack_ok, ack_result = _wait_command_ack(
            mav,
            mavutil,
            telem,
            int(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM),
            timeout_s=min(10.0, remaining),
        )
        if not ack_ok:
            raise PreflightError("ARM rejected" if ack_result is not None else "ARM timeout waiting COMMAND_ACK")
        hb_t0 = float(telem.last_heartbeat_time_s)
        armed_confirmed = _wait_until_armed(mav, mavutil, telem, timeout_s=min(6.0, max(0.1, t_deadline - time.monotonic())))
        if (float(telem.last_heartbeat_time_s) > hb_t0) and (not armed_confirmed):
            _log_preflight_event(events, "arm", "fail", ack_result=ack_result, **_heartbeat_snapshot(telem))
            raise PreflightError("ARM acknowledged but heartbeat armed flag did not set")
        _log_preflight_event(events, "arm", "ok", ack_result=ack_result, armed_confirmed=int(bool(armed_confirmed)))

    if auto_takeoff:
        target_alt = float(args.auto_takeoff_m)
        tol = max(0.5, 0.1 * target_alt)
        _log_preflight_event(events, "alt_wait", "start", timeout_s=1.5)
        alt_wait_deadline = time.monotonic() + 1.5
        while time.monotonic() < alt_wait_deadline:
            alt_est, alt_src = _best_altitude_estimate_m(telem)
            if alt_est is not None:
                _log_preflight_event(
                    events,
                    "alt_wait",
                    "ok",
                    alt_m=float(alt_est),
                    alt_source=str(alt_src),
                )
                break
            _ = _recv_one_blocking_update(mav, mavutil, telem, timeout_s=0.05)
        else:
            _log_preflight_event(events, "alt_wait", "timeout")

        mode_guided_now = _guided_mode_confirmed(mav, telem)
        if not mode_guided_now:
            _log_preflight_event(events, "takeoff", "blocked", reason="not_guided", mode=str(telem.mode))
            raise PreflightError(
                f"NAV_TAKEOFF requires GUIDED mode (mode={telem.mode}, guided={int(mode_guided_now)}). Enable --auto-guided 1."
            )
        if not bool(telem.armed):
            _log_preflight_event(events, "takeoff", "blocked", reason="not_armed")
            raise PreflightError(
                "[bridge][preflight] Cannot auto takeoff while disarmed. Enable --auto-arm 1 or arm manually."
            )
        alt_est, alt_src = _best_altitude_estimate_m(telem)
        if alt_est is not None and float(alt_est) >= (target_alt - tol):
            result["takeoff_skipped"] = 1
            result["takeoff_skip_reason"] = (
                f"already_in_air alt={float(alt_est):.2f}m src={str(alt_src)} target={target_alt:.2f}m tol={tol:.2f}m"
            )
            print(f"[bridge][preflight] auto_takeoff skipped: {result['takeoff_skip_reason']}")
            _log_preflight_event(
                events,
                "takeoff",
                "skipped",
                target_alt_m=target_alt,
                tolerance_m=tol,
                rel_alt_m=float(alt_est),
                alt_source=str(alt_src),
                reason=str(result["takeoff_skip_reason"]),
            )
            _poll_mavlink_state(mav, mavutil, telem, max_msgs=120)
            return result
        if alt_est is None:
            print("[bridge][preflight] alt unavailable after grace wait; attempting NAV_TAKEOFF")
        _log_preflight_event(events, "takeoff", "start", target_alt_m=target_alt, tolerance_m=tol)
        _cmd_takeoff(mav, mavutil, target_alt)
        takeoff_cmd = int(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF)
        ack_result: int | None = None
        ack_failed = False
        timeout_s = min(max(20.0, target_alt * 8.0), max(0.1, t_deadline - time.monotonic()))
        timeout_deadline = time.monotonic() + timeout_s
        reached = False
        while time.monotonic() < timeout_deadline:
            msg = _recv_one_blocking_update(mav, mavutil, telem, timeout_s=1.0)
            if msg is not None and msg.get_type() == "COMMAND_ACK" and int(getattr(msg, "command", -1)) == takeoff_cmd:
                ack_result = int(getattr(msg, "result", -1))
                failed_res = {
                    int(mavutil.mavlink.MAV_RESULT_DENIED),
                    int(mavutil.mavlink.MAV_RESULT_UNSUPPORTED),
                    int(mavutil.mavlink.MAV_RESULT_FAILED),
                    int(mavutil.mavlink.MAV_RESULT_TEMPORARILY_REJECTED),
                }
                if ack_result in failed_res:
                    ack_failed = True
                    guided_now = _guided_mode_confirmed(mav, telem)
                    raise PreflightError(
                        "TAKEOFF ACK failed "
                        f"(result={ack_result}, mode={telem.mode}, guided={int(guided_now)}). "
                        "Ensure GUIDED mode or enable --auto-guided 1."
                    )
            alt_up_m = 0.0 if telem.relative_alt_m is None else max(0.0, float(telem.relative_alt_m))
            if alt_up_m >= (target_alt - 0.5):
                reached = True
                break
        if not reached:
            reached = 0.0 if telem.relative_alt_m is None else max(0.0, float(telem.relative_alt_m))
            _log_preflight_event(
                events,
                "takeoff",
                "fail",
                ack_result=ack_result,
                ack_failed=int(bool(ack_failed)),
                target_alt_m=target_alt,
                reached_alt_m=float(reached),
            )
            raise PreflightError(f"takeoff timeout (target={target_alt:.2f}m reached={reached:.2f}m)")
        _log_preflight_event(
            events,
            "takeoff",
            "ok",
            ack_result=ack_result,
            reached_alt_m=0.0 if telem.relative_alt_m is None else float(telem.relative_alt_m),
        )

    _poll_mavlink_state(mav, mavutil, telem, max_msgs=120)
    mode_guided = _guided_mode_confirmed(mav, telem)
    in_air_alt = 0.0 if telem.relative_alt_m is None else max(0.0, float(telem.relative_alt_m))
    in_air = bool(in_air_alt >= min_air_alt_m)
    if (not mode_guided) and (not auto_guided) and require_mode_known:
        _log_preflight_event(events, "gate_guided", "blocked", mode=str(telem.mode))
        raise PreflightError(
            f"[bridge][preflight] Blocking start: mode={telem.mode}. Set vehicle to GUIDED or enable --auto-guided 1."
        )
    if (not bool(telem.armed)) and (not auto_arm):
        _log_preflight_event(events, "gate_armed", "blocked", armed=0)
        raise PreflightError(
            "[bridge][preflight] Blocking start: vehicle is not armed. Arm in QGC or enable --auto-arm 1."
        )
    if (not in_air) and float(args.auto_takeoff_m) <= 0.0:
        _log_preflight_event(events, "gate_in_air", "blocked", alt_m=float(in_air_alt))
        raise PreflightError(
            "[bridge][preflight] Blocking start: vehicle is not in air. Take off in QGC or set --auto-takeoff-m > 0."
        )
    _log_preflight_event(
        events,
        "preflight_ready",
        "ok",
        require_mode_known=int(require_mode_known),
        **_heartbeat_snapshot(telem),
    )
    return result


def _send_local_ned_target(
    mav,
    mavutil,
    target_xyz: np.ndarray,
    telem: TelemetryState,
    use_vel_caps: bool,
    vxy_cap: float,
    vz_cap: float,
    fixed_yaw_rad: float | None = None,
    yaw_mode: str = "none",
    yaw_rate_max_dps: float = 90.0,
    yaw_dt_s: float = 0.2,
) -> tuple[float, float, float]:
    tx, ty, tz = float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])
    dx = tx - float(telem.x)
    dy = ty - float(telem.y)
    dz = tz - float(telem.z)
    dist_xy = float(math.hypot(dx, dy))
    vx = 0.0
    vy = 0.0
    vz = 0.0
    if bool(use_vel_caps):
        if dist_xy > 1e-6:
            ux = dx / dist_xy
            uy = dy / dist_xy
            vx = float(ux * max(0.0, vxy_cap))
            vy = float(uy * max(0.0, vxy_cap))
        vz = float(np.clip(dz, -abs(vz_cap), abs(vz_cap)))
        type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        )
    else:
        type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        )
    yaw_cmd = 0.0
    yaw_rate_cmd = 0.0
    yaw_mode_eff = str(yaw_mode).strip().lower()
    if yaw_mode_eff == "none":
        type_mask = int(type_mask | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE)
    else:
        max_delta = float(np.radians(max(1.0, float(yaw_rate_max_dps))) * max(1e-3, float(yaw_dt_s)))
        if yaw_mode_eff == "fixed":
            desired_yaw = float(fixed_yaw_rad) if fixed_yaw_rad is not None else float(telem.yaw)
        elif yaw_mode_eff == "face-vel":
            if float(math.hypot(vx, vy)) > 1e-6:
                desired_yaw = float(math.atan2(vy, vx))
            elif float(math.hypot(dx, dy)) > 1e-6:
                # Fallback for position-only mode (use_vel_caps=0): face target direction.
                desired_yaw = float(math.atan2(dy, dx))
            else:
                desired_yaw = float(telem.yaw)
        else:
            type_mask = int(type_mask | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE)
            desired_yaw = float(telem.yaw)
        if int(type_mask & mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE) == 0:
            yaw_err = float(math.atan2(math.sin(desired_yaw - float(telem.yaw)), math.cos(desired_yaw - float(telem.yaw))))
            yaw_cmd = float(telem.yaw + np.clip(yaw_err, -max_delta, max_delta))
    mav.mav.set_position_target_local_ned_send(
        int(time.time() * 1000) & 0xFFFFFFFF,
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        int(type_mask),
        tx,
        ty,
        tz,
        float(vx),
        float(vy),
        float(vz),
        0.0,
        0.0,
        0.0,
        float(yaw_cmd),
        float(yaw_rate_cmd),
    )
    return float(vx), float(vy), float(vz)


def _poll_mavlink_state(mav, mavutil, state: TelemetryState, max_msgs: int = 60) -> None:
    for _ in range(max(1, int(max_msgs))):
        msg = mav.recv_match(blocking=False)
        if msg is None:
            break
        _handle_mavlink_message(msg, mavutil, state)


def _health_ok(state: TelemetryState, ekf_mode: str) -> tuple[bool, str]:
    mode = str(ekf_mode).strip().lower()
    if mode == "ignore":
        return True, "ignored"
    if mode == "wait":
        if not state.gps_known or not state.ekf_known:
            return True, "waiting_unknown"
    elif mode != "strict":
        mode = "strict"
    if not state.gps_known:
        return False, "gps_unknown"
    if not state.ekf_known:
        return False, "ekf_unknown"
    if not state.gps_ok:
        return False, "gps_not_ok"
    if not state.ekf_ok:
        return False, "ekf_not_ok"
    return True, "ok"


def _return_to_spawn_and_land(
    *,
    mav,
    mavutil,
    telem: TelemetryState,
    scan_origin_x: float,
    scan_origin_y: float,
    alt_m: float,
    use_vel_caps: bool,
    vxy_cap: float,
    vz_cap: float,
    timeout_s: float,
    reach_radius_m: float,
    fixed_yaw_rad: float | None = None,
    yaw_mode: str = "none",
    yaw_rate_max_dps: float = 90.0,
    yaw_dt_s: float = 0.2,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "attempted": 1,
        "reached_spawn": 0,
        "land_mode_set": 0,
        "return_duration_s": 0.0,
        "final_dist_to_spawn_m": float("nan"),
        "reason": "",
    }
    t0 = time.monotonic()
    timeout_s_eff = max(1.0, float(timeout_s))
    reach_radius_eff = max(0.2, float(reach_radius_m))
    target_xyz_abs = np.asarray(
        [float(scan_origin_x), float(scan_origin_y), -abs(float(alt_m))],
        dtype=np.float32,
    )
    while (time.monotonic() - t0) < timeout_s_eff:
        _poll_mavlink_state(mav, mavutil, telem, max_msgs=80)
        dx = float(scan_origin_x) - float(telem.x)
        dy = float(scan_origin_y) - float(telem.y)
        dist = float(math.hypot(dx, dy))
        out["final_dist_to_spawn_m"] = float(dist)
        if dist <= reach_radius_eff:
            out["reached_spawn"] = 1
            out["reason"] = "spawn_reached"
            break
        _send_local_ned_target(
            mav,
            mavutil,
            target_xyz_abs,
            telem,
            use_vel_caps=bool(use_vel_caps),
            vxy_cap=float(vxy_cap),
            vz_cap=float(vz_cap),
            fixed_yaw_rad=fixed_yaw_rad,
            yaw_mode=str(yaw_mode),
            yaw_rate_max_dps=float(yaw_rate_max_dps),
            yaw_dt_s=float(yaw_dt_s),
        )
        time.sleep(0.1)
    if int(out["reached_spawn"]) != 1:
        out["reason"] = "spawn_return_timeout"

    out["land_mode_set"] = int(_set_mode(mav, mavutil, "LAND"))
    if int(out["land_mode_set"]) != 1 and not out["reason"]:
        out["reason"] = "land_mode_set_failed"
    if int(out["land_mode_set"]) == 1 and not out["reason"]:
        out["reason"] = "land_mode_set_ok"
    out["return_duration_s"] = float(max(0.0, time.monotonic() - t0))
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ArduPilot LOCAL_NED scan bridge with scale-aware profile selection.")
    parser.add_argument("--dry-run", type=int, default=1, help="1: no MAVLink commands, 0: send MAVLink commands")
    parser.add_argument("--sitl-recommended", type=int, default=1, choices=(0, 1))
    parser.add_argument("--sitl-recommended-source", type=str, default="latest")
    parser.add_argument("--model", type=str, default="auto", help="Model path or auto for scan profile selector")
    parser.add_argument("--task", type=str, default="scan")
    parser.add_argument("--preset", type=str, default="A2", choices=list_presets())
    parser.add_argument("--efficiency-profile", type=str, default="none", choices=("none", "best_efficiency"))
    parser.add_argument("--connection", type=str, default="udp:127.0.0.1:14550")
    parser.add_argument("--anchor-origin", type=int, default=1, choices=(0, 1))
    parser.add_argument(
        "--scan-origin-mode",
        type=str,
        default="drone",
        choices=("drone", "fixed"),
        help="drone: tie scan-area reference to vehicle position (if --anchor-origin=1), fixed: use --scan-origin-x/y.",
    )
    parser.add_argument(
        "--scan-origin-ref",
        type=str,
        default="center",
        choices=("center", "sw", "se", "nw", "ne"),
        help="Which point is represented by scan origin input/anchor (center or scanable-area corner).",
    )
    parser.add_argument(
        "--scan-origin-x",
        type=float,
        default=0.0,
        help="LOCAL_NED X (meters) of scan origin reference point when --scan-origin-mode=fixed.",
    )
    parser.add_argument(
        "--scan-origin-y",
        type=float,
        default=0.0,
        help="LOCAL_NED Y (meters) of scan origin reference point when --scan-origin-mode=fixed.",
    )
    parser.add_argument("--prefer-sitl-tcp", type=int, default=1, choices=(0, 1))
    parser.add_argument("--preflight-timeout-s", type=float, default=30.0)
    parser.add_argument("--require-mode-known", type=int, default=-1)
    parser.add_argument("--ekf-mode", type=str, default="auto", choices=("auto", "strict", "wait", "ignore"))
    parser.add_argument("--ack-verbosity", type=str, default="important", choices=("all", "important"))
    parser.add_argument("--auto-guided", type=int, default=0, choices=(0, 1))
    parser.add_argument("--auto-arm", type=int, default=0, choices=(0, 1))
    parser.add_argument("--auto-takeoff-m", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=0, help="Optional hard step cap (0 disables this cap)")
    parser.add_argument("--scan-path-len-scale", type=float, default=1.0)
    parser.add_argument("--bounds-m", type=float, nargs=2, default=(40.0, 40.0), metavar=("W", "H"))
    parser.add_argument("--margin-m", type=float, default=2.0)
    parser.add_argument("--camera-hfov-deg", type=float, default=70.0)
    parser.add_argument("--camera-vfov-deg", type=float, default=50.0)
    parser.add_argument("--footprint-model", type=str, default="ellipse", choices=("ellipse", "circle_min", "circle_area"))
    parser.add_argument("--footprint-min-passes", type=int, default=1)
    parser.add_argument("--footprint-fov-scale", type=float, default=1.0)
    parser.add_argument("--footprint-count-dist-factor", type=float, default=0.3)
    parser.add_argument("--footprint-count-dist-min-m", type=float, default=0.2)
    parser.add_argument("--rate-hz", type=float, default=5.0)
    parser.add_argument("--policy-hz", type=float, default=2.0)
    parser.add_argument("--coverage-hz", type=float, default=5.0)
    parser.add_argument("--lookahead-enable", type=int, default=1, choices=(0, 1))
    parser.add_argument("--lookahead-time-s", type=float, default=0.6)
    parser.add_argument("--lookahead-cap-m", type=float, default=2.0)
    parser.add_argument("--lane-keep-enable", type=int, default=0, choices=(0, 1))
    parser.add_argument("--lane-dir-mode", type=str, default="auto", choices=("auto", "fixed"))
    parser.add_argument("--lane-kp", type=float, default=0.4)
    parser.add_argument("--lane-max-corr-m", type=float, default=1.5)
    parser.add_argument("--step-sched-enable", type=int, default=1, choices=(0, 1))
    parser.add_argument("--step-len-corner-mult", type=float, default=0.7)
    parser.add_argument("--step-len-boundary-mult", type=float, default=0.8)
    parser.add_argument("--boundary-near-m", type=float, default=3.0)
    parser.add_argument("--step-len-m", type=float, default=None)
    parser.add_argument("--accept-radius-m", type=float, default=None)
    parser.add_argument("--adaptive-tracking", type=int, default=0, choices=(0, 1))
    parser.add_argument("--adapt-interval-s", type=float, default=5.0)
    parser.add_argument("--dist-p95-high", type=float, default=2.5)
    parser.add_argument("--progress-low", type=float, default=0.2)
    parser.add_argument("--policy-hz-min", type=float, default=1.0)
    parser.add_argument("--policy-hz-max", type=float, default=3.0)
    parser.add_argument("--policy-hz-stable-dist-p95", type=float, default=1.5)
    parser.add_argument("--policy-hz-stable-progress", type=float, default=0.4)
    parser.add_argument("--step-len-min", type=float, default=2.0)
    parser.add_argument("--step-len-max", type=float, default=8.0)
    parser.add_argument("--accept-radius-min", type=float, default=0.75)
    parser.add_argument("--accept-radius-max", type=float, default=2.0)
    parser.add_argument("--max-hold-s", type=float, default=3.0)
    parser.add_argument("--alt-m", type=float, default=10.0)
    parser.add_argument("--stop-cov", type=float, default=0.95)
    parser.add_argument("--ignore-ekf", type=int, default=0)
    parser.add_argument("--clamp-stop-count", type=int, default=100)
    parser.add_argument("--oob-recovery", type=int, default=1, choices=(0, 1))
    parser.add_argument("--oob-recovery-seconds", type=float, default=3.0)
    parser.add_argument("--oob-recovery-vxy-cap", type=float, default=0.6)
    parser.add_argument("--oob-recovery-accept-boost", type=float, default=0.5)
    parser.add_argument("--oob-clamp-fast-count", type=int, default=3)
    parser.add_argument("--oob-clamp-fast-window-s", type=float, default=5.0)
    parser.add_argument("--target-refresh-mode", type=str, default="hold", choices=("hold", "always"))
    parser.add_argument("--strict-lawnmower", type=int, default=0, choices=(0, 1))
    parser.add_argument("--strict-vmin", type=float, default=0.4)
    parser.add_argument("--strict-slowdown-mult", type=float, default=3.0)
    parser.add_argument("--yaw-mode", type=str, default="none", choices=("fixed", "face-vel", "none"))
    parser.add_argument("--yaw-rate-max-dps", type=float, default=90.0)
    parser.add_argument("--return-land-on-complete", type=int, default=0, choices=(0, 1))
    parser.add_argument("--return-land-timeout-s", type=float, default=45.0)
    parser.add_argument("--return-land-radius-m", type=float, default=1.0)
    parser.add_argument("--use-vel-caps", type=int, default=1, choices=(0, 1))
    parser.add_argument("--vxy-cap", type=float, default=None)
    parser.add_argument("--vz-cap", type=float, default=0.5)
    parser.add_argument("--corner-angle-deg", type=float, default=60.0)
    parser.add_argument("--corner-slow-seconds", type=float, default=2.0)
    parser.add_argument("--corner-vxy-cap", type=float, default=0.8)
    parser.add_argument(
        "--turn-style",
        type=str,
        default="sharp",
        choices=("sharp", "arc"),
        help="Turn style for corners: 'sharp' for 90-degree, 'arc' for smooth rounded turns."
    )
    parser.add_argument(
        "--turn-radius-m",
        type=float,
        default=2.5,
        help="Turn radius in meters for arc turns (used only if --turn-style=arc)."
    )
    parser.add_argument(
        "--cfg-override",
        action="append",
        default=[],
        help="Override preset fields with key=value (repeatable).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw_argv = list(sys.argv[1:])
    require_mode_known_explicit = _flag_was_set(raw_argv, "--require-mode-known")
    ekf_mode_explicit = _flag_was_set(raw_argv, "--ekf-mode")
    ignore_ekf_explicit = _flag_was_set(raw_argv, "--ignore-ekf")
    explicit_flags = {
        "lookahead_enable": _flag_was_set(raw_argv, "--lookahead-enable"),
        "step_sched_enable": _flag_was_set(raw_argv, "--step-sched-enable"),
        "lane_keep_enable": _flag_was_set(raw_argv, "--lane-keep-enable"),
        "adaptive_tracking": _flag_was_set(raw_argv, "--adaptive-tracking"),
        "lane_kp": _flag_was_set(raw_argv, "--lane-kp"),
        "corner_vxy_cap": _flag_was_set(raw_argv, "--corner-vxy-cap"),
        "yaw_mode": _flag_was_set(raw_argv, "--yaw-mode"),
        "strict_lawnmower": _flag_was_set(raw_argv, "--strict-lawnmower"),
        "return_land_on_complete": _flag_was_set(raw_argv, "--return-land-on-complete"),
        "use_vel_caps": _flag_was_set(raw_argv, "--use-vel-caps"),
        "step_len_m": _flag_was_set(raw_argv, "--step-len-m"),
        "accept_radius_m": _flag_was_set(raw_argv, "--accept-radius-m"),
        "policy_hz": _flag_was_set(raw_argv, "--policy-hz"),
        "coverage_hz": _flag_was_set(raw_argv, "--coverage-hz"),
        "camera_hfov_deg": _flag_was_set(raw_argv, "--camera-hfov-deg"),
        "camera_vfov_deg": _flag_was_set(raw_argv, "--camera-vfov-deg"),
        "footprint_model": _flag_was_set(raw_argv, "--footprint-model"),
        "footprint_fov_scale": _flag_was_set(raw_argv, "--footprint-fov-scale"),
        "footprint_min_passes": _flag_was_set(raw_argv, "--footprint-min-passes"),
        "footprint_count_dist_factor": _flag_was_set(raw_argv, "--footprint-count-dist-factor"),
        "footprint_count_dist_min_m": _flag_was_set(raw_argv, "--footprint-count-dist-min-m"),
    }
    efficiency_profile_name = str(args.efficiency_profile).strip().lower()
    efficiency_profile_applied: list[str] = []
    if efficiency_profile_name == "best_efficiency":
        profile_cfg: dict[str, Any] = {
            "strict_lawnmower": 1,
            "return_land_on_complete": 1,
            "yaw_mode": "face-vel",
            "yaw_rate_max_dps": 20.0,
            "use_vel_caps": 1,
            "vxy_cap": 1.1,
            "step_len_m": 8.0,
            "accept_radius_m": 1.6,
            "policy_hz": 1.5,
            "coverage_hz": 1.0,
            "strict_vmin": 0.7,
            "strict_slowdown_mult": 3.5,
            "turn_style": "sharp",
            "camera_hfov_deg": 151.5,
            "camera_vfov_deg": 131.3,
            "footprint_model": "circle_min",
            "footprint_fov_scale": 0.35,
            "footprint_min_passes": 1,
            "footprint_count_dist_factor": 0.7,
            "footprint_count_dist_min_m": 0.5,
        }
        for k, v in profile_cfg.items():
            if bool(explicit_flags.get(k, False)):
                continue
            setattr(args, k, v)
            efficiency_profile_applied.append(str(k))
        print(
            f"[bridge] efficiency_profile=best_efficiency "
            f"applied={','.join(efficiency_profile_applied) if efficiency_profile_applied else 'none'}"
        )
    cfg = get_preset(args.preset)
    cfg_overrides = parse_override_pairs(args.cfg_override)
    if cfg_overrides:
        apply_overrides(cfg, cfg_overrides)
    if args.task.strip().lower() == "scan":
        setattr(cfg, "scan_path_len_scale", float(args.scan_path_len_scale))

    task_key = args.task.strip().lower()
    model_arg = (args.model or "").strip()
    auto_tokens = {"", "auto", "production", "production_scan"}
    selected_profile = "manual"
    if task_key == "scan" and model_arg.lower() in auto_tokens:
        model_path_obj, profile = resolve_scan_production_model_path(cfg)
        apply_scan_obs_profile(cfg, profile)
        assert_scan_obs_profile(cfg, profile, ctx="ardupilot_bridge:auto")
        model_path = str(model_path_obj)
        selected_profile = str(profile.name)
    elif model_arg:
        model_path = normalize_model_path(model_arg)
    else:
        model_path = ""

    scale_upper = float(get_scan_path_scale_upper(cfg))
    sitl_src_raw = str(args.sitl_recommended_source or "latest").strip()
    sitl_enabled = bool(int(args.sitl_recommended) == 1 and sitl_src_raw.lower() != "none")
    sitl_recommended_source_used = "disabled" if not sitl_enabled else sitl_src_raw
    sitl_recommended_arm = "A_baseline"
    sitl_recommended_profile_match = False
    if sitl_enabled:
        sitl_recommended_arm, winner_cfg, sitl_recommended_source_used, sitl_recommended_profile_match = (
            _resolve_sitl_recommended_profile(sitl_src_raw, scale_upper)
        )
        applied: list[str] = []
        for key in ("lookahead_enable", "step_sched_enable", "lane_keep_enable", "adaptive_tracking"):
            if bool(explicit_flags.get(key, False)):
                continue
            if key in winner_cfg:
                setattr(args, key, int(winner_cfg[key]))
                applied.append(key)
        for key in ("lane_kp", "corner_vxy_cap"):
            if bool(explicit_flags.get(key, False)):
                continue
            if key in winner_cfg:
                setattr(args, key, float(winner_cfg[key]))
                applied.append(key)
        print(
            f"[bridge] SITL recommended applied: {sitl_recommended_arm} "
            f"(source={sitl_recommended_source_used} profile_match={int(sitl_recommended_profile_match)} "
            f"updated={','.join(applied) if applied else 'none'})"
        )
    elif int(args.sitl_recommended) == 1 and sitl_src_raw.lower() == "none":
        print("[bridge] SITL recommended disabled by source=none")

    if int(args.require_mode_known) not in (-1, 0, 1):
        raise SystemExit("[bridge] --require-mode-known must be one of {-1,0,1}")
    if (not require_mode_known_explicit) and int(args.require_mode_known) == -1:
        args.require_mode_known = 0 if sitl_enabled else 1
    elif int(args.require_mode_known) == -1:
        args.require_mode_known = 1

    if ekf_mode_explicit:
        args.ekf_mode = str(args.ekf_mode).strip().lower()
    elif ignore_ekf_explicit:
        args.ekf_mode = "ignore" if int(args.ignore_ekf) == 1 else "strict"
    elif sitl_enabled:
        args.ekf_mode = "wait"
    else:
        args.ekf_mode = "ignore" if int(args.ignore_ekf) == 1 else "strict"
    print(
        f"[bridge] preflight defaults: require_mode_known={int(args.require_mode_known)} ekf_mode={args.ekf_mode}"
    )

    scan_max_steps_eff = int(effective_scan_max_steps(cfg)) if task_key == "scan" else int(getattr(cfg, "max_steps", 300))
    step_len = float(args.step_len_m) if args.step_len_m is not None else _auto_step_len(scale_upper)
    accept_radius = float(args.accept_radius_m) if args.accept_radius_m is not None else _auto_accept_radius(scale_upper)
    base_vxy_cap = float(args.vxy_cap) if args.vxy_cap is not None else _auto_vxy_cap(scale_upper)
    initial_step_len = float(step_len)
    initial_accept_radius = float(accept_radius)
    initial_vxy_cap = float(base_vxy_cap)
    use_vel_caps = bool(int(args.use_vel_caps))
    vz_cap = abs(float(args.vz_cap))
    adaptive_enabled = bool(int(args.adaptive_tracking))
    adapt_interval_s = max(0.1, float(args.adapt_interval_s))
    step_len_min = min(float(args.step_len_min), float(args.step_len_max))
    step_len_max = max(float(args.step_len_min), float(args.step_len_max))
    accept_radius_min = min(float(args.accept_radius_min), float(args.accept_radius_max))
    accept_radius_max = max(float(args.accept_radius_min), float(args.accept_radius_max))
    vxy_cap_tuned = float(base_vxy_cap)
    oob_recovery_enabled = bool(int(args.oob_recovery))
    oob_recovery_seconds = max(0.5, float(args.oob_recovery_seconds))
    oob_recovery_vxy_cap = max(0.1, float(args.oob_recovery_vxy_cap))
    oob_recovery_accept_boost = max(0.0, float(args.oob_recovery_accept_boost))
    oob_clamp_fast_count = max(1, int(args.oob_clamp_fast_count))
    oob_clamp_fast_window_s = max(0.5, float(args.oob_clamp_fast_window_s))
    policy_hz = max(1e-3, float(args.policy_hz))
    coverage_hz = max(1e-3, float(args.coverage_hz))
    lookahead_enabled = bool(int(args.lookahead_enable))
    lookahead_time_s = max(0.0, float(args.lookahead_time_s))
    lookahead_cap_m = max(0.0, float(args.lookahead_cap_m))
    lane_keep_enabled = bool(int(args.lane_keep_enable))
    lane_dir_mode = str(args.lane_dir_mode).strip().lower()
    lane_kp = max(0.0, float(args.lane_kp))
    lane_max_corr_m = max(0.0, float(args.lane_max_corr_m))
    step_sched_enabled = bool(int(args.step_sched_enable))
    step_len_corner_mult = float(np.clip(float(args.step_len_corner_mult), 0.1, 2.0))
    step_len_boundary_mult = float(np.clip(float(args.step_len_boundary_mult), 0.1, 2.0))
    boundary_near_m = max(0.0, float(args.boundary_near_m))
    policy_hz_min = min(float(args.policy_hz_min), float(args.policy_hz_max))
    policy_hz_max = max(float(args.policy_hz_min), float(args.policy_hz_max))
    policy_hz_stable_dist_p95 = max(0.0, float(args.policy_hz_stable_dist_p95))
    policy_hz_stable_progress = float(args.policy_hz_stable_progress)
    policy_hz_current = float(np.clip(policy_hz, policy_hz_min, policy_hz_max))
    strict_lawnmower = bool(int(args.strict_lawnmower) == 1)
    strict_hold_steps = 1
    strict_accept_ticks = 0
    strict_fixed_yaw_rad = 0.0
    strict_fixed_yaw_deg = 0.0
    strict_vmin = max(0.1, float(args.strict_vmin))
    strict_slowdown_mult = max(0.5, float(args.strict_slowdown_mult))
    if strict_lawnmower:
        lookahead_enabled = False
        adaptive_enabled = False
        step_sched_enabled = False
        oob_recovery_enabled = False
        lane_keep_enabled = False
        if args.accept_radius_m is None:
            accept_radius = 1.5
        policy_hz_min = float(policy_hz)
        policy_hz_max = float(policy_hz)
        policy_hz_current = float(policy_hz)
    yaw_mode_effective = str(args.yaw_mode).strip().lower()
    if (not bool(explicit_flags.get("yaw_mode", False))) and strict_lawnmower and yaw_mode_effective == "none":
        # Preserve previous strict behavior only when yaw mode is otherwise unset.
        yaw_mode_effective = "fixed"
    bounds_w = float(args.bounds_m[0])
    bounds_h = float(args.bounds_m[1])
    half_w = 0.5 * bounds_w
    half_h = 0.5 * bounds_h
    anchor_origin_enabled = bool(int(args.anchor_origin) == 1)
    scan_origin_mode = str(args.scan_origin_mode).strip().lower()
    scan_origin_ref = str(args.scan_origin_ref).strip().lower()
    center_off_x, center_off_y = scan_center_offset_for_reference(
        scan_origin_ref,
        bounds_w,
        bounds_h,
        float(args.margin_m),
    )
    scan_origin_x = float(args.scan_origin_x) + float(center_off_x)
    scan_origin_y = float(args.scan_origin_y) + float(center_off_y)
    x_min_rel, x_max_rel, y_min_rel, y_max_rel = bounds_minmax_xy(
        0.0,
        0.0,
        bounds_w,
        bounds_h,
        float(args.margin_m),
    )
    x_min_scan, x_max_scan, y_min_scan, y_max_scan = bounds_minmax_xy(
        scan_origin_x,
        scan_origin_y,
        bounds_w,
        bounds_h,
        float(args.margin_m),
    )

    print(
        f"[bridge] profile={selected_profile} model={model_path or '<none>'} "
        f"path_scale_upper={scale_upper:.3f} scan_max_steps_eff={scan_max_steps_eff}"
    )
    connection_requested = str(args.connection)
    connection_used = _maybe_switch_to_sitl_master(
        connection=connection_requested,
        prefer_sitl_tcp=int(args.prefer_sitl_tcp),
        probe_timeout_s=1.0,
    )

    run_root = Path("runs") / "ardupilot_scan"
    run_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = _next_free_dir(run_root / f"{stamp}_{_safe_slug(selected_profile)}")
    run_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = run_dir / "telemetry.jsonl"
    summary_path = run_dir / "summary.json"

    cov_cell = float(getattr(cfg, "scan_cov_cell_size", 1.0) or 1.0)
    cov_radius = float(getattr(cfg, "scan_coverage_radius", 0.5) or 0.5)
    cov = CoverageGridTracker(bounds_w, bounds_h, cov_cell, cov_radius, boundary_cells=2)
    cov_footprint = CoverageGridTracker(bounds_w, bounds_h, cov_cell, cov_radius, boundary_cells=2, track_pass_counts=True)
    hfov_half_rad = float(np.radians(max(1e-3, float(args.camera_hfov_deg)) * 0.5))
    vfov_half_rad = float(np.radians(max(1e-3, float(args.camera_vfov_deg)) * 0.5))
    footprint_model = str(args.footprint_model).strip().lower()
    footprint_min_passes = max(1, int(args.footprint_min_passes))
    footprint_fov_scale = max(1e-6, float(args.footprint_fov_scale))
    footprint_count_dist_factor = max(0.0, float(args.footprint_count_dist_factor))
    footprint_count_dist_min_m = max(0.0, float(args.footprint_count_dist_min_m))
    patch_size = int(getattr(cfg, "scan_obs_patch_size", 5))
    boundary_feat = bool(getattr(cfg, "scan_obs_boundary_feat", True))

    dense_step_len = float(step_len)
    if adaptive_enabled:
        dense_step_len = float(min(step_len, step_len_min))
    targets = _build_lawnmower_targets(
        bounds_w,
        bounds_h,
        float(args.margin_m),
        dense_step_len,
        float(args.alt_m),
        turn_style=str(args.turn_style),
        turn_radius_m=float(args.turn_radius_m),
    )
    target_idx = 0
    target_stride = max(1, int(round(step_len / max(1e-3, dense_step_len))))
    t0 = time.monotonic()
    # Pre-anchor placeholders; target state is reset after preflight/origin anchor.
    target_xyz_rel = np.asarray([0.0, 0.0, -abs(float(args.alt_m))], dtype=np.float32)
    target_xyz_abs = target_xyz_rel.copy()
    clamp_count = 0
    target_set_t = t0
    best_dist_since_set = float("inf")
    last_improve_t = t0
    target_was_clamped = False
    num_target_updates = 0
    num_corner_events = 0
    corner_until_t = 0.0
    step_count = 0
    num_policy_ticks = 0
    policy_action = np.zeros(2, dtype=np.float32)
    chosen_target_xy = np.asarray([float(target_xyz_abs[0]), float(target_xyz_abs[1])], dtype=np.float32)
    lane_dir = np.zeros(2, dtype=np.float32)
    lane_origin = np.asarray([0.0, 0.0], dtype=np.float32)
    lane_has_ref = False
    num_lane_resets = 0
    cross_track_abs_hist: list[float] = []
    lookahead_delta_hist: list[float] = []
    step_len_eff_hist: list[float] = []
    oob_event_count = 0
    oob_recovery_active = False
    oob_recovery_until_t = 0.0
    oob_recovery_enter_t = 0.0
    oob_recovery_time_sum_s = 0.0
    clamp_times = deque()

    telem = TelemetryState(
        z=-abs(float(args.alt_m)),
        mode="GUIDED_DRY" if int(args.dry_run) == 1 else "UNKNOWN",
        ack_verbosity=str(args.ack_verbosity),
    )
    mav = None
    mavutil = None
    preflight_events: list[dict[str, Any]] = []
    preflight_result: dict[str, Any] = {"status": "ok", "reason": ""}
    takeoff_skipped = 0
    takeoff_skip_reason = ""
    message_intervals_sent = False
    requested_message_intervals: list[str] = []
    if int(args.dry_run) == 0:
        mav, mavutil = _connect_mavlink(connection_used)
        try:
            hb_msg = mav.wait_heartbeat(timeout=5)
        except Exception as exc:
            preflight_result = {"status": "fail", "reason": f"timeout waiting heartbeat after connect: {exc}"}
            print(f"[bridge][preflight] {preflight_result['reason']}")
            hb_msg = None
        if hb_msg is not None:
            _handle_mavlink_message(hb_msg, mavutil, telem)
            print(
                "[bridge][debug] heartbeat after connect: "
                f"base_mode={int(getattr(hb_msg, 'base_mode', 0))} "
                f"custom_mode={int(getattr(hb_msg, 'custom_mode', 0))} "
                f"system_status={int(getattr(hb_msg, 'system_status', 0))}"
            )
        message_intervals_sent, requested_message_intervals = _request_message_intervals_once(
            mav,
            mavutil,
            rate_hz=float(coverage_hz),
            already_sent=bool(message_intervals_sent),
        )
        _poll_mavlink_state(mav, mavutil, telem, max_msgs=120)
        print(
            f"[bridge] MAVLink connected: {connection_used} mode={telem.mode} "
            f"armed={int(bool(telem.armed))} alt={max(0.0, -float(telem.z)):.2f}m"
        )
        if str(preflight_result.get("status", "")) == "ok":
            try:
                pf_result = _preflight(args=args, mav=mav, mavutil=mavutil, telem=telem, events=preflight_events)
                takeoff_skipped = int(pf_result.get("takeoff_skipped", 0))
                takeoff_skip_reason = str(pf_result.get("takeoff_skip_reason", ""))
                print(
                    f"[bridge] preflight complete mode={telem.mode} "
                    f"armed={int(bool(telem.armed))} alt={max(0.0, -float(telem.z)):.2f}m"
                )
            except PreflightError as exc:
                preflight_result = {"status": "fail", "reason": str(exc)}
                print(f"[bridge][preflight] failed: {exc}")
    else:
        print("[bridge] Dry-run mode enabled. No MAVLink commands are sent.")
        try:
            pf_result = _preflight(args=args, mav=mav, mavutil=mavutil, telem=telem, events=preflight_events)
            takeoff_skipped = int(pf_result.get("takeoff_skipped", 0))
            takeoff_skip_reason = str(pf_result.get("takeoff_skip_reason", ""))
        except PreflightError as exc:
            preflight_result = {"status": "fail", "reason": str(exc)}
            print(f"[bridge][preflight] failed: {exc}")

    if str(preflight_result.get("status", "")) == "ok":
        if scan_origin_mode == "drone" and anchor_origin_enabled:
            if int(args.dry_run) == 0:
                _poll_mavlink_state(mav, mavutil, telem, max_msgs=120)
            anchor_ref_x = float(telem.x)
            anchor_ref_y = float(telem.y)
            scan_origin_x = float(anchor_ref_x + center_off_x)
            scan_origin_y = float(anchor_ref_y + center_off_y)
            print(
                f"[bridge] anchored scan origin ref={scan_origin_ref} "
                f"at vehicle=({anchor_ref_x:.2f}, {anchor_ref_y:.2f}) -> center=({scan_origin_x:.2f}, {scan_origin_y:.2f})"
            )
        elif scan_origin_mode == "fixed":
            print(
                f"[bridge] fixed scan origin ref={scan_origin_ref} "
                f"input=({float(args.scan_origin_x):.2f}, {float(args.scan_origin_y):.2f}) "
                f"-> center=({scan_origin_x:.2f}, {scan_origin_y:.2f})"
            )
    x_min_scan, x_max_scan, y_min_scan, y_max_scan = bounds_minmax_xy(
        scan_origin_x,
        scan_origin_y,
        bounds_w,
        bounds_h,
        float(args.margin_m),
    )
    if str(preflight_result.get("status", "")) == "ok":
        # Reset planner/target state after anchor to avoid first-sample distance spikes.
        target_idx = 0
        target_xyz_rel = np.asarray([0.0, 0.0, -abs(float(args.alt_m))], dtype=np.float32)
        target_xyz_abs = target_xyz_rel.copy()
        target_xyz_abs[0] = float(scan_origin_x) + float(target_xyz_rel[0])
        target_xyz_abs[1] = float(scan_origin_y) + float(target_xyz_rel[1])
        print(
            f"[bridge] target state reset after anchor: tgt_abs=({float(target_xyz_abs[0]):.2f},{float(target_xyz_abs[1]):.2f}), "
            "tgt_rel=(0,0)"
        )
        chosen_target_xy = np.asarray([float(target_xyz_abs[0]), float(target_xyz_abs[1])], dtype=np.float32)
        target_set_t = t0
        best_dist_since_set = float("inf")
        last_improve_t = t0
        target_was_clamped = False
        clamp_count = 0
        clamp_times = deque()
        if strict_lawnmower:
            strict_fixed_yaw_rad = float(telem.yaw)
            strict_fixed_yaw_deg = float(np.degrees(strict_fixed_yaw_rad))
            strict_slowdown_dist = float(strict_slowdown_mult * max(1e-6, float(accept_radius)))
            print(
                f"[bridge] strict_lawnmower=1 vxy_cap={float(base_vxy_cap):.2f} "
                f"accept_radius={float(accept_radius):.2f} hold_steps={int(strict_hold_steps)} "
                f"fixed_yaw_deg={float(strict_fixed_yaw_deg):.1f} "
                f"vmin={float(strict_vmin):.2f} slowdown_dist={float(strict_slowdown_dist):.2f}"
            )
        if int(args.dry_run) == 0:
            _send_local_ned_target(
                mav,
                mavutil,
                target_xyz_abs,
                telem,
                use_vel_caps=bool(use_vel_caps),
                vxy_cap=float(base_vxy_cap),
                vz_cap=float(vz_cap),
                fixed_yaw_rad=(float(strict_fixed_yaw_rad) if strict_lawnmower else None),
                yaw_mode=str(yaw_mode_effective),
                yaw_rate_max_dps=float(args.yaw_rate_max_dps),
                yaw_dt_s=float(1.0 / max(1e-3, float(args.rate_hz))),
            )

    dist_hist: list[float] = []
    coverage_hist: list[float] = []
    coverage_footprint_hist: list[float] = []
    coverage_footprint_kx_hist: list[float] = []
    alt_used_hist: list[float] = []
    progress_hist: list[float] = []
    prev_dist = None
    time_to_cov95_s: float | None = None
    adapt_window = deque()
    adapt_next_t = t0 + adapt_interval_s
    adapt_event_count = 0
    adapt_bad_count = 0
    adapt_good_count = 0
    adapt_last_p95 = 0.0
    adapt_last_progress = 0.0
    exit_reason = "unknown"
    rc = 0
    return_land_result: dict[str, Any] = {
        "attempted": 0,
        "reached_spawn": 0,
        "land_mode_set": 0,
        "return_duration_s": 0.0,
        "final_dist_to_spawn_m": float("nan"),
        "reason": "",
    }
    log_next_t = t0
    dt = 1.0 / max(1e-3, float(args.rate_hz))
    policy_dt = 1.0 / max(1e-3, float(policy_hz_current))
    coverage_dt = 1.0 / max(1e-3, float(coverage_hz))
    next_policy_t = t0
    next_coverage_t = t0
    coverage = 0.0
    coverage_footprint = 0.0
    coverage_footprint_kx = 0.0
    footprint_count_updates = 0
    footprint_count_skips = 0
    last_footprint_count_xy_rel: np.ndarray | None = None
    frame_mismatch_warned = False
    frame_mismatch_any = 0
    deadline_s = float(scan_max_steps_eff) / max(1e-3, float(args.rate_hz))

    if str(preflight_result.get("status", "")) != "ok":
        exit_reason = f"preflight_fail:{preflight_result.get('reason', 'unknown')}"
        rc = 3

    with telemetry_path.open("w", encoding="utf-8") as telem_f:
        while str(preflight_result.get("status", "")) == "ok":
            loop_t = time.monotonic()
            elapsed = float(loop_t - t0)
            if int(args.steps) > 0 and step_count >= int(args.steps):
                exit_reason = "step_limit"
                break
            if elapsed >= deadline_s:
                exit_reason = "time_budget"
                break
            if clamp_count > int(args.clamp_stop_count):
                print(f"[bridge] WARNING: clamp_count={clamp_count} exceeded {int(args.clamp_stop_count)}")
                exit_reason = "clamp_limit"
                break

            if int(args.dry_run) == 0:
                _poll_mavlink_state(mav, mavutil, telem, max_msgs=80)
                ok_health, health_reason = _health_ok(telem, str(args.ekf_mode))
                if not ok_health:
                    _set_mode(mav, mavutil, "LAND")
                    exit_reason = f"ekf_gps_fail:{health_reason}"
                    rc = 2
                    break
            else:
                telem.mode = "GUIDED_DRY"
                telem.gps_known = True
                telem.ekf_known = True
                telem.gps_ok = True
                telem.ekf_ok = True

            policy_tick_flag = 0
            if loop_t >= next_policy_t:
                policy_tick_flag = 1
                num_policy_ticks += 1
                next_policy_t = loop_t + policy_dt

            if loop_t >= next_coverage_t:
                cov.update(float(telem.x) - float(scan_origin_x), float(telem.y) - float(scan_origin_y))
                alt_up_m, _ = _best_altitude_estimate_m(telem)
                alt_used_m = float(max(0.0, alt_up_m if alt_up_m is not None else float(args.alt_m)))
                rx = float(max(0.0, alt_used_m * math.tan(hfov_half_rad) * footprint_fov_scale))
                ry = float(max(0.0, alt_used_m * math.tan(vfov_half_rad) * footprint_fov_scale))
                if footprint_model == "circle_min":
                    footprint_r_eff = float(min(rx, ry))
                elif footprint_model == "circle_area":
                    footprint_r_eff = float(math.sqrt(max(0.0, rx * ry)))
                else:
                    footprint_r_eff = float(math.sqrt(max(0.0, rx * ry)))
                footprint_count_dist_thresh = max(
                    float(footprint_count_dist_min_m),
                    float(footprint_count_dist_factor * max(0.0, footprint_r_eff)),
                )
                cur_xy_rel = np.asarray(
                    [float(telem.x) - float(scan_origin_x), float(telem.y) - float(scan_origin_y)],
                    dtype=np.float32,
                )
                should_count_footprint = True
                if last_footprint_count_xy_rel is not None:
                    moved_m = float(np.linalg.norm(cur_xy_rel - last_footprint_count_xy_rel))
                    should_count_footprint = bool(moved_m >= footprint_count_dist_thresh)
                if footprint_model == "circle_min":
                    if should_count_footprint:
                        cov_footprint.update_disk_radius(
                            float(cur_xy_rel[0]),
                            float(cur_xy_rel[1]),
                            float(min(rx, ry)),
                            count_hits=True,
                        )
                elif footprint_model == "circle_area":
                    if should_count_footprint:
                        cov_footprint.update_disk_radius(
                            float(cur_xy_rel[0]),
                            float(cur_xy_rel[1]),
                            float(math.sqrt(max(0.0, rx * ry))),
                            count_hits=True,
                        )
                else:
                    if should_count_footprint:
                        cov_footprint.update_ellipse(
                            float(cur_xy_rel[0]),
                            float(cur_xy_rel[1]),
                            float(rx),
                            float(ry),
                            count_hits=True,
                        )
                if should_count_footprint:
                    footprint_count_updates += 1
                    last_footprint_count_xy_rel = cur_xy_rel
                else:
                    footprint_count_skips += 1
                coverage = cov.coverage_frac()
                coverage_footprint = cov_footprint.coverage_frac_at_least(1)
                coverage_footprint_kx = cov_footprint.coverage_frac_at_least(footprint_min_passes)
                next_coverage_t = loop_t + coverage_dt
            coverage_hist.append(float(coverage))
            coverage_footprint_hist.append(float(coverage_footprint))
            coverage_footprint_kx_hist.append(float(coverage_footprint_kx))
            alt_up_m_cur, _ = _best_altitude_estimate_m(telem)
            alt_used_hist.append(float(max(0.0, alt_up_m_cur if alt_up_m_cur is not None else float(args.alt_m))))
            if time_to_cov95_s is None and coverage >= 0.95:
                time_to_cov95_s = float(elapsed)
            if coverage >= float(args.stop_cov):
                exit_reason = "stop_cov"
                break

            pos_xy_abs = np.asarray([float(telem.x), float(telem.y)], dtype=np.float32)
            pos_xy_rel = np.asarray([float(telem.x) - float(scan_origin_x), float(telem.y) - float(scan_origin_y)], dtype=np.float32)
            target_xyz_abs[0] = float(scan_origin_x) + float(target_xyz_rel[0])
            target_xyz_abs[1] = float(scan_origin_y) + float(target_xyz_rel[1])
            dist_abs = float(math.hypot(float(target_xyz_abs[0]) - float(pos_xy_abs[0]), float(target_xyz_abs[1]) - float(pos_xy_abs[1])))
            dist_rel = float(math.hypot(float(target_xyz_rel[0]) - float(pos_xy_rel[0]), float(target_xyz_rel[1]) - float(pos_xy_rel[1])))
            frame_mismatch = int(abs(float(dist_abs) - float(dist_rel)) > 1e-3)
            if frame_mismatch == 1:
                frame_mismatch_any = 1
                if not frame_mismatch_warned:
                    print(
                        f"[bridge][warn] frame mismatch: dist_abs={dist_abs:.4f} dist_rel={dist_rel:.4f} "
                        f"pos_abs=({float(pos_xy_abs[0]):.2f},{float(pos_xy_abs[1]):.2f}) "
                        f"tgt_abs=({float(target_xyz_abs[0]):.2f},{float(target_xyz_abs[1]):.2f})"
                    )
                    frame_mismatch_warned = True
            if int(policy_tick_flag) == 1:
                policy_action = np.asarray(
                    [float(target_xyz_rel[0]) - float(pos_xy_rel[0]), float(target_xyz_rel[1]) - float(pos_xy_rel[1])],
                    dtype=np.float32,
                )
                px, py = compute_next_target(
                    pos_xy_rel,
                    cov,
                    policy_action,
                )
                chosen_target_xy = np.asarray(
                    [float(scan_origin_x) + float(px), float(scan_origin_y) + float(py)],
                    dtype=np.float32,
                )
            dist_hist.append(dist_abs)
            adapt_window.append((float(elapsed), float(dist_abs)))
            cutoff_t = float(elapsed) - adapt_interval_s
            while len(adapt_window) > 1 and float(adapt_window[0][0]) < cutoff_t:
                adapt_window.popleft()
            if prev_dist is not None:
                progress_per_s = float((float(prev_dist) - dist_abs) / max(1e-6, dt))
            else:
                progress_per_s = 0.0
            progress_hist.append(float(progress_per_s))
            prev_dist = dist_abs
            if dist_abs < (best_dist_since_set - 1e-3):
                best_dist_since_set = dist_abs
                last_improve_t = loop_t

            adapt_event = None
            if loop_t >= adapt_next_t:
                if len(adapt_window) >= 2:
                    arr = np.asarray([float(d) for _, d in adapt_window], dtype=np.float64)
                    p95_dist_window = float(np.percentile(arr, 95.0))
                    t_first = float(adapt_window[0][0])
                    t_last = float(adapt_window[-1][0])
                    progress_window = 0.0
                    for i in range(1, len(adapt_window)):
                        progress_window += max(0.0, float(adapt_window[i - 1][1]) - float(adapt_window[i][1]))
                    mean_progress_window = float(progress_window / max(1e-6, t_last - t_first))
                else:
                    p95_dist_window = float(dist_abs)
                    mean_progress_window = 0.0
                adapt_last_p95 = float(p95_dist_window)
                adapt_last_progress = float(mean_progress_window)
                if adaptive_enabled:
                    bad_tracking = bool(
                        (p95_dist_window > float(args.dist_p95_high))
                        or (mean_progress_window < float(args.progress_low))
                    )
                    old_step_len = float(step_len)
                    old_accept_radius = float(accept_radius)
                    old_vxy_cap = float(vxy_cap_tuned)
                    if bad_tracking:
                        step_len = float(np.clip(step_len * 0.85, step_len_min, step_len_max))
                        accept_radius = float(np.clip(accept_radius * 1.10, accept_radius_min, accept_radius_max))
                        if use_vel_caps:
                            vxy_cap_tuned = max(0.5, float(vxy_cap_tuned * 0.90))
                        adapt_bad_count += 1
                        adapt_action = "bad"
                    else:
                        step_len = float(np.clip(step_len * 1.05, step_len_min, step_len_max))
                        accept_radius = float(np.clip(accept_radius * 0.98, accept_radius_min, accept_radius_max))
                        if use_vel_caps:
                            vxy_cap_tuned = min(2.0, float(vxy_cap_tuned * 1.03))
                        adapt_good_count += 1
                        adapt_action = "good"
                    target_stride = max(1, int(round(step_len / max(1e-3, dense_step_len))))
                    adapt_event_count += 1
                    adapt_event = {
                        "action": adapt_action,
                        "p95_dist_to_target": float(p95_dist_window),
                        "mean_progress_per_s": float(mean_progress_window),
                        "step_len_before": float(old_step_len),
                        "step_len_after": float(step_len),
                        "accept_radius_before": float(old_accept_radius),
                        "accept_radius_after": float(accept_radius),
                        "vxy_cap_before": float(old_vxy_cap),
                        "vxy_cap_after": float(vxy_cap_tuned),
                        "target_stride": int(target_stride),
                    }
                if not strict_lawnmower:
                    stable_tracking = bool(
                        (p95_dist_window < float(policy_hz_stable_dist_p95))
                        and (mean_progress_window > float(policy_hz_stable_progress))
                    )
                    old_policy_hz = float(policy_hz_current)
                    if stable_tracking:
                        policy_hz_current = max(policy_hz_min, float(policy_hz_current * 0.90))
                    else:
                        policy_hz_current = min(policy_hz_max, float(policy_hz_current * 1.10))
                    policy_dt = 1.0 / max(1e-3, float(policy_hz_current))
                    if abs(policy_hz_current - old_policy_hz) > 1e-6:
                        print(
                            f"[bridge][policy] stable={int(stable_tracking)} "
                            f"p95={p95_dist_window:.2f} prog={mean_progress_window:.2f} "
                            f"hz={old_policy_hz:.2f}->{policy_hz_current:.2f}"
                        )
                adapt_next_t = loop_t + adapt_interval_s
                if adapt_event is not None:
                    print(
                        f"[bridge][adapt] action={adapt_event['action']} p95={p95_dist_window:.2f} "
                        f"prog={mean_progress_window:.2f} step={step_len:.2f} "
                        f"accept={accept_radius:.2f} vxy={vxy_cap_tuned:.2f} stride={target_stride}"
                    )

            oob_event_flag = 0
            oob_event: dict[str, Any] = {}
            while len(clamp_times) > 0 and float(clamp_times[0]) < (loop_t - oob_clamp_fast_window_s):
                clamp_times.popleft()
            if oob_recovery_active:
                comfy_lim_x = max(0.0, half_w - (float(args.margin_m) + 1.0))
                comfy_lim_y = max(0.0, half_h - (float(args.margin_m) + 1.0))
                local_x = float(telem.x) - float(scan_origin_x)
                local_y = float(telem.y) - float(scan_origin_y)
                comfortably_inside = bool((abs(local_x) <= comfy_lim_x) and (abs(local_y) <= comfy_lim_y))
                if comfortably_inside or (loop_t >= oob_recovery_until_t):
                    oob_recovery_active = False
                    oob_recovery_time_sum_s += float(max(0.0, loop_t - oob_recovery_enter_t))
            if oob_recovery_enabled and (not oob_recovery_active):
                sustained_clamped_stall = bool(target_was_clamped and (loop_t - last_improve_t) > 2.0)
                clamp_fast = bool(len(clamp_times) >= oob_clamp_fast_count)
                if sustained_clamped_stall or clamp_fast:
                    oob_recovery_active = True
                    oob_recovery_enter_t = float(loop_t)
                    oob_recovery_until_t = float(loop_t + oob_recovery_seconds)
                    oob_event_count += 1
                    oob_event_flag = 1
                    sx, sy, _ = clamp_target_xy_minmax(
                        0.0,
                        0.0,
                        x_min_rel,
                        x_max_rel,
                        y_min_rel,
                        y_max_rel,
                    )
                    target_xyz_rel[0] = sx
                    target_xyz_rel[1] = sy
                    target_xyz_abs[0] = float(scan_origin_x) + float(target_xyz_rel[0])
                    target_xyz_abs[1] = float(scan_origin_y) + float(target_xyz_rel[1])
                    chosen_target_xy = np.asarray([float(target_xyz_abs[0]), float(target_xyz_abs[1])], dtype=np.float32)
                    policy_action = np.asarray(
                        [float(target_xyz_rel[0]) - float(pos_xy_rel[0]), float(target_xyz_rel[1]) - float(pos_xy_rel[1])],
                        dtype=np.float32,
                    )
                    target_was_clamped = False
                    target_set_t = loop_t
                    best_dist_since_set = float("inf")
                    last_improve_t = loop_t
                    oob_event = {
                        "reason": "clamped_stall" if sustained_clamped_stall else "clamp_fast",
                        "recovery_seconds": float(oob_recovery_seconds),
                        "center_xy": [float(scan_origin_x), float(scan_origin_y)],
                    }
                    print(
                        f"[bridge][oob] recovery_enter reason={oob_event['reason']} "
                        f"target=({sx:.2f},{sy:.2f})"
                    )

            pos_xy = pos_xy_rel
            corner_active = (False if strict_lawnmower else bool(loop_t < corner_until_t))
            local_x = float(telem.x) - float(scan_origin_x)
            local_y = float(telem.y) - float(scan_origin_y)
            min_boundary_dist = float(min(half_w - abs(local_x), half_h - abs(local_y)))
            step_len_eff = float(step_len)
            if step_sched_enabled:
                if bool(corner_active):
                    step_len_eff = float(step_len_eff * step_len_corner_mult)
                elif min_boundary_dist < float(boundary_near_m):
                    step_len_eff = float(step_len_eff * step_len_boundary_mult)
            step_len_eff = float(np.clip(step_len_eff, step_len_min, step_len_max))
            step_len_eff_hist.append(float(step_len_eff))
            target_stride_eff = max(1, int(round(step_len_eff / max(1e-3, dense_step_len))))
            if strict_lawnmower:
                target_stride_eff = 1

            lane_disabled = bool(
                corner_active
                or oob_recovery_active
                or (min_boundary_dist < (float(args.margin_m) + 1.0))
            )
            cross_track_signed = 0.0
            if lane_has_ref:
                lane_perp_now = np.asarray([-float(lane_dir[1]), float(lane_dir[0])], dtype=np.float32)
                cross_track_signed = float(np.dot(pos_xy - lane_origin, lane_perp_now))
                if lane_keep_enabled:
                    cross_track_abs_hist.append(abs(cross_track_signed))
            cross_track_val = float(abs(cross_track_signed))
            lookahead_delta_m = 0.0

            effective_accept_radius = float(accept_radius + (oob_recovery_accept_boost if oob_recovery_active else 0.0))
            refresh_reason = "none"
            cond_a = dist_abs <= float(effective_accept_radius)
            cond_b = (loop_t - target_set_t) >= float(args.max_hold_s)
            cond_c = bool(target_was_clamped and (loop_t - last_improve_t) > 1.0)
            if strict_lawnmower:
                strict_accept_ticks = (strict_accept_ticks + 1) if bool(cond_a) else 0
                should_refresh = bool(cond_a and (strict_accept_ticks >= int(strict_hold_steps)))
                refresh_reason = "accept_hold" if should_refresh else "strict_wait_accept_hold"
                if target_idx >= (len(targets) - 1) and strict_accept_ticks >= int(strict_hold_steps):
                    exit_reason = "strict_lawnmower_complete"
                    break
            elif args.target_refresh_mode == "always":
                should_refresh = True
                refresh_reason = "always"
            else:
                should_refresh = bool(cond_a or cond_b or cond_c)
                if cond_a:
                    refresh_reason = "accept"
                elif cond_b:
                    refresh_reason = "hold_timeout"
                elif cond_c:
                    refresh_reason = "clamped_stall"
            if oob_recovery_active:
                should_refresh = False
                refresh_reason = "oob_recovery_hold"

            target_update_flag = 0
            update_clamped_flag = 0
            if should_refresh and target_idx < (len(targets) - 1) and int(policy_tick_flag) == 1:
                old_target = target_xyz_rel.copy()
                target_idx = min(len(targets) - 1, target_idx + int(max(1, target_stride_eff)))
                candidate = targets[target_idx].copy()
                dir_hint = np.asarray([float(candidate[0]) - float(pos_xy[0]), float(candidate[1]) - float(pos_xy[1])], dtype=np.float32)
                dir_unit = _unit_xy(dir_hint, default_xy=policy_action)
                policy_action = (dir_unit * float(step_len_eff)).astype(np.float32)
                if strict_lawnmower:
                    target_xy = np.asarray([float(candidate[0]), float(candidate[1])], dtype=np.float32)
                else:
                    nx, ny = compute_next_target(pos_xy, cov, policy_action)
                    target_xy = np.asarray([float(nx), float(ny)], dtype=np.float32)
                speed_xy = float(math.hypot(float(telem.vx), float(telem.vy)))
                if lookahead_enabled and (not oob_recovery_active) and speed_xy > 0.3:
                    lookahead_delta = _clamp_norm_xy(
                        np.asarray([float(telem.vx), float(telem.vy)], dtype=np.float32) * float(lookahead_time_s),
                        float(lookahead_cap_m),
                    )
                    target_xy = target_xy + lookahead_delta
                    lookahead_delta_m = float(np.linalg.norm(lookahead_delta))

                if lane_dir_mode == "fixed":
                    if not lane_has_ref:
                        lane_dir = dir_unit.copy()
                        lane_origin = pos_xy.copy()
                        lane_has_ref = True
                        num_lane_resets += 1
                else:
                    if not lane_has_ref:
                        lane_dir = dir_unit.copy()
                        lane_origin = pos_xy.copy()
                        lane_has_ref = True
                        num_lane_resets += 1
                    else:
                        lane_dot = float(np.clip(np.dot(lane_dir, dir_unit), -1.0, 1.0))
                        lane_ang_deg = float(np.degrees(np.arccos(lane_dot)))
                        if lane_ang_deg > 60.0:
                            lane_dir = dir_unit.copy()
                            lane_origin = pos_xy.copy()
                            num_lane_resets += 1

                if lane_keep_enabled and lane_has_ref and (not lane_disabled):
                    lane_perp = np.asarray([-float(lane_dir[1]), float(lane_dir[0])], dtype=np.float32)
                    cross_track_signed = float(np.dot(pos_xy - lane_origin, lane_perp))
                    corr_mag = min(float(abs(cross_track_signed) * lane_kp), float(lane_max_corr_m))
                    corr_vec = lane_perp * float(np.sign(cross_track_signed)) * float(corr_mag)
                    target_xy = target_xy - corr_vec
                    cross_track_val = float(abs(cross_track_signed))

                candidate[0] = float(target_xy[0])
                candidate[1] = float(target_xy[1])
                cx, cy, was_clamped_now = clamp_target_xy_minmax(
                    float(candidate[0]),
                    float(candidate[1]),
                    x_min_rel,
                    x_max_rel,
                    y_min_rel,
                    y_max_rel,
                )
                candidate[0] = cx
                candidate[1] = cy
                target_xyz_rel = candidate
                target_xyz_abs[0] = float(scan_origin_x) + float(target_xyz_rel[0])
                target_xyz_abs[1] = float(scan_origin_y) + float(target_xyz_rel[1])
                chosen_target_xy = np.asarray([float(target_xyz_abs[0]), float(target_xyz_abs[1])], dtype=np.float32)
                target_was_clamped = bool(was_clamped_now)
                if was_clamped_now:
                    clamp_count += 1
                    update_clamped_flag = 1
                    clamp_times.append(float(loop_t))
                num_target_updates += 1
                target_update_flag = 1
                target_set_t = loop_t
                best_dist_since_set = float("inf")
                last_improve_t = loop_t
                strict_accept_ticks = 0
                v0 = np.asarray([float(old_target[0]) - float(pos_xy[0]), float(old_target[1]) - float(pos_xy[1])], dtype=np.float64)
                v1 = np.asarray([float(target_xyz_rel[0]) - float(pos_xy[0]), float(target_xyz_rel[1]) - float(pos_xy[1])], dtype=np.float64)
                n0 = float(np.linalg.norm(v0))
                n1 = float(np.linalg.norm(v1))
                if (not strict_lawnmower) and n0 > 1e-6 and n1 > 1e-6:
                    cang = float(np.clip(np.dot(v0, v1) / (n0 * n1), -1.0, 1.0))
                    ang_deg = float(np.degrees(np.arccos(cang)))
                    if ang_deg >= float(args.corner_angle_deg):
                        num_corner_events += 1
                        corner_until_t = loop_t + max(0.0, float(args.corner_slow_seconds))
                        corner_active = True
                if int(args.dry_run) == 1:
                    print(
                        f"[bridge][dry] target_update idx={target_idx}/{len(targets)-1} reason={refresh_reason} "
                        f"clamped={int(bool(was_clamped_now))}"
                    )
            elif should_refresh and int(policy_tick_flag) == 0:
                refresh_reason = "wait_policy_tick"
            lookahead_delta_hist.append(float(lookahead_delta_m))

            current_vxy_cap = float(vxy_cap_tuned)
            if strict_lawnmower:
                r = float(max(1e-6, effective_accept_radius))
                slowdown_dist = float(strict_slowdown_mult * r)
                if dist_abs <= slowdown_dist:
                    frac = float(np.clip(dist_abs / max(1e-6, slowdown_dist), 0.0, 1.0))
                    current_vxy_cap = float(strict_vmin + (float(vxy_cap_tuned) - float(strict_vmin)) * frac)
                else:
                    current_vxy_cap = float(vxy_cap_tuned)
            if bool(use_vel_caps) and bool(corner_active):
                current_vxy_cap = min(current_vxy_cap, float(args.corner_vxy_cap))
            if bool(use_vel_caps) and bool(oob_recovery_active):
                current_vxy_cap = min(current_vxy_cap, float(oob_recovery_vxy_cap))

            if int(args.dry_run) == 0:
                vx_cmd, vy_cmd, vz_cmd = _send_local_ned_target(
                    mav,
                    mavutil,
                    target_xyz_abs,
                    telem,
                    use_vel_caps=bool(use_vel_caps),
                    vxy_cap=float(current_vxy_cap),
                    vz_cap=float(vz_cap),
                    fixed_yaw_rad=(float(strict_fixed_yaw_rad) if strict_lawnmower else None),
                    yaw_mode=str(yaw_mode_effective),
                    yaw_rate_max_dps=float(args.yaw_rate_max_dps),
                    yaw_dt_s=float(dt),
                )
            else:
                dx = float(target_xyz_abs[0]) - telem.x
                dy = float(target_xyz_abs[1]) - telem.y
                dz = float(target_xyz_abs[2]) - telem.z
                dxy = float(math.hypot(dx, dy))
                if dxy > 1e-6:
                    vx_cmd = float(dx / dxy * (current_vxy_cap if use_vel_caps else 1.0))
                    vy_cmd = float(dy / dxy * (current_vxy_cap if use_vel_caps else 1.0))
                else:
                    vx_cmd = 0.0
                    vy_cmd = 0.0
                vz_cmd = float(np.clip(dz, -float(vz_cap), float(vz_cap))) if use_vel_caps else 0.0
                telem.vx = vx_cmd
                telem.vy = vy_cmd
                telem.vz = vz_cmd
                telem.x += vx_cmd * dt
                telem.y += vy_cmd * dt
                telem.z += vz_cmd * dt
                if step_count < 5 or target_update_flag:
                    print(
                        f"[bridge][dry] step={step_count+1} dist={dist_abs:.2f} cap={current_vxy_cap:.2f} "
                        f"corner={int(corner_active)} refresh={refresh_reason}"
                    )

            patch = cov.local_patch(
                float(telem.x) - float(scan_origin_x),
                float(telem.y) - float(scan_origin_y),
                patch_size=patch_size,
            )
            bx, by = cov.boundary_features(float(telem.x) - float(scan_origin_x), float(telem.y) - float(scan_origin_y))
            row = {
                "t": float(elapsed),
                "selected_profile": selected_profile,
                "model_path": model_path,
                "scan_max_steps_eff": int(scan_max_steps_eff),
                "scan_origin_x": float(scan_origin_x),
                "scan_origin_y": float(scan_origin_y),
                "pos_x_abs": float(pos_xy_abs[0]),
                "pos_y_abs": float(pos_xy_abs[1]),
                "tgt_x_abs": float(target_xyz_abs[0]),
                "tgt_y_abs": float(target_xyz_abs[1]),
                "dist_abs": float(dist_abs),
                "pos_x_rel": float(pos_xy_rel[0]),
                "pos_y_rel": float(pos_xy_rel[1]),
                "tgt_x_rel": float(target_xyz_rel[0]),
                "tgt_y_rel": float(target_xyz_rel[1]),
                "dist_rel": float(dist_rel),
                "frame_mismatch": int(frame_mismatch),
                "pos": [float(telem.x), float(telem.y), float(telem.z)],
                "vel": [float(telem.vx), float(telem.vy), float(telem.vz)],
                "yaw": float(telem.yaw),
                "mode": str(telem.mode),
                "ekf_ok": bool(telem.ekf_ok),
                "gps_ok": bool(telem.gps_ok),
                "target": [float(target_xyz_abs[0]), float(target_xyz_abs[1]), float(target_xyz_abs[2])],
                "chosen_target_xy": [float(chosen_target_xy[0]), float(chosen_target_xy[1])],
                "policy_action_raw": [float(policy_action[0]), float(policy_action[1])],
                "dist_to_target": float(dist_abs),
                "progress_per_s": float(progress_per_s),
                "cross_track": float(cross_track_val),
                "lookahead_delta_m": float(lookahead_delta_m),
                "coverage_frac": float(coverage),
                "coverage_frac_footprint": float(coverage_footprint),
                "time_to_95_reached_flag": int(time_to_cov95_s is not None),
                "time_to_95_s": None if time_to_cov95_s is None else float(time_to_cov95_s),
                "clamp_flag": int(update_clamped_flag),
                "target_update_flag": int(target_update_flag),
                "policy_tick_flag": int(policy_tick_flag),
                "policy_hz_current": float(policy_hz_current),
                "corner_active": int(bool(corner_active)),
                "current_vxy_cap": float(current_vxy_cap),
                "current_step_len": float(step_len),
                "step_len_eff": float(step_len_eff),
                "current_accept_radius": float(accept_radius),
                "effective_accept_radius": float(effective_accept_radius),
                "target_stride": int(target_stride_eff),
                "adaptive_tracking": int(adaptive_enabled),
                "adaptive_event_flag": int(adapt_event is not None),
                "adaptive_event": adapt_event if adapt_event is not None else {},
                "oob_recovery_active": int(oob_recovery_active),
                "oob_event_flag": int(oob_event_flag),
                "oob_event": oob_event,
                "obs_patch_size": int(patch_size),
                "obs_patch": patch.astype(float).tolist(),
                "boundary_feat": [float(bx), float(by)] if boundary_feat else [],
            }
            telem_f.write(json.dumps(row) + "\n")

            if loop_t >= log_next_t:
                ok_health, _ = _health_ok(telem, str(args.ekf_mode))
                print(
                    f"[bridge] t={elapsed:6.1f}s cov={coverage:.3f} dist={dist_abs:.2f} "
                    f"clamp_count={clamp_count} mode={telem.mode} health_ok={int(ok_health)}"
                )
                log_next_t = loop_t + 1.0

            step_count += 1
            sleep_s = max(0.0, (t0 + step_count * dt) - time.monotonic())
            if sleep_s > 0.0:
                time.sleep(sleep_s)

    if oob_recovery_active:
        oob_recovery_time_sum_s += float(max(0.0, time.monotonic() - oob_recovery_enter_t))
    if (
        int(args.dry_run) == 0
        and str(exit_reason) == "strict_lawnmower_complete"
        and int(args.return_land_on_complete) == 1
        and (mav is not None)
        and (mavutil is not None)
    ):
        print(
            f"[bridge] strict completion reached; returning to spawn ({float(scan_origin_x):.2f}, {float(scan_origin_y):.2f}) then LAND"
        )
        return_land_result = _return_to_spawn_and_land(
            mav=mav,
            mavutil=mavutil,
            telem=telem,
            scan_origin_x=float(scan_origin_x),
            scan_origin_y=float(scan_origin_y),
            alt_m=float(args.alt_m),
            use_vel_caps=bool(use_vel_caps),
            vxy_cap=float(base_vxy_cap),
            vz_cap=float(vz_cap),
            timeout_s=float(args.return_land_timeout_s),
            reach_radius_m=float(args.return_land_radius_m),
            fixed_yaw_rad=(float(strict_fixed_yaw_rad) if strict_lawnmower else None),
            yaw_mode=str(yaw_mode_effective),
            yaw_rate_max_dps=float(args.yaw_rate_max_dps),
            yaw_dt_s=float(dt),
        )
        print(
            "[bridge] return+land: "
            f"reached_spawn={int(return_land_result.get('reached_spawn', 0))} "
            f"land_mode_set={int(return_land_result.get('land_mode_set', 0))} "
            f"dist={float(return_land_result.get('final_dist_to_spawn_m', float('nan'))):.2f} "
            f"reason={return_land_result.get('reason', '')}"
        )
    elif int(args.return_land_on_complete) == 1 and str(exit_reason) == "strict_lawnmower_complete":
        print("[bridge][dry] strict completion reached; would return to spawn, set LAND, and exit.")
    duration_s = float(max(0.0, time.monotonic() - t0))
    mean_dist = float(np.mean(np.asarray(dist_hist, dtype=np.float64))) if dist_hist else 0.0
    p95_dist = float(np.percentile(np.asarray(dist_hist, dtype=np.float64), 95.0)) if dist_hist else 0.0
    coverage_mean = float(np.mean(np.asarray(coverage_hist, dtype=np.float64))) if coverage_hist else 0.0
    coverage_footprint_mean = (
        float(np.mean(np.asarray(coverage_footprint_hist, dtype=np.float64))) if coverage_footprint_hist else 0.0
    )
    coverage_footprint_kx_mean = (
        float(np.mean(np.asarray(coverage_footprint_kx_hist, dtype=np.float64))) if coverage_footprint_kx_hist else 0.0
    )
    alt_used_mean_m = float(np.mean(np.asarray(alt_used_hist, dtype=np.float64))) if alt_used_hist else float(args.alt_m)
    mean_prog_per_s = float(np.mean(np.asarray(progress_hist, dtype=np.float64))) if progress_hist else 0.0
    p95_prog_per_s = float(np.percentile(np.asarray(progress_hist, dtype=np.float64), 95.0)) if progress_hist else 0.0
    clamp_rate_per_min = float(clamp_count / max(duration_s / 60.0, 1e-6))
    mean_abs_cross_track = (
        float(np.mean(np.asarray(cross_track_abs_hist, dtype=np.float64)))
        if (lane_keep_enabled and cross_track_abs_hist)
        else 0.0
    )
    mean_lookahead_delta_m = (
        float(np.mean(np.asarray(lookahead_delta_hist, dtype=np.float64))) if lookahead_delta_hist else 0.0
    )
    mean_step_len_eff = float(np.mean(np.asarray(step_len_eff_hist, dtype=np.float64))) if step_len_eff_hist else float(step_len)
    footprint_cells_total = int(cov_footprint.total_cells)
    footprint_cells_covered_1x = 0
    footprint_cells_overlap_2x = 0
    footprint_overlap_ratio_covered = 0.0
    footprint_pass_count_mean_on_covered = 0.0
    footprint_extra_passes_total = 0
    footprint_overlap_excess_ratio = 0.0
    if cov_footprint.pass_counts is not None:
        pc = np.asarray(cov_footprint.pass_counts, dtype=np.int64)
        covered_mask = pc >= 1
        overlap_mask = pc >= 2
        footprint_cells_covered_1x = int(np.count_nonzero(covered_mask))
        footprint_cells_overlap_2x = int(np.count_nonzero(overlap_mask))
        footprint_overlap_ratio_covered = float(
            footprint_cells_overlap_2x / max(1, footprint_cells_covered_1x)
        )
        if footprint_cells_covered_1x > 0:
            footprint_pass_count_mean_on_covered = float(np.mean(pc[covered_mask]))
        footprint_extra_passes_total = int(np.sum(np.maximum(pc - 1, 0)))
        footprint_passes_total = int(np.sum(pc))
        footprint_overlap_excess_ratio = float(
            footprint_extra_passes_total / max(1, footprint_passes_total)
        )
    summary: dict[str, Any] = {
        "selected_profile": selected_profile,
        "model_path": model_path,
        "sitl_recommended_enabled": int(sitl_enabled),
        "sitl_recommended_source": str(sitl_recommended_source_used),
        "sitl_recommended_arm": str(sitl_recommended_arm),
        "sitl_recommended_profile_match": int(bool(sitl_recommended_profile_match)),
        "efficiency_profile": str(efficiency_profile_name),
        "efficiency_profile_applied": [str(x) for x in efficiency_profile_applied],
        "connection_requested": str(connection_requested),
        "connection_used": str(connection_used),
        "requested_message_intervals": [str(x) for x in requested_message_intervals],
        "anchor_origin": int(args.anchor_origin),
        "scan_origin_mode": str(scan_origin_mode),
        "scan_origin_ref": str(scan_origin_ref),
        "scan_origin_ref_input_xy": [float(args.scan_origin_x), float(args.scan_origin_y)],
        "scan_center_offset_from_ref_xy": [float(center_off_x), float(center_off_y)],
        "scan_origin_xy": [float(scan_origin_x), float(scan_origin_y)],
        "bounds_minmax_xy": [float(x_min_scan), float(x_max_scan), float(y_min_scan), float(y_max_scan)],
        "oob_center_xy": [float(scan_origin_x), float(scan_origin_y)],
        "frame_mismatch_any": int(frame_mismatch_any),
        "prefer_sitl_tcp": int(args.prefer_sitl_tcp),
        "preflight_timeout_s": float(args.preflight_timeout_s),
        "require_mode_known": int(args.require_mode_known),
        "ekf_mode": str(args.ekf_mode),
        "ack_verbosity": str(args.ack_verbosity),
        "preflight_result": preflight_result,
        "last_heartbeat": _heartbeat_snapshot(telem),
        "auto_guided": int(args.auto_guided),
        "auto_arm": int(args.auto_arm),
        "auto_takeoff_m": float(args.auto_takeoff_m),
        "takeoff_skipped": int(takeoff_skipped),
        "takeoff_skip_reason": str(takeoff_skip_reason),
        "preflight_events": preflight_events,
        "scale": float(scale_upper),
        "bounds": [float(bounds_w), float(bounds_h)],
        "margin_m": float(args.margin_m),
        "step_len": float(step_len),
        "accept_radius": float(accept_radius),
        "final_step_len_m": float(step_len),
        "mean_step_len_eff": float(mean_step_len_eff),
        "final_accept_radius_m": float(accept_radius),
        "rate_hz": float(args.rate_hz),
        "policy_hz": float(policy_hz),
        "policy_hz_min": float(policy_hz_min),
        "policy_hz_max": float(policy_hz_max),
        "policy_hz_current": float(policy_hz_current),
        "coverage_hz": float(coverage_hz),
        "num_policy_ticks": int(num_policy_ticks),
        "scan_max_steps_eff": int(scan_max_steps_eff),
        "duration_s": float(duration_s),
        "final_coverage": float(cov.coverage_frac()),
        "final_coverage_path": float(cov.coverage_frac()),
        "final_coverage_footprint": float(cov_footprint.coverage_frac_at_least(1)),
        "final_coverage_footprint_1x": float(cov_footprint.coverage_frac_at_least(1)),
        "final_coverage_footprint_kx": float(cov_footprint.coverage_frac_at_least(footprint_min_passes)),
        "coverage_mean": float(coverage_mean),
        "coverage_footprint_mean": float(coverage_footprint_mean),
        "coverage_footprint_kx_mean": float(coverage_footprint_kx_mean),
        "time_to_95_s": None if time_to_cov95_s is None else float(time_to_cov95_s),
        "hfov_deg": float(args.camera_hfov_deg),
        "vfov_deg": float(args.camera_vfov_deg),
        "footprint_model": str(footprint_model),
        "footprint_min_passes": int(footprint_min_passes),
        "footprint_fov_scale": float(footprint_fov_scale),
        "footprint_count_dist_factor": float(footprint_count_dist_factor),
        "footprint_count_dist_min_m": float(footprint_count_dist_min_m),
        "footprint_count_updates": int(footprint_count_updates),
        "footprint_count_skips": int(footprint_count_skips),
        "alt_used_m": float(alt_used_mean_m),
        "footprint_cells_total": int(footprint_cells_total),
        "footprint_cells_covered_1x": int(footprint_cells_covered_1x),
        "footprint_cells_overlap_2x": int(footprint_cells_overlap_2x),
        "footprint_overlap_ratio_covered": float(footprint_overlap_ratio_covered),
        "footprint_pass_count_mean_on_covered": float(footprint_pass_count_mean_on_covered),
        "footprint_extra_passes_total": int(footprint_extra_passes_total),
        "footprint_overlap_excess_ratio": float(footprint_overlap_excess_ratio),
        "boundary_covered_frac": float(cov.boundary_covered_frac()),
        "interior_covered_frac": float(cov.interior_covered_frac()),
        "clamp_count": int(clamp_count),
        "clamp_rate_per_min": float(clamp_rate_per_min),
        "oob_events": int(oob_event_count),
        "oob_recovery_time_s": float(oob_recovery_time_sum_s),
        "mean_abs_cross_track": float(mean_abs_cross_track),
        "num_lane_resets": int(num_lane_resets),
        "lookahead_enabled": int(lookahead_enabled),
        "mean_lookahead_delta_m": float(mean_lookahead_delta_m),
        "lane_keep_enabled": int(lane_keep_enabled),
        "exit_reason": str(exit_reason),
        "mean_dist_to_target": float(mean_dist),
        "p95_dist_to_target": float(p95_dist),
        "mean_progress_per_s": float(mean_prog_per_s),
        "p95_progress_per_s": float(p95_prog_per_s),
        "num_target_updates": int(num_target_updates),
        "num_corner_events": int(num_corner_events),
        "target_refresh_mode": str(args.target_refresh_mode),
        "turn_style": str(args.turn_style),
        "turn_radius_m": float(args.turn_radius_m),
        "strict_lawnmower": int(strict_lawnmower),
        "strict_hold_steps": int(strict_hold_steps),
        "strict_fixed_yaw_deg": float(strict_fixed_yaw_deg),
        "strict_vmin": float(strict_vmin),
        "strict_slowdown_mult": float(strict_slowdown_mult),
        "yaw_mode": str(yaw_mode_effective),
        "yaw_rate_max_dps": float(args.yaw_rate_max_dps),
        "fixed_yaw_deg_final": (
            float(strict_fixed_yaw_deg) if str(yaw_mode_effective) == "fixed" else None
        ),
        "return_land_on_complete": int(args.return_land_on_complete),
        "return_land_timeout_s": float(args.return_land_timeout_s),
        "return_land_radius_m": float(args.return_land_radius_m),
        "return_land_result": return_land_result,
        "use_vel_caps": int(args.use_vel_caps),
        "adaptive_tracking": int(adaptive_enabled),
        "adaptive_event_count": int(adapt_event_count),
        "adaptive_bad_count": int(adapt_bad_count),
        "adaptive_good_count": int(adapt_good_count),
        "adaptive_last_p95_dist_to_target": float(adapt_last_p95),
        "adaptive_last_mean_progress_per_s": float(adapt_last_progress),
        "initial_step_len": float(initial_step_len),
        "initial_accept_radius": float(initial_accept_radius),
        "initial_vxy_cap": float(initial_vxy_cap),
        "vxy_cap": float(vxy_cap_tuned),
        "final_vxy_cap": float(vxy_cap_tuned),
        "vz_cap": float(vz_cap),
        "corner_vxy_cap": float(args.corner_vxy_cap),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[bridge] telemetry: {telemetry_path}")
    print(f"[bridge] summary:   {summary_path}")
    print(f"[bridge] result: {json.dumps(summary, indent=2)}")
    raise SystemExit(int(rc))


if __name__ == "__main__":
    main()
