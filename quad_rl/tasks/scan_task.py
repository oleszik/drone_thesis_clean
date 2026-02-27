from __future__ import annotations

import numpy as np

from quad_rl.tasks.base_task import BaseTask, TaskStep, wrap_angle


class ScanTask(BaseTask):
    name = "scan"

    def __init__(self, cfg=None) -> None:
        super().__init__(cfg)
        self.path_points = np.zeros((2, 3), dtype=np.float32)
        self.seg_lengths = np.ones((1,), dtype=np.float32)
        self.cum_lengths = np.zeros((2,), dtype=np.float32)
        self.turn_vertices = np.zeros((2,), dtype=bool)
        self.total_length = 0.0
        self.seg_idx = 0
        self.s_on_seg = 0.0
        self.prev_progress = 0.0
        self.turn_hold_count = 0
        self.target_point = np.zeros((3,), dtype=np.float32)
        self.target_yaw = 0.0
        self.done = False
        self.segment_eps = 1e-3
        self.coverage_cell_size = 0.25
        self.coverage_radius = 0.0
        self.coverage_x_min = 0.0
        self.coverage_y_min = 0.0
        self.coverage_nx = 1
        self.coverage_ny = 1
        self.covered_mask = np.zeros((1, 1), dtype=bool)
        self.covered_cells = 0
        self.coverage_total_cells = 1
        self.revisit_steps = 0
        self.time_to_95_step = -1
        self.ep_path_len_scale = 1.0
        self.ep_lookahead = float(getattr(self.cfg, "scan_lookahead", 1.0))
        self.ep_spacing = float(getattr(self.cfg, "scan_spacing", 0.8))
        self.prev_v_xy = np.zeros((2,), dtype=np.float32)

    @staticmethod
    def _sample_range(
        rng: np.random.Generator,
        low: float,
        high: float,
        default: float,
        *,
        min_value: float,
    ) -> float:
        lo = float(low)
        hi = float(high)
        if hi < lo:
            lo, hi = hi, lo
        if not np.isfinite(lo) or not np.isfinite(hi):
            return max(float(default), min_value)
        if abs(hi - lo) < 1e-9:
            return max(float(lo), min_value)
        return max(float(rng.uniform(lo, hi)), min_value)

    def _build_lawnmower(self, center: np.ndarray) -> np.ndarray:
        rows = int(self.cfg.scan_rows)
        cols = int(self.cfg.scan_cols)
        scale = max(float(self.ep_path_len_scale), 0.1)
        if abs(scale - 1.0) > 1e-6:
            rc_scale = float(np.sqrt(scale))
            rows = max(2, int(round(rows * rc_scale)))
            cols = max(2, int(round(cols * rc_scale)))
        spacing = max(float(self.ep_spacing), float(getattr(self.cfg, "scan_min_wp_spacing", 0.0)))

        points = []
        x0 = center[0] - 0.5 * spacing * max(0, rows - 1)
        y0 = center[1] - 0.5 * spacing * max(0, cols - 1)
        z = float(center[2])

        for r in range(rows):
            y_indices = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
            x = x0 + r * spacing
            for c in y_indices:
                y = y0 + c * spacing
                points.append(np.array([x, y, z], dtype=np.float32))

        return np.asarray(points, dtype=np.float32)

    def _compact_path(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        min_spacing = float(getattr(self.cfg, "scan_min_wp_spacing", 0.0))
        compact = [points[0]]
        for p in points[1:]:
            if np.linalg.norm((p - compact[-1])[:2]) >= max(1e-6, min_spacing):
                compact.append(p)

        compact_path = np.asarray(compact, dtype=np.float32)
        if len(compact_path) < 2:
            dx = max(0.5, min_spacing)
            p1 = compact_path[0] + np.array([dx, 0.0, 0.0], dtype=np.float32)
            compact_path = np.vstack([compact_path, p1]).astype(np.float32)
        return compact_path

    def _clamp_path_to_world(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        bound = float(getattr(self.cfg, "world_xy_bound", 0.0))
        if bound <= 0.0:
            return np.asarray(points, dtype=np.float32).copy()
        margin = max(float(getattr(self.cfg, "scan_coverage_radius", 0.5)), 0.5, 1e-3)
        limit = max(0.0, bound - margin)
        clamped = np.asarray(points, dtype=np.float32).copy()
        clamped[:, 0] = np.clip(clamped[:, 0], -limit, limit)
        clamped[:, 1] = np.clip(clamped[:, 1], -limit, limit)
        return clamped

    def _rebuild_path_cache(self) -> None:
        self.seg_lengths = np.linalg.norm(np.diff(self.path_points[:, :2], axis=0), axis=1).astype(np.float32)
        self.cum_lengths = np.concatenate(
            [np.zeros((1,), dtype=np.float32), np.cumsum(self.seg_lengths, dtype=np.float32)]
        )
        self.total_length = float(self.cum_lengths[-1])

        self.turn_vertices = np.zeros((len(self.path_points),), dtype=bool)
        for i in range(1, len(self.path_points) - 1):
            v0 = self.path_points[i] - self.path_points[i - 1]
            v1 = self.path_points[i + 1] - self.path_points[i]
            n0 = float(np.linalg.norm(v0[:2]))
            n1 = float(np.linalg.norm(v1[:2]))
            if n0 <= 1e-6 or n1 <= 1e-6:
                continue
            cosang = float(np.clip(np.dot(v0[:2], v1[:2]) / (n0 * n1), -1.0, 1.0))
            angle = float(np.arccos(cosang))
            if angle > 0.2:
                self.turn_vertices[i] = True

    def _closest_on_segment(self, pos: np.ndarray, seg_idx: int):
        a = self.path_points[seg_idx]
        b = self.path_points[seg_idx + 1]
        seg_len = float(self.seg_lengths[seg_idx])
        if seg_len <= 1e-6:
            cte = float(np.linalg.norm(pos[:2] - a[:2]))
            return a.copy(), 0.0, cte

        seg_dir = (b[:2] - a[:2]) / seg_len
        s = float(np.dot(pos[:2] - a[:2], seg_dir))
        s = float(np.clip(s, 0.0, seg_len))

        closest = a.copy()
        closest[:2] = a[:2] + seg_dir * s
        frac = s / seg_len
        closest[2] = float(a[2] + frac * (b[2] - a[2]))
        cte = float(np.linalg.norm(pos[:2] - closest[:2]))
        return closest, s, cte

    def _project_s_raw(self, pos: np.ndarray, seg_idx: int) -> tuple[float, float]:
        a = self.path_points[seg_idx]
        b = self.path_points[seg_idx + 1]
        seg_len = float(self.seg_lengths[seg_idx])
        if seg_len <= 1e-6:
            return 0.0, seg_len
        seg_dir = (b[:2] - a[:2]) / seg_len
        s_raw = float(np.dot(pos[:2] - a[:2], seg_dir))
        return s_raw, seg_len

    def _point_and_tangent_at_progress(self, progress: float):
        g = float(np.clip(progress, 0.0, self.total_length))
        last_seg = len(self.seg_lengths) - 1
        if last_seg < 0:
            return self.path_points[0].copy(), self.target_yaw

        seg_idx = int(np.searchsorted(self.cum_lengths, g, side="right") - 1)
        seg_idx = int(np.clip(seg_idx, 0, last_seg))
        seg_len = float(self.seg_lengths[seg_idx])
        s_local = float(g - float(self.cum_lengths[seg_idx]))
        s_local = float(np.clip(s_local, 0.0, seg_len))

        a = self.path_points[seg_idx]
        b = self.path_points[seg_idx + 1]
        if seg_len <= 1e-6:
            return a.copy(), self.target_yaw

        seg_dir = (b[:2] - a[:2]) / seg_len
        point = a.copy()
        point[:2] = a[:2] + seg_dir * s_local
        point[2] = float(a[2] + (s_local / seg_len) * (b[2] - a[2]))
        yaw = float(np.arctan2(seg_dir[1], seg_dir[0]))
        return point, yaw

    def _init_coverage_grid(self) -> None:
        cell_size = max(float(getattr(self.cfg, "scan_cov_cell_size", 0.25)), 1e-3)
        spacing = max(float(getattr(self.cfg, "scan_spacing", 1.0)), cell_size)
        cov_radius = max(float(getattr(self.cfg, "scan_coverage_radius", 0.5 * spacing)), 0.0)

        path_xy = self.path_points[:, :2]
        mins = np.min(path_xy, axis=0)
        maxs = np.max(path_xy, axis=0)
        # Coverage is defined over the scan footprint (not radius-padded world area).
        pad = 0.5 * cell_size
        mins = mins - pad
        maxs = maxs + pad
        span = np.maximum(maxs - mins, cell_size)

        nx = max(1, int(np.ceil(float(span[0]) / cell_size)))
        ny = max(1, int(np.ceil(float(span[1]) / cell_size)))

        self.coverage_cell_size = float(cell_size)
        self.coverage_radius = float(cov_radius)
        self.coverage_x_min = float(mins[0])
        self.coverage_y_min = float(mins[1])
        self.coverage_nx = int(nx)
        self.coverage_ny = int(ny)
        self.covered_mask = np.zeros((self.coverage_nx, self.coverage_ny), dtype=bool)
        self.covered_cells = 0
        self.coverage_total_cells = int(self.coverage_nx * self.coverage_ny)
        self.revisit_steps = 0
        self.time_to_95_step = -1

    def _coverage_ratio(self) -> float:
        return float(self.covered_cells / max(self.coverage_total_cells, 1))

    def _overlap_ratio(self) -> float:
        return float(self.revisit_steps / max(self.covered_cells, 1))

    def _mark_coverage(self, pos_xy: np.ndarray, step_idx: int) -> tuple[int, int]:
        x = float(pos_xy[0])
        y = float(pos_xy[1])
        cs = float(self.coverage_cell_size)
        r = float(self.coverage_radius)
        r2 = r * r

        ix0 = int(np.floor((x - r - self.coverage_x_min) / cs))
        ix1 = int(np.floor((x + r - self.coverage_x_min) / cs))
        iy0 = int(np.floor((y - r - self.coverage_y_min) / cs))
        iy1 = int(np.floor((y + r - self.coverage_y_min) / cs))

        ix0 = int(np.clip(ix0, 0, self.coverage_nx - 1))
        ix1 = int(np.clip(ix1, 0, self.coverage_nx - 1))
        iy0 = int(np.clip(iy0, 0, self.coverage_ny - 1))
        iy1 = int(np.clip(iy1, 0, self.coverage_ny - 1))

        new_hits = 0
        revisit_hits = 0
        for ix in range(ix0, ix1 + 1):
            cx = self.coverage_x_min + (float(ix) + 0.5) * cs
            dx = cx - x
            for iy in range(iy0, iy1 + 1):
                cy = self.coverage_y_min + (float(iy) + 0.5) * cs
                dy = cy - y
                if (dx * dx + dy * dy) > r2:
                    continue
                if bool(self.covered_mask[ix, iy]):
                    revisit_hits += 1
                else:
                    self.covered_mask[ix, iy] = True
                    new_hits += 1

        if new_hits > 0:
            self.covered_cells += int(new_hits)
        elif step_idx > 0 and revisit_hits > 0:
            self.revisit_steps += 1

        if self.time_to_95_step < 0 and self._coverage_ratio() >= 0.95:
            self.time_to_95_step = int(step_idx)
        return int(new_hits), int(revisit_hits)

    def reset(self, env, rng: np.random.Generator) -> None:
        super().reset(env, rng)
        self.ep_path_len_scale = self._sample_range(
            rng,
            getattr(self.cfg, "scan_path_len_scale_min", getattr(self.cfg, "scan_path_len_scale", 1.0)),
            getattr(self.cfg, "scan_path_len_scale_max", getattr(self.cfg, "scan_path_len_scale", 1.0)),
            getattr(self.cfg, "scan_path_len_scale", 1.0),
            min_value=0.1,
        )
        self.ep_lookahead = self._sample_range(
            rng,
            getattr(self.cfg, "scan_lookahead_min", getattr(self.cfg, "scan_lookahead", 1.0)),
            getattr(self.cfg, "scan_lookahead_max", getattr(self.cfg, "scan_lookahead", 1.0)),
            getattr(self.cfg, "scan_lookahead", 1.0),
            min_value=0.0,
        )
        spacing_base = float(getattr(self.cfg, "scan_spacing", 0.8))
        spacing_jitter = max(0.0, float(getattr(self.cfg, "scan_spacing_jitter", 0.0)))
        if spacing_jitter > 0.0:
            self.ep_spacing = spacing_base + float(rng.uniform(-spacing_jitter, spacing_jitter))
        else:
            self.ep_spacing = spacing_base
        self.ep_spacing = max(self.ep_spacing, float(getattr(self.cfg, "scan_min_wp_spacing", 0.0)), 1e-3)

        center = env.state.pos.copy()
        raw_points = self._build_lawnmower(center)
        raw_points = self._clamp_path_to_world(raw_points)
        self.path_points = self._compact_path(raw_points)
        self.path_points = self._clamp_path_to_world(self.path_points)
        self.path_points = self._compact_path(self.path_points)
        self._rebuild_path_cache()
        self._init_coverage_grid()

        if bool(getattr(self.cfg, "scan_start_on_path", False)) and len(self.path_points) >= 2:
            p0 = self.path_points[0].copy()
            d01 = self.path_points[1][:2] - self.path_points[0][:2]
            yaw0 = float(np.arctan2(float(d01[1]), float(d01[0]))) if float(np.linalg.norm(d01)) > 1e-6 else 0.0
            env.state.pos = p0.copy()
            env.state.vel = np.zeros(3, dtype=np.float32)
            env.state.roll = 0.0
            env.state.pitch = 0.0
            env.state.yaw = yaw0
            env.state.p = 0.0
            env.state.q = 0.0
            env.state.r = 0.0
            if hasattr(env, "prev_action"):
                env.prev_action = np.zeros(3, dtype=np.float32)
            start_xy_noise = max(0.0, float(getattr(self.cfg, "scan_start_xy_noise", 0.0)))
            start_yaw_noise = max(0.0, float(getattr(self.cfg, "scan_start_yaw_noise", 0.0)))
            if start_xy_noise > 0.0:
                env.state.pos[0] += float(rng.uniform(-start_xy_noise, start_xy_noise))
                env.state.pos[1] += float(rng.uniform(-start_xy_noise, start_xy_noise))
            if start_yaw_noise > 0.0:
                env.state.yaw = wrap_angle(float(env.state.yaw) + float(rng.uniform(-start_yaw_noise, start_yaw_noise)))

        if bool(getattr(self.cfg, "scan_debug_oob", False)):
            x = float(env.state.pos[0])
            y = float(env.state.pos[1])
            bound = float(self.cfg.world_xy_bound)
            reset_oob = bool(abs(x) > bound or abs(y) > bound)
            pmin = np.min(self.path_points[:, :2], axis=0)
            pmax = np.max(self.path_points[:, :2], axis=0)
            area_x_min = float(self.coverage_x_min)
            area_x_max = float(self.coverage_x_min + self.coverage_nx * self.coverage_cell_size)
            area_y_min = float(self.coverage_y_min)
            area_y_max = float(self.coverage_y_min + self.coverage_ny * self.coverage_cell_size)
            print(
                f"[scan oob debug] reset x={x:.3f} y={y:.3f} "
                f"world_xy_bound={bound:.3f} oob={int(reset_oob)} "
                f"path_xy_min=({float(pmin[0]):.3f},{float(pmin[1]):.3f}) "
                f"path_xy_max=({float(pmax[0]):.3f},{float(pmax[1]):.3f}) "
                f"scan_area_x=[{area_x_min:.3f},{area_x_max:.3f}] "
                f"scan_area_y=[{area_y_min:.3f},{area_y_max:.3f}]"
            )

        self.seg_idx = 0
        self.s_on_seg = 0.0
        self.turn_hold_count = 0
        self.done = False
        self.prev_v_xy = np.asarray(env.state.vel[:2], dtype=np.float32).copy()
        _, self.s_on_seg, _ = self._closest_on_segment(env.state.pos, 0)
        self.prev_progress = float(self.cum_lengths[0] + self.s_on_seg)
        lookahead = float(self.ep_lookahead)
        self.target_point, self.target_yaw = self._point_and_tangent_at_progress(self.prev_progress + lookahead)
        self._mark_coverage(env.state.pos[:2], step_idx=0)

    def step(self, env) -> TaskStep:
        if self.done:
            info = {
                "coverage": self._coverage_ratio(),
                "covered_cells": int(self.covered_cells),
                "total_cells": int(self.coverage_total_cells),
                "overlap": self._overlap_ratio(),
                "revisits": int(self.revisit_steps),
                "cov_new_cells": 0,
                "cov_revisit_cells": 0,
                "time_to_95": int(self.time_to_95_step),
                "seg_idx": int(self.seg_idx),
            }
            return TaskStep(reward=0.0, success=True, done=True, info=info)

        pos = env.state.pos.copy()
        n_segs = int(max(1, len(self.seg_lengths)))
        last_seg = n_segs - 1
        hold_turns_only = bool(getattr(self.cfg, "scan_hold_turns_only", False))
        turn_hold_radius = float(getattr(self.cfg, "scan_turn_hold_radius", getattr(self.cfg, "seq_success_radius", 0.8)))
        turn_v_penalty = max(0.0, float(getattr(self.cfg, "scan_turn_v_penalty", 0.0)))
        turn_v_target = max(0.0, float(getattr(self.cfg, "scan_turn_v_target", 0.2)))
        v_xy = float(np.hypot(float(env.state.vel[0]), float(env.state.vel[1])))
        turn_v_penalty_term = 0.0

        seg = int(np.clip(self.seg_idx, 0, last_seg))
        s_raw, _ = self._project_s_raw(pos, seg)
        self.s_on_seg = max(0.0, float(s_raw))

        # Carry overshoot across segments: while passed endpoint, advance segment.
        # On the last segment, never advance past the terminal index.
        for _ in range(len(self.seg_lengths) + 2):
            seg = int(np.clip(self.seg_idx, 0, last_seg))
            self.seg_idx = seg
            seg_len = float(self.seg_lengths[seg])
            if self.s_on_seg < seg_len:
                break
            if seg >= last_seg:
                self.s_on_seg = seg_len
                break

            vertex_idx = seg + 1
            is_turn = bool(self.turn_vertices[vertex_idx])
            if hold_turns_only and is_turn:
                dist_turn = float(np.linalg.norm(pos[:2] - self.path_points[vertex_idx, :2]))
                if dist_turn <= turn_hold_radius:
                    turn_v_penalty_term += turn_v_penalty * max(0.0, v_xy - turn_v_target) ** 2
            self.seg_idx = min(self.seg_idx + 1, last_seg)
            self.s_on_seg -= seg_len

        self.seg_idx = int(np.clip(self.seg_idx, 0, last_seg))
        seg_idx = int(self.seg_idx)
        seg_len = float(self.seg_lengths[seg_idx])
        self.s_on_seg = float(np.clip(self.s_on_seg, 0.0, seg_len))
        _, _, cte = self._closest_on_segment(pos, seg_idx)

        seg_prefix_len = float(self.cum_lengths[seg_idx])
        progress = float(seg_prefix_len + self.s_on_seg)
        progress_delta = float(progress - self.prev_progress)
        self.prev_progress = progress
        cov_new_cells, cov_revisit_cells = self._mark_coverage(pos[:2], step_idx=int(getattr(env, "step_count", 0)))
        coverage_area = self._coverage_ratio()
        overlap = self._overlap_ratio()

        lookahead = float(self.ep_lookahead)
        target_progress = min(self.total_length, progress + max(0.0, lookahead))
        self.target_point, self.target_yaw = self._point_and_tangent_at_progress(target_progress)
        yaw_err = wrap_angle(self.target_yaw - float(env.state.yaw))
        dist_to_lookahead = float(np.linalg.norm(self.target_point - pos))
        yaw_rate = float(env.state.r)
        curr_v_xy = np.asarray(env.state.vel[:2], dtype=np.float32)
        dv_xy = float(np.linalg.norm(curr_v_xy - self.prev_v_xy))
        self.prev_v_xy = curr_v_xy.copy()

        k_prog = float(getattr(self.cfg, "scan_k_prog", self.cfg.seq_k_prog))
        k_ct = float(getattr(self.cfg, "scan_k_ct", 0.5))
        k_yaw = float(getattr(self.cfg, "scan_k_yaw", 0.2))
        k_la = float(getattr(self.cfg, "scan_k_la", 0.0))
        la_clip = max(0.0, float(getattr(self.cfg, "scan_la_clip", 3.0)))
        k_act = max(0.0, float(getattr(self.cfg, "scan_k_act", 0.0)))
        prog_eps = max(0.0, float(getattr(self.cfg, "scan_prog_eps", 0.01)))
        k_stuck = max(0.0, float(getattr(self.cfg, "scan_k_stuck", 0.0)))
        k_cov_gain = max(0.0, float(getattr(self.cfg, "scan_k_cov_gain", 0.0)))
        k_cov_revisit = max(0.0, float(getattr(self.cfg, "scan_k_cov_revisit", 0.0)))
        cov_late_thresh = float(np.clip(getattr(self.cfg, "scan_cov_late_thresh", 1.0), 0.0, 1.0))
        k_cov_gain_late = max(0.0, float(getattr(self.cfg, "scan_k_cov_gain_late", 0.0)))
        k_cov_stall = max(0.0, float(getattr(self.cfg, "scan_k_cov_stall", 0.0)))
        k_yawrate = max(0.0, float(getattr(self.cfg, "scan_k_yawrate", 0.0)))
        k_dvxy = max(0.0, float(getattr(self.cfg, "scan_k_dvxy", 0.0)))
        action_norm = np.asarray(getattr(env, "last_action_norm", np.zeros(4, dtype=np.float32)), dtype=np.float32).reshape(-1)
        ax = float(action_norm[0]) if action_norm.size >= 1 else 0.0
        ay = float(action_norm[1]) if action_norm.size >= 2 else 0.0
        act_penalty_term = k_act * (ax * ax + ay * ay)
        stuck_penalty_term = k_stuck if progress_delta < prog_eps else 0.0
        yawrate_penalty_term = k_yawrate * abs(yaw_rate)
        dvxy_penalty_term = k_dvxy * dv_xy
        cov_gain_late_term = k_cov_gain_late * float(cov_new_cells) if coverage_area >= cov_late_thresh else 0.0
        cov_stall_penalty_term = k_cov_stall if (coverage_area >= cov_late_thresh and cov_new_cells <= 0) else 0.0
        reward = (
            k_prog * progress_delta
            - k_ct * cte
            - k_yaw * abs(yaw_err)
            - k_la * min(dist_to_lookahead, la_clip)
            - turn_v_penalty_term
            - yawrate_penalty_term
            - dvxy_penalty_term
            - act_penalty_term
            - stuck_penalty_term
            + k_cov_gain * float(cov_new_cells)
            + cov_gain_late_term
            - k_cov_revisit * float(cov_revisit_cells)
            - cov_stall_penalty_term
            - float(self.cfg.step_penalty)
        )

        is_last = bool(seg_idx >= (n_segs - 1))
        success_cond = bool(is_last and (progress >= (self.total_length - self.segment_eps)))
        self.done = success_cond
        success = success_cond

        progress_coverage = float(progress / max(self.total_length, 1e-6))
        info = {
            "seg_idx": int(seg_idx),
            "n_segs": int(n_segs),
            "n_segments": int(n_segs),
            "coverage": float(coverage_area),
            "covered_cells": int(self.covered_cells),
            "total_cells": int(self.coverage_total_cells),
            "overlap": float(overlap),
            "revisits": int(self.revisit_steps),
            "cov_new_cells": int(cov_new_cells),
            "cov_revisit_cells": int(cov_revisit_cells),
            "time_to_95": int(self.time_to_95_step),
            "scan_path_len_scale_ep": float(self.ep_path_len_scale),
            "scan_lookahead_ep": float(self.ep_lookahead),
            "scan_spacing_ep": float(self.ep_spacing),
            "path_coverage": float(progress_coverage),
            "path_total_len": float(self.total_length),
            "progress_along_path": progress,
            "cross_track_error": float(cte),
            "yaw_err_tangent": float(yaw_err),
            "yaw_rate": float(yaw_rate),
            "dv_xy": float(dv_xy),
            "dist_to_lookahead": dist_to_lookahead,
            "s_on_seg": float(self.s_on_seg),
            "seg_len": float(seg_len),
            "is_last_seg": bool(is_last),
            "success_cond": bool(success_cond),
            "would_be_success": bool(success_cond),
            "s_local": float(self.s_on_seg),
        }
        return TaskStep(reward=reward, success=success, done=self.done, info=info)

    def get_target(self, env):
        return self.target_point, self.target_yaw
