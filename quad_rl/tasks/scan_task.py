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
        self.boundary_mask = np.zeros((1, 1), dtype=bool)
        self.covered_cells = 0
        self.coverage_total_cells = 1
        self.revisit_steps = 0
        self.time_to_95_step = -1
        self.ep_path_len_scale = 1.0
        self.ep_lookahead = float(getattr(self.cfg, "scan_lookahead", 1.0))
        self.ep_spacing = float(getattr(self.cfg, "scan_spacing", 0.8))
        self.prev_v_xy = np.zeros((2,), dtype=np.float32)
        self.visit_counts = np.zeros((1, 1), dtype=np.int32)
        self.coverage2x_cells = 0
        self.prev_cov_ratio = 0.0
        self.prev_overlap2x_ratio = 0.0
        self.ep_total_new_coverage = 0.0
        self.ep_total_overlap_added = 0.0
        self.ep_speed_cmd_sum = 0.0
        self.ep_speed_cmd_steps = 0
        self.ep_total_overshoot = 0.0

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
        self.visit_counts = np.zeros((self.coverage_nx, self.coverage_ny), dtype=np.int32)
        edge_cells = 2
        ix = np.arange(self.coverage_nx, dtype=np.int32)[:, None]
        iy = np.arange(self.coverage_ny, dtype=np.int32)[None, :]
        self.boundary_mask = (
            (ix < edge_cells)
            | (ix >= (self.coverage_nx - edge_cells))
            | (iy < edge_cells)
            | (iy >= (self.coverage_ny - edge_cells))
        )
        self.covered_cells = 0
        self.coverage2x_cells = 0
        self.coverage_total_cells = int(self.coverage_nx * self.coverage_ny)
        self.revisit_steps = 0
        self.time_to_95_step = -1

    def _maybe_apply_finish_curriculum(self, env, rng: np.random.Generator) -> bool:
        if bool(getattr(env, "is_eval", False)):
            return False
        if not bool(getattr(self.cfg, "scan_finishcurr_enable", False)):
            return False
        prob = float(np.clip(getattr(self.cfg, "scan_finishcurr_prob", 0.0), 0.0, 1.0))
        if prob <= 0.0 or float(rng.uniform(0.0, 1.0)) > prob:
            return False

        cov_range = getattr(self.cfg, "scan_finishcurr_cov_range", (0.88, 0.92))
        try:
            cov_lo = float(cov_range[0])
            cov_hi = float(cov_range[1])
        except Exception:
            cov_lo, cov_hi = 0.88, 0.92
        if cov_hi < cov_lo:
            cov_lo, cov_hi = cov_hi, cov_lo
        cov_lo = float(np.clip(cov_lo, 0.0, 1.0))
        cov_hi = float(np.clip(cov_hi, 0.0, 1.0))
        target_cov = float(rng.uniform(cov_lo, cov_hi)) if abs(cov_hi - cov_lo) > 1e-9 else cov_lo

        total = int(self.coverage_total_cells)
        if total <= 0:
            return False
        pocket_frac = float(np.clip(getattr(self.cfg, "scan_finishcurr_pocket_frac", 0.08), 0.0, 1.0))
        boundary_bias = float(np.clip(getattr(self.cfg, "scan_finishcurr_boundary_bias", 0.7), 0.0, 1.0))

        n_uncovered_target = int(round((1.0 - target_cov) * float(total)))
        n_pocket = int(round(pocket_frac * float(total)))
        n_uncovered = int(np.clip(max(n_uncovered_target, n_pocket), 1, max(1, total - 1)))

        boundary_flat = np.flatnonzero(self.boundary_mask.reshape(-1))
        interior_flat = np.flatnonzero(~self.boundary_mask.reshape(-1))
        n_boundary = int(round(float(n_uncovered) * boundary_bias))
        n_boundary = min(n_boundary, int(boundary_flat.size))
        n_interior = n_uncovered - n_boundary
        if n_interior > int(interior_flat.size):
            spill = n_interior - int(interior_flat.size)
            n_interior = int(interior_flat.size)
            n_boundary = min(int(boundary_flat.size), n_boundary + spill)

        uncovered_parts = []
        if n_boundary > 0:
            uncovered_parts.append(rng.choice(boundary_flat, size=n_boundary, replace=False))
        if n_interior > 0:
            uncovered_parts.append(rng.choice(interior_flat, size=n_interior, replace=False))
        if len(uncovered_parts) == 0:
            return False

        uncovered_idx = np.concatenate(uncovered_parts).astype(np.int64, copy=False)
        flat = self.covered_mask.reshape(-1)
        flat[:] = True
        flat[uncovered_idx] = False
        self.covered_cells = int(np.count_nonzero(flat))
        self.revisit_steps = 0
        self.time_to_95_step = 0 if self._coverage_ratio() >= 0.95 else -1
        return True

    def _coverage_ratio(self) -> float:
        return float(self.covered_cells / max(self.coverage_total_cells, 1))

    def _overlap_ratio(self) -> float:
        return float(self.revisit_steps / max(self.covered_cells, 1))

    def _overlap2x_ratio(self) -> float:
        return float(self.coverage2x_cells / max(self.coverage_total_cells, 1))

    def get_obs_aug_features(self, env) -> np.ndarray:
        patch_size = int(getattr(self.cfg, "scan_obs_patch_size", 5))
        patch_size = max(1, patch_size)
        if patch_size % 2 == 0:
            patch_size += 1
        half = patch_size // 2

        x = float(env.state.pos[0])
        y = float(env.state.pos[1])
        cs = float(max(self.coverage_cell_size, 1e-6))
        ix_center = int(np.floor((x - self.coverage_x_min) / cs))
        iy_center = int(np.floor((y - self.coverage_y_min) / cs))

        patch = np.ones((patch_size, patch_size), dtype=np.float32)
        for dx in range(-half, half + 1):
            gx = ix_center + dx
            if gx < 0 or gx >= self.coverage_nx:
                continue
            for dy in range(-half, half + 1):
                gy = iy_center + dy
                if gy < 0 or gy >= self.coverage_ny:
                    continue
                patch[dx + half, dy + half] = 1.0 if bool(self.covered_mask[gx, gy]) else 0.0

        feats = [patch.reshape(-1)]
        if bool(getattr(self.cfg, "scan_obs_boundary_feat", True)):
            x_max = self.coverage_x_min + float(self.coverage_nx) * cs
            y_max = self.coverage_y_min + float(self.coverage_ny) * cs
            d_x = min(x - self.coverage_x_min, x_max - x)
            d_y = min(y - self.coverage_y_min, y_max - y)
            span_x = max(x_max - self.coverage_x_min, 1e-6)
            span_y = max(y_max - self.coverage_y_min, 1e-6)
            d_x_norm = float(np.clip(d_x / span_x, 0.0, 1.0))
            d_y_norm = float(np.clip(d_y / span_y, 0.0, 1.0))
            feats.append(np.asarray([d_x_norm, d_y_norm], dtype=np.float32))

        if bool(getattr(self.cfg, "scan_obs_global_coverage_enable", False)):
            g = int(getattr(self.cfg, "scan_obs_global_size", 8))
            g = max(1, g)
            cov = np.asarray(self.covered_mask, dtype=np.float32)
            global_map = np.zeros((g, g), dtype=np.float32)
            for gx in range(g):
                x0 = int(np.floor((gx * self.coverage_nx) / g))
                x1 = int(np.floor(((gx + 1) * self.coverage_nx) / g))
                x1 = max(x1, x0 + 1)
                x1 = min(x1, self.coverage_nx)
                for gy in range(g):
                    y0 = int(np.floor((gy * self.coverage_ny) / g))
                    y1 = int(np.floor(((gy + 1) * self.coverage_ny) / g))
                    y1 = max(y1, y0 + 1)
                    y1 = min(y1, self.coverage_ny)
                    block = cov[x0:x1, y0:y1]
                    global_map[gx, gy] = float(np.mean(block)) if block.size else 0.0
            feats.append(global_map.reshape(-1))

        return np.concatenate(feats, axis=0).astype(np.float32, copy=False)

    def _in_scan_oob_recovery(self, env) -> bool:
        if int(getattr(env, "scan_oob_consec", 0)) > 0:
            return True

        pos = np.asarray(env.state.pos, dtype=np.float32)
        bound = float(getattr(self.cfg, "world_xy_bound", 0.0))
        if abs(float(pos[0])) > bound or abs(float(pos[1])) > bound:
            return True

        z = float(pos[2])
        scan_z_min = getattr(self.cfg, "scan_z_min", None)
        scan_z_max = getattr(self.cfg, "scan_z_max", None)
        z_low_limit = float(scan_z_min) - 0.2 if scan_z_min is not None else float(self.cfg.world_z_min) - 0.05
        z_high_limit = float(scan_z_max) + 0.2 if scan_z_max is not None else float(self.cfg.world_z_max)
        return bool(z < z_low_limit or z > z_high_limit)

    def _teacher_action_ref_norm(self, env, progress: float) -> np.ndarray:
        # Scripted lawnmower teacher: pursue lookahead point and align yaw to tangent.
        teacher_progress = min(self.total_length, progress + max(0.0, float(self.ep_lookahead)))
        target_point, target_yaw = self._point_and_tangent_at_progress(teacher_progress)

        pos_xy = np.asarray(env.state.pos[:2], dtype=np.float32)
        to_target = np.asarray(target_point[:2], dtype=np.float32) - pos_xy

        v_xy_limit = float(getattr(self.cfg, "scan_v_xy_max", getattr(self.cfg, "v_xy_max", 1.0)))
        if hasattr(env, "_xy_action_limit"):
            try:
                v_xy_limit = float(env._xy_action_limit())
            except Exception:
                pass
        v_xy_limit = max(v_xy_limit, 1e-6)

        v_ref = 1.5 * to_target.astype(np.float64)
        v_spd = float(np.linalg.norm(v_ref))
        if v_spd > v_xy_limit:
            v_ref *= float(v_xy_limit / max(v_spd, 1e-9))

        yaw_limit = float(getattr(self.cfg, "scan_yaw_rate_max", getattr(self.cfg, "yaw_rate_max", 1.0)))
        if hasattr(env, "_yaw_rate_action_limit"):
            try:
                yaw_limit = float(env._yaw_rate_action_limit())
            except Exception:
                pass
        yaw_limit = max(yaw_limit, 1e-6)
        yaw_err = wrap_angle(float(target_yaw) - float(env.state.yaw))
        yaw_rate_ref = float(np.clip(2.0 * yaw_err, -yaw_limit, yaw_limit))

        ref = np.array(
            [
                float(v_ref[0]) / v_xy_limit,
                float(v_ref[1]) / v_xy_limit,
                yaw_rate_ref / yaw_limit,
            ],
            dtype=np.float32,
        )
        return np.clip(ref, -1.0, 1.0)

    def _mark_coverage(self, pos_xy: np.ndarray, step_idx: int) -> tuple[int, int, int, int]:
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
        new_boundary_hits = 0
        overlap2x_new_hits = 0
        for ix in range(ix0, ix1 + 1):
            cx = self.coverage_x_min + (float(ix) + 0.5) * cs
            dx = cx - x
            for iy in range(iy0, iy1 + 1):
                cy = self.coverage_y_min + (float(iy) + 0.5) * cs
                dy = cy - y
                if (dx * dx + dy * dy) > r2:
                    continue
                prev_count = int(self.visit_counts[ix, iy])
                self.visit_counts[ix, iy] = prev_count + 1
                if prev_count <= 0:
                    self.covered_mask[ix, iy] = True
                    new_hits += 1
                    if bool(self.boundary_mask[ix, iy]):
                        new_boundary_hits += 1
                else:
                    revisit_hits += 1
                if prev_count == 1:
                    overlap2x_new_hits += 1

        if new_hits > 0:
            self.covered_cells += int(new_hits)
        elif step_idx > 0 and revisit_hits > 0:
            self.revisit_steps += 1
        if overlap2x_new_hits > 0:
            self.coverage2x_cells += int(overlap2x_new_hits)

        if self.time_to_95_step < 0 and self._coverage_ratio() >= 0.95:
            self.time_to_95_step = int(step_idx)
        return int(new_hits), int(revisit_hits), int(new_boundary_hits), int(overlap2x_new_hits)

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
        self._maybe_apply_finish_curriculum(env, rng)

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
        self.prev_cov_ratio = float(self._coverage_ratio())
        self.prev_overlap2x_ratio = float(self._overlap2x_ratio())
        self.ep_total_new_coverage = 0.0
        self.ep_total_overlap_added = 0.0
        self.ep_speed_cmd_sum = 0.0
        self.ep_speed_cmd_steps = 0
        self.ep_total_overshoot = 0.0

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
                "cov_boundary_new_cells": 0,
                "cov_overlap2x_new_cells": 0,
                "time_to_95": int(self.time_to_95_step),
                "total_new_coverage": float(self.ep_total_new_coverage),
                "total_overlap_added": float(self.ep_total_overlap_added),
                "mean_speed_cmd": float(self.ep_speed_cmd_sum / max(1, self.ep_speed_cmd_steps)),
                "duration_s": float(getattr(env, "step_count", 0) * float(getattr(self.cfg, "dt", 0.1))),
                "total_overshoot": float(self.ep_total_overshoot),
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
        s_raw, seg_len_raw = self._project_s_raw(pos, seg)
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
        cov_new_cells, cov_revisit_cells, cov_boundary_new_cells, cov_overlap2x_new_cells = self._mark_coverage(
            pos[:2], step_idx=int(getattr(env, "step_count", 0))
        )
        coverage_area = self._coverage_ratio()
        overlap = self._overlap_ratio()

        lookahead = float(self.ep_lookahead)
        target_progress = min(self.total_length, progress + max(0.0, lookahead))
        self.target_point, self.target_yaw = self._point_and_tangent_at_progress(target_progress)
        yaw_err = wrap_angle(self.target_yaw - float(env.state.yaw))
        dist_to_lookahead = float(np.linalg.norm(self.target_point - pos))
        yaw_rate = float(env.state.r)
        yaw_world = float(env.state.yaw)
        vx_world = float(env.state.vel[0])
        vy_world = float(env.state.vel[1])
        # World-frame XY velocity projected to body-frame lateral component.
        v_y_body = float((-np.sin(yaw_world) * vx_world) + (np.cos(yaw_world) * vy_world))
        curr_v_xy = np.asarray(env.state.vel[:2], dtype=np.float32)
        dv_xy = float(np.linalg.norm(curr_v_xy - self.prev_v_xy))
        self.prev_v_xy = curr_v_xy.copy()

        k_cov = float(getattr(self.cfg, "scan_k_cov", 10.0))
        k_ov = float(getattr(self.cfg, "scan_k_ov", 6.0))
        k_t = float(getattr(self.cfg, "scan_k_t", 0.2))
        k_speed = float(getattr(self.cfg, "scan_k_speed", 0.6))
        k_overshoot = float(getattr(self.cfg, "scan_k_overshoot", 2.0))
        dt = float(getattr(self.cfg, "dt", 0.1))

        coverage_footprint = float(coverage_area)
        overlap_2x = float(self._overlap2x_ratio())
        d_cov = float(coverage_footprint - float(self.prev_cov_ratio))
        d_ov = float(overlap_2x - float(self.prev_overlap2x_ratio))

        excess_distance = float(max(0.0, float(s_raw) - float(seg_len_raw)))
        speed_term = float(max(0.0, progress_delta) / max(dt, 1e-6))

        r_cov = float(k_cov * d_cov)
        r_ov = float(-k_ov * d_ov)
        r_time = float(-k_t * dt)
        r_speed = float(k_speed * speed_term)
        r_overshoot = float(-k_overshoot * excess_distance)
        reward = float(r_cov + r_ov + r_time + r_speed + r_overshoot)

        self.prev_cov_ratio = float(coverage_footprint)
        self.prev_overlap2x_ratio = float(overlap_2x)
        self.ep_total_new_coverage += float(max(0.0, d_cov))
        self.ep_total_overlap_added += float(max(0.0, d_ov))
        self.ep_total_overshoot += float(excess_distance)
        speed_cmd = float(np.hypot(float(env.prev_action[0]), float(env.prev_action[1])))
        self.ep_speed_cmd_sum += float(speed_cmd)
        self.ep_speed_cmd_steps += 1


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
            "cov_boundary_new_cells": int(cov_boundary_new_cells),
            "cov_overlap2x_new_cells": int(cov_overlap2x_new_cells),
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
            "v_y_body": float(v_y_body),
            "dv_xy": float(dv_xy),
            "dist_to_lookahead": dist_to_lookahead,
            "s_on_seg": float(self.s_on_seg),
            "seg_len": float(seg_len),
            "is_last_seg": bool(is_last),
            "success_cond": bool(success_cond),
            "would_be_success": bool(success_cond),
            "s_local": float(self.s_on_seg),
            "coverage_footprint": float(coverage_footprint),
            "overlap_2x": float(overlap_2x),
            "r_cov": float(r_cov),
            "r_ov": float(r_ov),
            "r_time": float(r_time),
            "r_speed": float(r_speed),
            "r_overshoot": float(r_overshoot),
            "excess_distance": float(excess_distance),
            "total_new_coverage": float(self.ep_total_new_coverage),
            "total_overlap_added": float(self.ep_total_overlap_added),
            "mean_speed_cmd": float(self.ep_speed_cmd_sum / max(1, self.ep_speed_cmd_steps)),
            "duration_s": float(getattr(env, "step_count", 0) * float(getattr(self.cfg, "dt", 0.1))),
            "total_overshoot": float(self.ep_total_overshoot),
        }
        return TaskStep(reward=reward, success=success, done=self.done, info=info)

    def get_target(self, env):
        return self.target_point, self.target_yaw
