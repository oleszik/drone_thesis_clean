from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from quad_rl.curriculum.presets import get_preset, list_presets
from quad_rl.envs.quad15d_env import Quad15DEnv
from quad_rl.tasks import build_task
from quad_rl.utils.config_overrides import apply_overrides, parse_override_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate scan coverage feasibility with scripted lawnmower tracking.")
    parser.add_argument("--preset", type=str, default="A2_ABL_B125_T85", choices=list_presets())
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=456)
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument(
        "--geom-ds",
        type=float,
        default=0.05,
        help="Arc-length step (m) for dense geometry ceiling stamping.",
    )
    parser.add_argument(
        "--cfg-override",
        action="append",
        default=[],
        help="Override preset fields with key=value (repeatable).",
    )
    return parser.parse_args()


def _mark_disk(
    covered_mask: np.ndarray,
    x: float,
    y: float,
    radius: float,
    cell_size: float,
    x_min: float,
    y_min: float,
) -> tuple[int, int]:
    nx, ny = covered_mask.shape
    r = float(max(radius, 0.0))
    cs = float(max(cell_size, 1e-6))
    r2 = r * r

    ix0 = int(np.floor((x - r - x_min) / cs))
    ix1 = int(np.floor((x + r - x_min) / cs))
    iy0 = int(np.floor((y - r - y_min) / cs))
    iy1 = int(np.floor((y + r - y_min) / cs))
    ix0 = int(np.clip(ix0, 0, nx - 1))
    ix1 = int(np.clip(ix1, 0, nx - 1))
    iy0 = int(np.clip(iy0, 0, ny - 1))
    iy1 = int(np.clip(iy1, 0, ny - 1))

    new_hits = 0
    revisit_hits = 0
    for ix in range(ix0, ix1 + 1):
        cx = x_min + (float(ix) + 0.5) * cs
        dx = cx - x
        for iy in range(iy0, iy1 + 1):
            cy = y_min + (float(iy) + 0.5) * cs
            dy = cy - y
            if (dx * dx + dy * dy) > r2:
                continue
            if bool(covered_mask[ix, iy]):
                revisit_hits += 1
            else:
                covered_mask[ix, iy] = True
                new_hits += 1
    return int(new_hits), int(revisit_hits)


def _simulate_progress_stamping(
    task,
    ds: float,
    max_steps: int | None,
) -> dict[str, float]:
    nx = int(task.coverage_nx)
    ny = int(task.coverage_ny)
    total_cells = int(max(1, nx * ny))
    cell_size = float(task.coverage_cell_size)
    radius = float(task.coverage_radius)
    x_min = float(task.coverage_x_min)
    y_min = float(task.coverage_y_min)
    total_len = float(task.total_length)
    step_limit = int(max_steps) if max_steps is not None else -1
    step_size = float(max(ds, 1e-6))

    covered_mask = np.zeros((nx, ny), dtype=bool)
    covered_cells = 0
    revisit_steps = 0
    time_to_95 = -1
    progress = 0.0
    step_idx = 0

    p0, _ = task._point_and_tangent_at_progress(0.0)
    new_hits, revisit_hits = _mark_disk(covered_mask, float(p0[0]), float(p0[1]), radius, cell_size, x_min, y_min)
    covered_cells += int(new_hits)
    if new_hits <= 0 and revisit_hits > 0:
        revisit_steps += 1
    if time_to_95 < 0 and (covered_cells / total_cells) >= 0.95:
        time_to_95 = 0

    while progress < total_len:
        if step_limit >= 0 and step_idx >= step_limit:
            break
        progress = min(total_len, progress + step_size)
        step_idx += 1
        p, _ = task._point_and_tangent_at_progress(progress)
        new_hits, revisit_hits = _mark_disk(covered_mask, float(p[0]), float(p[1]), radius, cell_size, x_min, y_min)
        if new_hits > 0:
            covered_cells += int(new_hits)
        elif revisit_hits > 0:
            revisit_steps += 1
        if time_to_95 < 0 and (covered_cells / total_cells) >= 0.95:
            time_to_95 = int(step_idx)

    coverage = float(covered_cells / total_cells)
    overlap = float(revisit_steps / max(covered_cells, 1))
    reached_end = bool(progress >= (total_len - 1e-6))
    return {
        "coverage": coverage,
        "overlap": overlap,
        "covered_cells": int(covered_cells),
        "total_cells": int(total_cells),
        "time_to_95": int(time_to_95),
        "steps_used": int(step_idx),
        "reached_end": bool(reached_end),
    }


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _min(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.min(np.asarray(values, dtype=np.float64)))


def main() -> None:
    args = parse_args()
    cfg = get_preset(args.preset)
    cfg_overrides = parse_override_pairs(args.cfg_override)
    applied_overrides = apply_overrides(cfg, cfg_overrides) if cfg_overrides else {}
    scan_max_steps = int(getattr(cfg, "scan_max_steps", cfg.max_steps))
    scan_v_xy_max = getattr(cfg, "scan_v_xy_max", None)
    if scan_v_xy_max is None:
        scan_v_xy_max = cfg.v_xy_max
    speed = float(scan_v_xy_max)
    dt = float(cfg.dt)
    ds_time_limited = speed * dt
    ds_geom = float(max(args.geom_ds, 1e-6))

    env = Quad15DEnv(
        task=build_task("scan", cfg),
        cfg=cfg,
        max_steps=scan_max_steps,
        seed=int(args.seed),
    )

    geom_coverages: list[float] = []
    geom_overlaps: list[float] = []
    geom_time95: list[float] = []
    time_coverages: list[float] = []
    time_overlaps: list[float] = []
    time_time95: list[float] = []
    path_total_lens: list[float] = []
    path_req_steps: list[float] = []
    req_feasible_flags: list[float] = []
    geom_feasible_95: list[float] = []
    time_feasible_95: list[float] = []

    for ep in range(int(args.episodes)):
        env.reset(seed=int(args.seed) + ep)
        task = env.task
        total_len = float(getattr(task, "total_length", 0.0))
        required_steps = total_len / max(ds_time_limited, 1e-6)

        geom = _simulate_progress_stamping(task, ds=ds_geom, max_steps=None)
        time_limited = _simulate_progress_stamping(task, ds=ds_time_limited, max_steps=scan_max_steps)

        geom_coverages.append(float(geom["coverage"]))
        geom_overlaps.append(float(geom["overlap"]))
        geom_time95.append(float(geom["time_to_95"] if geom["time_to_95"] >= 0 else scan_max_steps + 1))
        time_coverages.append(float(time_limited["coverage"]))
        time_overlaps.append(float(time_limited["overlap"]))
        time_time95.append(float(time_limited["time_to_95"] if time_limited["time_to_95"] >= 0 else scan_max_steps + 1))
        path_total_lens.append(total_len)
        path_req_steps.append(required_steps)
        req_feasible_flags.append(1.0 if required_steps <= scan_max_steps else 0.0)
        geom_feasible_95.append(1.0 if geom["coverage"] >= 0.95 else 0.0)
        time_feasible_95.append(1.0 if time_limited["coverage"] >= 0.95 else 0.0)

    summary = {
        "preset": args.preset,
        "cfg_overrides": applied_overrides,
        "episodes": int(args.episodes),
        "seed_base": int(args.seed),
        "scan_max_steps": int(scan_max_steps),
        "scan_v_xy_max": float(speed),
        "dt": float(dt),
        "ds_time_limited": float(ds_time_limited),
        "ds_geom": float(ds_geom),
        "path_total_len_mean": _mean(path_total_lens),
        "path_total_len_max": float(max(path_total_lens) if path_total_lens else 0.0),
        "required_steps_mean": _mean(path_req_steps),
        "required_steps_max": float(max(path_req_steps) if path_req_steps else 0.0),
        "required_steps_feasible_rate": _mean(req_feasible_flags),
        "geom_coverage_mean": _mean(geom_coverages),
        "geom_coverage_min": _min(geom_coverages),
        "geom_overlap_mean": _mean(geom_overlaps),
        "geom_time_to_95_mean": _mean(geom_time95),
        "geom_coverage_ge_95_rate": _mean(geom_feasible_95),
        "time_limited_coverage_mean": _mean(time_coverages),
        "time_limited_coverage_min": _min(time_coverages),
        "time_limited_overlap_mean": _mean(time_overlaps),
        "time_limited_time_to_95_mean": _mean(time_time95),
        "time_limited_coverage_ge_95_rate": _mean(time_feasible_95),
        "is_95_feasible_geom": bool(_mean(geom_feasible_95) > 0.99),
        "is_95_feasible_time_limited": bool(_mean(time_feasible_95) > 0.99),
    }

    print("[lawnmower_baseline] summary:", json.dumps(summary, indent=2))
    if args.json_out.strip():
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[lawnmower_baseline] wrote summary: {out_path}")


if __name__ == "__main__":
    main()
