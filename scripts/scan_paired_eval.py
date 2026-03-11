from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from quad_rl.curriculum.presets import get_preset, list_presets
from quad_rl.envs.quad15d_env import Quad15DEnv
from quad_rl.tasks import build_task
from quad_rl.tasks.base_task import wrap_angle
from quad_rl.utils.config_overrides import apply_overrides, parse_override_pairs
from quad_rl.utils.paths import normalize_model_path
from quad_rl.utils.scan_scale_profile import (
    apply_scan_obs_profile,
    assert_scan_obs_profile,
    effective_scan_max_steps,
    get_scan_path_scale_upper,
    resolve_scan_production_model_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired scan eval: RL policy vs scripted lawnmower controller.")
    parser.add_argument("--model", type=str, default="auto")
    parser.add_argument("--preset", type=str, default="A2_FINISH_TIMELATE_BOUNDARY", choices=list_presets())
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=456)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument(
        "--cfg-override",
        action="append",
        default=[],
        help="Override preset fields with key=value (repeatable).",
    )
    return parser.parse_args()


def _max_steps_for_task(task_name: str, cfg) -> int:
    key = task_name.strip().lower()
    if key == "sequence":
        return int(cfg.seq_max_steps)
    if key == "scan":
        return int(effective_scan_max_steps(cfg))
    return int(cfg.max_steps)


def _make_scan_env(cfg, seed: int) -> Quad15DEnv:
    task = build_task("scan", cfg)
    return Quad15DEnv(task=task, cfg=cfg, max_steps=_max_steps_for_task("scan", cfg), seed=seed, is_eval=True)


def _teacher_action(env: Quad15DEnv) -> np.ndarray:
    task = env.task
    if hasattr(task, "_teacher_action_ref_norm"):
        progress = float(getattr(task, "prev_progress", 0.0))
        ref = np.asarray(task._teacher_action_ref_norm(env, progress), dtype=np.float32).reshape(-1)
        action = np.zeros((4,), dtype=np.float32)
        if ref.size >= 1:
            action[0] = float(ref[0])
        if ref.size >= 2:
            action[1] = float(ref[1])
        if ref.size >= 3:
            action[3] = float(ref[2])
        return np.clip(action, -1.0, 1.0)
    return np.zeros((4,), dtype=np.float32)


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


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


def _simulate_scripted_lawnmower_metrics(task, cfg, seed: int) -> dict:
    nx = int(task.coverage_nx)
    ny = int(task.coverage_ny)
    covered = np.zeros((nx, ny), dtype=bool)
    boundary_mask = np.asarray(task.boundary_mask, dtype=bool)
    interior_mask = ~boundary_mask

    cell_size = float(task.coverage_cell_size)
    radius = float(task.coverage_radius)
    x_min = float(task.coverage_x_min)
    y_min = float(task.coverage_y_min)
    total_cells = int(max(1, nx * ny))
    total_len = float(task.total_length)
    dt = float(cfg.dt)
    speed = float(getattr(cfg, "scan_v_xy_max", cfg.v_xy_max))
    ds = speed * dt
    max_steps = int(effective_scan_max_steps(cfg))

    progress = 0.0
    steps = 0
    covered_cells = 0
    revisit_steps = 0
    time_to_95 = -1
    yawrates: list[float] = []
    prev_yaw = None

    p0, y0 = task._point_and_tangent_at_progress(0.0)
    new_hits, revisit_hits = _mark_disk(covered, float(p0[0]), float(p0[1]), radius, cell_size, x_min, y_min)
    covered_cells += int(new_hits)
    if new_hits <= 0 and revisit_hits > 0:
        revisit_steps += 1
    if (covered_cells / total_cells) >= 0.95:
        time_to_95 = 0
    prev_yaw = float(y0)

    while progress < total_len and steps < max_steps:
        progress = min(total_len, progress + max(ds, 1e-6))
        steps += 1
        p, yaw = task._point_and_tangent_at_progress(progress)
        new_hits, revisit_hits = _mark_disk(covered, float(p[0]), float(p[1]), radius, cell_size, x_min, y_min)
        if new_hits > 0:
            covered_cells += int(new_hits)
        elif revisit_hits > 0:
            revisit_steps += 1
        if time_to_95 < 0 and (covered_cells / total_cells) >= 0.95:
            time_to_95 = int(steps)

        curr_yaw = float(yaw)
        if prev_yaw is not None:
            yawrates.append(abs(wrap_angle(curr_yaw - prev_yaw)) / max(dt, 1e-6))
        prev_yaw = curr_yaw

    if time_to_95 < 0:
        time_to_95 = int(steps + 1)

    final_coverage = float(covered_cells / total_cells)
    boundary_total = int(np.count_nonzero(boundary_mask))
    interior_total = int(np.count_nonzero(interior_mask))
    boundary_covered = int(np.count_nonzero(covered & boundary_mask))
    interior_covered = int(np.count_nonzero(covered & interior_mask))
    boundary_covered_frac = float(boundary_covered / max(boundary_total, 1))
    interior_covered_frac = float(interior_covered / max(interior_total, 1))
    turn_vertices = np.asarray(getattr(task, "turn_vertices", np.zeros((0,), dtype=bool)), dtype=bool)
    turn_count = int(np.count_nonzero(turn_vertices[1:-1])) if turn_vertices.size >= 3 else 0

    return {
        "seed": int(seed),
        "mode": "scripted_lawnmower",
        "success": bool(progress >= (total_len - 1e-6)),
        "crash": False,
        "steps": int(steps),
        "final_coverage": float(final_coverage),
        "cov95_hit": bool(final_coverage >= 0.95),
        "time_to_95": int(time_to_95),
        "mean_speed": float(speed),
        "mean_abs_yawrate": _safe_mean(yawrates),
        "turn_count": int(turn_count),
        "oob_recovery_steps": 0,
        "boundary_covered_frac": float(boundary_covered_frac),
        "interior_covered_frac": float(interior_covered_frac),
        "revisit_steps": int(revisit_steps),
    }


def _run_episode(seed: int, cfg, model: PPO | None, mode: str) -> dict:
    env = _make_scan_env(cfg, seed=seed)
    obs, _ = env.reset(seed=seed)
    if mode != "rl":
        result = _simulate_scripted_lawnmower_metrics(env.task, cfg, seed=seed)
        env.close()
        return result

    done = False
    truncated = False
    steps = 0
    last_info: dict = {}
    speeds: list[float] = []
    yaw_abs: list[float] = []
    turn_count = 0
    oob_recovery_steps = 0
    prev_seg_idx = int(getattr(env.task, "seg_idx", 0))

    while not done and not truncated:
        if mode == "rl":
            assert model is not None
            action, _ = model.predict(obs, deterministic=True)
            action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        else:
            action_arr = _teacher_action(env)

        obs, _, done, truncated, info = env.step(action_arr)
        last_info = info
        steps += 1

        v_xy = float(np.hypot(float(env.state.vel[0]), float(env.state.vel[1])))
        speeds.append(v_xy)
        yaw_abs.append(abs(float(env.state.r)))

        seg_idx = int(info.get("seg_idx", prev_seg_idx))
        if seg_idx > prev_seg_idx and hasattr(env.task, "turn_vertices"):
            turns = np.asarray(env.task.turn_vertices, dtype=bool)
            for v in range(prev_seg_idx + 1, seg_idx + 1):
                if 0 <= v < turns.size and bool(turns[v]):
                    turn_count += 1
        prev_seg_idx = seg_idx

        if int(info.get("scan_oob_consec", 0)) > 0 or bool(info.get("scan_out_of_bounds_raw", False)):
            oob_recovery_steps += 1

    task = env.task
    covered_mask = np.asarray(getattr(task, "covered_mask", np.zeros((1, 1), dtype=bool)), dtype=bool)
    boundary_mask = np.asarray(getattr(task, "boundary_mask", np.zeros_like(covered_mask, dtype=bool)), dtype=bool)
    interior_mask = ~boundary_mask
    boundary_total = int(np.count_nonzero(boundary_mask))
    interior_total = int(np.count_nonzero(interior_mask))
    boundary_covered = int(np.count_nonzero(covered_mask & boundary_mask))
    interior_covered = int(np.count_nonzero(covered_mask & interior_mask))
    boundary_covered_frac = float(boundary_covered / max(boundary_total, 1))
    interior_covered_frac = float(interior_covered / max(interior_total, 1))

    final_coverage = float(last_info.get("coverage", 0.0))
    time_to_95 = int(last_info.get("time_to_95", -1))
    cov95_hit = bool(final_coverage >= 0.95 or (time_to_95 >= 0 and time_to_95 <= steps))
    if time_to_95 < 0:
        time_to_95 = int(steps + 1)

    result = {
        "seed": int(seed),
        "mode": mode,
        "success": bool(last_info.get("success", False)),
        "crash": bool(last_info.get("crash", False)),
        "steps": int(steps),
        "final_coverage": float(final_coverage),
        "cov95_hit": bool(cov95_hit),
        "time_to_95": int(time_to_95),
        "mean_speed": _safe_mean(speeds),
        "mean_abs_yawrate": _safe_mean(yaw_abs),
        "turn_count": int(turn_count),
        "oob_recovery_steps": int(oob_recovery_steps),
        "boundary_covered_frac": float(boundary_covered_frac),
        "interior_covered_frac": float(interior_covered_frac),
    }
    env.close()
    return result


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    arr_cov = np.asarray([float(r["final_coverage"]) for r in rows], dtype=np.float64)
    arr_cov95 = np.asarray([1.0 if bool(r["cov95_hit"]) else 0.0 for r in rows], dtype=np.float64)
    arr_t95 = np.asarray([float(r["time_to_95"]) for r in rows], dtype=np.float64)
    arr_steps = np.asarray([float(r["steps"]) for r in rows], dtype=np.float64)
    arr_speed = np.asarray([float(r["mean_speed"]) for r in rows], dtype=np.float64)
    arr_yaw = np.asarray([float(r["mean_abs_yawrate"]) for r in rows], dtype=np.float64)
    arr_turns = np.asarray([float(r["turn_count"]) for r in rows], dtype=np.float64)
    arr_oob_rec = np.asarray([float(r["oob_recovery_steps"]) for r in rows], dtype=np.float64)
    arr_bnd = np.asarray([float(r["boundary_covered_frac"]) for r in rows], dtype=np.float64)
    arr_int = np.asarray([float(r["interior_covered_frac"]) for r in rows], dtype=np.float64)
    arr_success = np.asarray([1.0 if bool(r["success"]) else 0.0 for r in rows], dtype=np.float64)
    arr_crash = np.asarray([1.0 if bool(r["crash"]) else 0.0 for r in rows], dtype=np.float64)
    return {
        "episodes": int(len(rows)),
        "success_rate": float(np.mean(arr_success)),
        "crash_rate": float(np.mean(arr_crash)),
        "coverage_mean": float(np.mean(arr_cov)),
        "coverage_min": float(np.min(arr_cov)),
        "coverage_ge_095_rate": float(np.mean(arr_cov95)),
        "time_to_95_mean": float(np.mean(arr_t95)),
        "steps_mean": float(np.mean(arr_steps)),
        "mean_speed": float(np.mean(arr_speed)),
        "mean_abs_yawrate": float(np.mean(arr_yaw)),
        "turn_count_mean": float(np.mean(arr_turns)),
        "oob_recovery_steps_mean": float(np.mean(arr_oob_rec)),
        "boundary_covered_frac_mean": float(np.mean(arr_bnd)),
        "interior_covered_frac_mean": float(np.mean(arr_int)),
    }


def main() -> None:
    args = parse_args()
    cfg = get_preset(args.preset)
    cfg_overrides = parse_override_pairs(args.cfg_override)
    applied_overrides = apply_overrides(cfg, cfg_overrides) if cfg_overrides else {}
    raw_model = (args.model or "").strip()
    auto_tokens = {"", "auto", "production", "production_scan"}
    if raw_model.lower() in auto_tokens:
        model_path_p, profile = resolve_scan_production_model_path(cfg)
        apply_scan_obs_profile(cfg, profile)
        assert_scan_obs_profile(cfg, profile, ctx="scan_paired_eval:auto")
        if not model_path_p.exists():
            raise FileNotFoundError(
                f"[scan_paired_eval] Auto-selected scan model does not exist: {model_path_p}. "
                f"Checked profile '{profile.name}' candidates: {profile.model_candidates}"
            )
        model_path = str(model_path_p)
        print(
            f"[scan_paired_eval] auto-scan profile={profile.name} path_scale_upper={get_scan_path_scale_upper(cfg):.3f} "
            f"scan_max_steps_eff={effective_scan_max_steps(cfg)} model={model_path}"
        )
    else:
        model_path = normalize_model_path(raw_model)
    model = PPO.load(model_path, device=args.device)

    episodes = []
    baseline_hit_rl_miss = []
    for ep in range(int(args.episodes)):
        ep_seed = int(args.seed) + ep
        rl = _run_episode(ep_seed, cfg, model=model, mode="rl")
        baseline = _run_episode(ep_seed, cfg, model=None, mode="scripted_lawnmower")
        episodes.append({"episode_idx": int(ep), "seed": int(ep_seed), "rl": rl, "baseline": baseline})
        if bool(baseline["cov95_hit"]) and not bool(rl["cov95_hit"]):
            baseline_hit_rl_miss.append(
                {
                    "episode_idx": int(ep),
                    "seed": int(ep_seed),
                    "rl_final_coverage": float(rl["final_coverage"]),
                    "baseline_final_coverage": float(baseline["final_coverage"]),
                    "rl_time_to_95": int(rl["time_to_95"]),
                    "baseline_time_to_95": int(baseline["time_to_95"]),
                }
            )

    rl_rows = [ep["rl"] for ep in episodes]
    base_rows = [ep["baseline"] for ep in episodes]
    summary = {
        "preset": args.preset,
        "cfg_overrides": applied_overrides,
        "model": model_path,
        "episodes": int(args.episodes),
        "seed_base": int(args.seed),
        "rl_summary": _aggregate(rl_rows),
        "baseline_summary": _aggregate(base_rows),
        "baseline_hits_rl_misses": baseline_hit_rl_miss,
        "episodes_detail": episodes,
    }

    print("[scan_paired_eval] summary:", json.dumps(summary["rl_summary"], indent=2))
    print("[scan_paired_eval] baseline:", json.dumps(summary["baseline_summary"], indent=2))
    print(f"[scan_paired_eval] baseline_hits_rl_misses={len(baseline_hit_rl_miss)}")

    if args.json_out.strip():
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[scan_paired_eval] wrote summary: {out_path}")


if __name__ == "__main__":
    main()
