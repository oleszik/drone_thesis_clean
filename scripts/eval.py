from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from quad_rl.curriculum.presets import get_preset, list_presets
from quad_rl.envs.quad15d_env import Quad15DEnv
from quad_rl.tasks import build_task
from quad_rl.utils.config_overrides import apply_overrides, parse_override_pairs
from quad_rl.utils.paths import normalize_model_path
from quad_rl.utils.scan_scale_profile import (
    apply_scan_obs_profile,
    assert_scan_obs_profile,
    effective_scan_max_steps,
    get_scan_path_scale_upper,
    resolve_scan_production_model_path,
    scan_step_scaling_details,
)


def _slug(text: str) -> str:
    raw = (text or "").strip()
    out = "".join(ch if (ch.isalnum() or ch in "-_") else "-" for ch in raw)
    out = out.strip("-_")
    return out or "model"


def _next_free_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    i = 1
    while True:
        candidate = path.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO model on Quad15DEnv.")
    parser.add_argument("--model", type=str, default="auto")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--task", type=str, default="hover")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--preset", type=str, default="A2", choices=list_presets())
    parser.add_argument("--seed", type=int, default=456, help="Base eval seed; should differ from train seed.")
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument(
        "--cfg-override",
        action="append",
        default=[],
        help="Override preset fields with key=value (repeatable). Example: --cfg-override scan_k_cov_gain=0.012",
    )
    return parser.parse_args()


def _max_steps_for_task(task_name: str, cfg) -> int:
    key = task_name.strip().lower()
    if key == "sequence":
        return int(cfg.seq_max_steps)
    if key == "scan":
        return int(effective_scan_max_steps(cfg))
    return int(cfg.max_steps)


def _resolve_model_path(model_arg: str, task_key: str, cfg) -> Path:
    raw = (model_arg or "").strip()
    lower = raw.lower()
    auto_tokens = {"", "auto", "production", "production_scan"}
    if task_key == "scan" and lower in auto_tokens:
        model_path, profile = resolve_scan_production_model_path(cfg)
        apply_scan_obs_profile(cfg, profile)
        assert_scan_obs_profile(cfg, profile, ctx="eval:auto")
        if not model_path.exists():
            raise FileNotFoundError(
                f"[eval] Auto-selected scan model does not exist: {model_path}. "
                f"Checked profile '{profile.name}' candidates: {profile.model_candidates}"
            )
        print(
            f"[eval] auto-scan profile={profile.name} path_scale_upper={get_scan_path_scale_upper(cfg):.3f} "
            f"scan_max_steps_eff={effective_scan_max_steps(cfg)} model={model_path}"
        )
        return model_path
    if lower in auto_tokens:
        raise ValueError("[eval] --model is required for non-scan tasks.")
    return Path(normalize_model_path(raw))


def _load_model_preset_from_cfg(model_path: Path) -> dict | None:
    cfg_path = model_path.parent / "cfg.json"
    if not cfg_path.exists():
        return None
    try:
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    preset = payload.get("preset")
    return preset if isinstance(preset, dict) else None


def _enforce_scan_obs_aug_compat(model_path: Path, cfg, task_name: str, ctx: str) -> None:
    if (task_name or "").strip().lower() != "scan":
        return
    preset = _load_model_preset_from_cfg(model_path)
    if not preset:
        return
    required_enable = bool(preset.get("scan_obs_aug_enable", False))
    required_patch = int(preset.get("scan_obs_patch_size", 5))
    required_boundary = bool(preset.get("scan_obs_boundary_feat", True))
    required_global = bool(preset.get("scan_obs_global_coverage_enable", False))
    required_global_size = int(preset.get("scan_obs_global_size", 8))
    required_meas = bool(preset.get("obs_meas_aug_enable", False))
    required_ekf = bool(preset.get("obs_ekf_quality_enable", False))
    got_enable = bool(getattr(cfg, "scan_obs_aug_enable", False))
    got_patch = int(getattr(cfg, "scan_obs_patch_size", 5))
    got_boundary = bool(getattr(cfg, "scan_obs_boundary_feat", True))
    got_global = bool(getattr(cfg, "scan_obs_global_coverage_enable", False))
    got_global_size = int(getattr(cfg, "scan_obs_global_size", 8))
    got_meas = bool(getattr(cfg, "obs_meas_aug_enable", False))
    got_ekf = bool(getattr(cfg, "obs_ekf_quality_enable", False))
    if not required_enable and not required_meas:
        return
    if (
        (got_enable == required_enable)
        and (got_patch == required_patch)
        and (got_boundary == required_boundary)
        and (got_global == required_global)
        and (got_global_size == required_global_size)
        and (got_meas == required_meas)
        and (got_ekf == required_ekf)
    ):
        return
    raise ValueError(
        f"[{ctx}] Model requires scan obs augmentation from model cfg "
        f"(enable={required_enable}, patch={required_patch}, boundary_feat={required_boundary}, "
        f"global_cov={required_global}, global_size={required_global_size}, "
        f"meas_aug={required_meas}, ekf_aug={required_ekf}) "
        f"but runtime preset is (enable={got_enable}, patch={got_patch}, boundary_feat={got_boundary}, "
        f"global_cov={got_global}, global_size={got_global_size}, "
        f"meas_aug={got_meas}, ekf_aug={got_ekf}). "
        "Set matching --cfg-override values."
    )


def main():
    args = parse_args()
    cfg = get_preset(args.preset)
    cfg_overrides = parse_override_pairs(args.cfg_override)
    applied_overrides = apply_overrides(cfg, cfg_overrides) if cfg_overrides else {}
    task_key = args.task.strip().lower()
    model_path = _resolve_model_path(args.model, task_key, cfg)
    _enforce_scan_obs_aug_compat(model_path, cfg, task_key, ctx="eval")

    def make_env():
        task = build_task(args.task, cfg)
        max_steps = _max_steps_for_task(args.task, cfg)
        # Fresh env instance dedicated to evaluation (no training-env leakage).
        return Quad15DEnv(task=task, cfg=cfg, max_steps=max_steps, seed=args.seed, is_eval=True)

    env = DummyVecEnv([make_env])
    model = PPO.load(str(model_path), device=args.device)
    if model.action_space != env.action_space:
        print(
            "[eval] Loaded model action-space metadata differs from env. "
            "Overriding loaded metadata with normalized env action space."
        )
        model.action_space = env.action_space
        model.policy.action_space = env.action_space

    success_flags = []
    crash_flags = []
    steps_list = []
    oob_touch_step_counts = []
    oob_touch_episode_flags = []
    scan_cov_episode = []
    scan_overlap_episode = []
    scan_time95_episode = []
    scan_covered_cells_episode = []
    scan_total_cells_episode = []
    yaw_rate_abs_all = []
    dv_xy_all = []
    scan_max_steps_base, scan_path_len_scale, scan_max_steps, scan_max_steps_cap_hit = scan_step_scaling_details(cfg)
    scan_v_xy_max = getattr(cfg, "scan_v_xy_max", None)
    if scan_v_xy_max is None:
        scan_v_xy_max = cfg.v_xy_max
    scan_v_xy_max = float(scan_v_xy_max)
    dt = float(cfg.dt)
    for ep in range(args.episodes):
        if hasattr(env, "seed"):
            env.seed(int(args.seed) + ep)
        obs = env.reset()
        done = False
        steps = 0
        last_info = {}
        ep_yaw_rate_abs = []
        ep_dv_xy = []
        obs_arr = np.asarray(obs, dtype=np.float32)
        prev_vxy = np.zeros((2,), dtype=np.float32)
        if obs_arr.ndim >= 2 and obs_arr.shape[0] >= 1 and obs_arr.shape[1] >= 5:
            prev_vxy = obs_arr[0, 3:5].astype(np.float32, copy=True)
        scan_progress_start = None
        scan_progress_end = None
        scan_ct_vals: list[float] = []
        scan_la_vals: list[float] = []
        scan_path_total_len = None
        ep_oob_touch_steps = 0
        if task_key == "scan":
            local_env = env.envs[0] if getattr(env, "envs", None) else None
            if local_env is not None and hasattr(local_env, "task"):
                scan_path_total_len = getattr(local_env.task, "total_length", None)
                if scan_path_total_len is not None:
                    scan_path_total_len = float(scan_path_total_len)

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            done = bool(dones[0])  # DummyVecEnv: use dones[0]
            steps += 1
            last_info = infos[0]
            if bool(last_info.get("oob_touch", False)):
                ep_oob_touch_steps += 1
            obs_arr = np.asarray(obs, dtype=np.float32)
            if obs_arr.ndim >= 2 and obs_arr.shape[0] >= 1 and obs_arr.shape[1] >= 12:
                curr_vxy = obs_arr[0, 3:5].astype(np.float32, copy=False)
                curr_yaw_rate = float(obs_arr[0, 11])
                dv_xy = float(np.linalg.norm(curr_vxy - prev_vxy))
                ep_dv_xy.append(dv_xy)
                ep_yaw_rate_abs.append(abs(curr_yaw_rate))
                prev_vxy = curr_vxy.astype(np.float32, copy=True)

            if task_key == "scan":
                p = last_info.get("progress_along_path", None)
                if p is not None:
                    p_val = float(p)
                    if scan_progress_start is None:
                        scan_progress_start = p_val
                    scan_progress_end = p_val
                path_len = last_info.get("path_total_len", None)
                if path_len is not None:
                    scan_path_total_len = float(path_len)
                ct = last_info.get("cross_track_error", None)
                if ct is not None:
                    scan_ct_vals.append(float(ct))
                d_la = last_info.get("dist_to_lookahead", None)
                if d_la is not None:
                    scan_la_vals.append(float(d_la))
            if args.debug:
                msg = (
                    f"[eval] ep={ep + 1} step={steps} "
                    f"reward={float(rewards[0]):+.3f} "
                    f"dist={last_info.get('dist', 0.0):.3f} "
                    f"wp_idx={last_info.get('wp_idx', '-')}"
                )
                print(msg)

        success = bool(last_info.get("success", False))
        crash = bool(last_info.get("crash", False))
        success_flags.append(1.0 if success else 0.0)
        crash_flags.append(1.0 if crash else 0.0)
        steps_list.append(float(steps))
        yaw_rate_abs_all.extend(ep_yaw_rate_abs)
        dv_xy_all.extend(ep_dv_xy)
        oob_touch_step_counts.append(float(ep_oob_touch_steps))
        oob_touch_episode_flags.append(1.0 if ep_oob_touch_steps > 0 else 0.0)
        ep_yaw_abs_mean = float(np.mean(np.asarray(ep_yaw_rate_abs, dtype=np.float32))) if ep_yaw_rate_abs else 0.0
        ep_yaw_abs_p95 = (
            float(np.percentile(np.asarray(ep_yaw_rate_abs, dtype=np.float32), 95.0))
            if ep_yaw_rate_abs
            else 0.0
        )
        ep_dv_xy_mean = float(np.mean(np.asarray(ep_dv_xy, dtype=np.float32))) if ep_dv_xy else 0.0
        print(
            f"[eval] episode={ep + 1} success={int(success)} crash={int(crash)} "
            f"steps={steps} oob_touch_steps={ep_oob_touch_steps}"
        )
        print(
            f"[eval] smooth ep={ep + 1} yaw_rate_abs_mean={ep_yaw_abs_mean:.3f} "
            f"yaw_rate_abs_p95={ep_yaw_abs_p95:.3f} dv_xy_mean={ep_dv_xy_mean:.3f}"
        )
        if task_key == "scan":
            p0 = float(scan_progress_start) if scan_progress_start is not None else 0.0
            p1 = float(scan_progress_end) if scan_progress_end is not None else p0
            p_delta = p1 - p0
            ct_mean = float(np.mean(np.asarray(scan_ct_vals, dtype=np.float32))) if scan_ct_vals else 0.0
            d_la_mean = float(np.mean(np.asarray(scan_la_vals, dtype=np.float32))) if scan_la_vals else 0.0
            path_total_len = float(scan_path_total_len) if scan_path_total_len is not None else 0.0
            ep_coverage = float(last_info.get("coverage", 0.0))
            ep_overlap = float(last_info.get("overlap", 0.0))
            ep_covered_cells = int(last_info.get("covered_cells", 0))
            ep_total_cells = int(last_info.get("total_cells", 0))
            ep_time95 = int(last_info.get("time_to_95", -1))
            if ep_time95 < 0:
                ep_time95 = int(scan_max_steps + 1)
            scan_cov_episode.append(ep_coverage)
            scan_overlap_episode.append(ep_overlap)
            scan_time95_episode.append(float(ep_time95))
            scan_covered_cells_episode.append(float(ep_covered_cells))
            scan_total_cells_episode.append(float(ep_total_cells))
            required_steps = (path_total_len / max(scan_v_xy_max, 1e-6)) * (1.0 / max(dt, 1e-6))
            impossible = required_steps > float(scan_max_steps)
            dbg_seg_idx = int(last_info.get("seg_idx", -1))
            dbg_n_segs = int(last_info.get("n_segs", last_info.get("n_segments", 0)))
            dbg_s_on_seg = float(last_info.get("s_on_seg", last_info.get("s_local", 0.0)))
            dbg_seg_len = float(last_info.get("seg_len", 0.0))
            dbg_is_last = bool(last_info.get("is_last_seg", False))
            dbg_success_cond = bool(last_info.get("success_cond", False))
            dbg_would_be_success = bool(last_info.get("would_be_success", dbg_success_cond))
            dbg_vx_cmd = float(last_info.get("vx_cmd", 0.0))
            dbg_vy_cmd = float(last_info.get("vy_cmd", 0.0))
            dbg_v_xy_limit_eff = float(last_info.get("v_xy_limit_eff", 0.0))
            dbg_action_raw = last_info.get("action_raw", np.zeros((2,), dtype=np.float32))
            dbg_action_raw = np.asarray(dbg_action_raw, dtype=np.float32).reshape(-1)
            dbg_action_raw_x = float(dbg_action_raw[0]) if dbg_action_raw.size >= 1 else 0.0
            dbg_action_raw_y = float(dbg_action_raw[1]) if dbg_action_raw.size >= 2 else 0.0
            dbg_wall_active = bool(last_info.get("scan_wall_active", False))
            dbg_d_edge = last_info.get("scan_d_edge", None)
            dbg_d_edge = float(dbg_d_edge) if dbg_d_edge is not None else float("nan")
            print(
                f"[eval] scan sanity ep={ep + 1} path_total_len={path_total_len:.3f} "
                f"required_steps_min={required_steps:.1f} scan_max_steps={scan_max_steps} "
                f"cant_succeed={int(impossible)} path_len_scale={scan_path_len_scale:.3f} "
                f"scan_max_steps_base={scan_max_steps_base} cap_hit={int(scan_max_steps_cap_hit)}"
            )
            print(
                f"[eval] scan last ep={ep + 1} seg_idx={dbg_seg_idx} n_segs={dbg_n_segs} "
                f"s_on_seg={dbg_s_on_seg:.3f} seg_len={dbg_seg_len:.3f} "
                f"is_last_seg={int(dbg_is_last)} success_cond={int(dbg_success_cond)} "
                f"would_be_success={int(dbg_would_be_success)}"
            )
            print(
                f"[eval] scan cmd ep={ep + 1} action_raw=({dbg_action_raw_x:.3f},{dbg_action_raw_y:.3f}) "
                f"vx_cmd={dbg_vx_cmd:.3f} vy_cmd={dbg_vy_cmd:.3f} "
                f"v_xy_limit_eff={dbg_v_xy_limit_eff:.3f} "
                f"d_edge={dbg_d_edge:.3f} wall_active={int(dbg_wall_active)}"
            )
            print(
                f"[eval] scan ep={ep + 1} progress_start={p0:.3f} "
                f"progress_end={p1:.3f} progress_delta={p_delta:.3f} "
                f"ct_mean={ct_mean:.3f} dist_la_mean={d_la_mean:.3f}"
            )
            print(
                f"[eval] scan coverage ep={ep + 1} covered_cells={ep_covered_cells} "
                f"total_cells={ep_total_cells} coverage={ep_coverage:.3f} "
                f"overlap={ep_overlap:.3f} time_to_95={ep_time95}"
            )

    success_arr = np.asarray(success_flags, dtype=np.float32)
    crash_arr = np.asarray(crash_flags, dtype=np.float32)
    steps_arr = np.asarray(steps_list, dtype=np.float32)
    oob_touch_steps_arr = np.asarray(oob_touch_step_counts, dtype=np.float32)
    oob_touch_eps_arr = np.asarray(oob_touch_episode_flags, dtype=np.float32)
    success_steps_arr = np.asarray(
        [s for s, ok in zip(steps_list, success_flags) if ok > 0.5], dtype=np.float32
    )

    success_rate_mean = float(np.mean(success_arr)) if success_arr.size else 0.0
    success_rate_std = float(np.std(success_arr)) if success_arr.size else 0.0
    steps_mean = float(np.mean(steps_arr)) if steps_arr.size else 0.0
    steps_std = float(np.std(steps_arr)) if steps_arr.size else 0.0
    crash_rate_mean = float(np.mean(crash_arr)) if crash_arr.size else 0.0
    crash_rate_std = float(np.std(crash_arr)) if crash_arr.size else 0.0
    mean_steps_success = float(np.mean(success_steps_arr)) if success_steps_arr.size else 0.0
    std_steps_success = float(np.std(success_steps_arr)) if success_steps_arr.size else 0.0
    success_count = int(np.sum(success_arr)) if success_arr.size else 0
    crash_count = int(np.sum(crash_arr)) if crash_arr.size else 0
    total_steps = int(np.sum(steps_arr)) if steps_arr.size else 0
    yaw_rate_abs_arr = np.asarray(yaw_rate_abs_all, dtype=np.float32)
    dv_xy_arr = np.asarray(dv_xy_all, dtype=np.float32)
    yaw_rate_abs_mean = float(np.mean(yaw_rate_abs_arr)) if yaw_rate_abs_arr.size else 0.0
    yaw_rate_abs_p95 = float(np.percentile(yaw_rate_abs_arr, 95.0)) if yaw_rate_abs_arr.size else 0.0
    dv_xy_mean = float(np.mean(dv_xy_arr)) if dv_xy_arr.size else 0.0
    oob_touch_count = int(np.sum(oob_touch_steps_arr)) if oob_touch_steps_arr.size else 0
    oob_touch_rate = float(oob_touch_count / max(total_steps, 1))
    oob_touch_episode_count = int(np.sum(oob_touch_eps_arr)) if oob_touch_eps_arr.size else 0
    oob_touch_episode_rate = float(np.mean(oob_touch_eps_arr)) if oob_touch_eps_arr.size else 0.0

    summary = {
        "episodes": int(args.episodes),
        "seed_base": int(args.seed),
        "cfg_overrides": applied_overrides,
        "success_count": success_count,
        "crash_count": crash_count,
        "oob_touch_count": oob_touch_count,
        "oob_touch_rate": oob_touch_rate,
        "oob_touch_episode_count": oob_touch_episode_count,
        "oob_touch_episode_rate": oob_touch_episode_rate,
        "success_rate_mean": success_rate_mean,
        "success_rate_std": success_rate_std,
        "steps_mean": steps_mean,
        "steps_std": steps_std,
        "mean_steps_success": mean_steps_success,
        "std_steps_success": std_steps_success,
        "crash_rate_mean": crash_rate_mean,
        "crash_rate_std": crash_rate_std,
        "yaw_rate_abs_mean": yaw_rate_abs_mean,
        "yaw_rate_abs_p95": yaw_rate_abs_p95,
        "dv_xy_mean": dv_xy_mean,
    }
    if task_key == "scan":
        cov_arr = np.asarray(scan_cov_episode, dtype=np.float32)
        overlap_arr = np.asarray(scan_overlap_episode, dtype=np.float32)
        time95_arr = np.asarray(scan_time95_episode, dtype=np.float32)
        covered_cells_arr = np.asarray(scan_covered_cells_episode, dtype=np.float32)
        total_cells_arr = np.asarray(scan_total_cells_episode, dtype=np.float32)
        coverage_mean = float(np.mean(cov_arr)) if cov_arr.size else 0.0
        coverage_min = float(np.min(cov_arr)) if cov_arr.size else 0.0
        coverage_ge_090_rate = float(np.mean(cov_arr >= 0.90)) if cov_arr.size else 0.0
        coverage_ge_093_rate = float(np.mean(cov_arr >= 0.93)) if cov_arr.size else 0.0
        coverage_ge_095_rate = float(np.mean(cov_arr >= 0.95)) if cov_arr.size else 0.0
        overlap_mean = float(np.mean(overlap_arr)) if overlap_arr.size else 0.0
        time_to_95_mean = float(np.mean(time95_arr)) if time95_arr.size else float(scan_max_steps + 1)
        covered_cells_mean = float(np.mean(covered_cells_arr)) if covered_cells_arr.size else 0.0
        total_cells_mean = float(np.mean(total_cells_arr)) if total_cells_arr.size else 0.0
        summary.update(
            {
                "scan_max_steps_base": int(scan_max_steps_base),
                "path_len_scale": float(scan_path_len_scale),
                "scan_max_steps_eff": int(scan_max_steps),
                "scan_max_steps_cap_hit": bool(scan_max_steps_cap_hit),
                "coverage_mean": coverage_mean,
                "coverage_min": coverage_min,
                "coverage_ge_090_rate": coverage_ge_090_rate,
                "coverage_ge_093_rate": coverage_ge_093_rate,
                "coverage_ge_095_rate": coverage_ge_095_rate,
                "overlap_mean": overlap_mean,
                "time_to_95_mean": time_to_95_mean,
                "covered_cells_mean": covered_cells_mean,
                "total_cells_mean": total_cells_mean,
                "coverage_episode": [float(x) for x in cov_arr.tolist()],
            }
        )
    print("[eval] summary:", json.dumps(summary, indent=2))
    print(f"[eval] success_count={success_count} crash_count={crash_count}")
    print(
        f"[eval] success_rate={success_rate_mean:.3f}+/-{success_rate_std:.3f} "
        f"steps={steps_mean:.1f}+/-{steps_std:.1f}"
    )

    if args.json_out.strip():
        out_req = Path(args.json_out)
        model_run = _slug(model_path.parent.name if model_path.parent is not None else "model")
        tag = f"{task_key}_{args.preset}_{model_run}_e{int(args.episodes)}_s{int(args.seed)}"
        if out_req.suffix.lower() != ".json":
            out_req.mkdir(parents=True, exist_ok=True)
            out_path = out_req / f"{tag}.json"
        else:
            out_req.parent.mkdir(parents=True, exist_ok=True)
            out_path = out_req
            if out_path.exists():
                out_path = out_path.with_name(f"{out_path.stem}_{tag}{out_path.suffix}")
        out_path = _next_free_path(out_path)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[eval] wrote summary: {out_path}")


if __name__ == "__main__":
    main()
