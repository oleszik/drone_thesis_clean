from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from quad_rl.curriculum.presets import get_preset, list_presets
from quad_rl.envs.quad15d_env import Quad15DEnv
from quad_rl.tasks import build_task
from quad_rl.utils.config_overrides import apply_overrides, parse_override_pairs
from quad_rl.utils.logging import dump_json, ensure_run_dir
from quad_rl.utils.paths import normalize_model_path
from quad_rl.utils.scan_scale_profile import (
    apply_scan_obs_profile,
    assert_scan_obs_profile,
    effective_scan_max_steps,
    get_scan_path_scale_upper,
    resolve_scan_production_model_path,
)
from quad_rl.utils.seeding import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO policy for Quad15DEnv.")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory under ./runs/")
    parser.add_argument("--total-timesteps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--task", type=str, default="hover")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--preset", type=str, default="A2", choices=list_presets())
    parser.add_argument("--load-model", type=str, default="")
    parser.add_argument("--n-steps", type=int, default=2048, help="PPO rollout steps per environment.")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Optional PPO learning rate override (applies to new or loaded models).",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=None,
        help="Optional PPO entropy coefficient override (applies to new or loaded models).",
    )
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


def _resolve_load_model_path(load_model_arg: str, task_key: str, cfg) -> Path | None:
    raw = (load_model_arg or "").strip()
    if not raw:
        return None
    lower = raw.lower()
    auto_tokens = {"auto", "production", "production_scan"}
    if task_key == "scan" and lower in auto_tokens:
        model_path, profile = resolve_scan_production_model_path(cfg)
        apply_scan_obs_profile(cfg, profile)
        assert_scan_obs_profile(cfg, profile, ctx="train:auto")
        if not model_path.exists():
            raise FileNotFoundError(
                f"[train] Auto-selected scan model does not exist: {model_path}. "
                f"Checked profile '{profile.name}' candidates: {profile.model_candidates}"
            )
        print(
            f"[train] auto-scan load profile={profile.name} path_scale_upper={get_scan_path_scale_upper(cfg):.3f} "
            f"scan_max_steps_eff={effective_scan_max_steps(cfg)} model={model_path}"
        )
        return model_path
    if lower in auto_tokens:
        raise ValueError("[train] --load-model auto is only supported for --task scan.")
    return Path(normalize_model_path(raw))


def _resolve_learning_rate(model: PPO, fallback: float = 3e-4) -> float:
    lr = getattr(model, "learning_rate", fallback)
    if callable(lr):
        try:
            return float(lr(1.0))
        except Exception:
            return float(fallback)
    try:
        return float(lr)
    except Exception:
        return float(fallback)


def _partial_policy_warmstart(new_model: PPO, old_model: PPO) -> tuple[int, int]:
    old_sd = old_model.policy.state_dict()
    new_sd = new_model.policy.state_dict()
    copied = 0
    partial = 0
    for key, new_val in new_sd.items():
        old_val = old_sd.get(key, None)
        if old_val is None:
            continue
        if old_val.shape == new_val.shape:
            new_sd[key] = old_val
            copied += 1
            continue
        if (
            old_val.ndim == 2
            and new_val.ndim == 2
            and old_val.shape[0] == new_val.shape[0]
            and old_val.shape[1] < new_val.shape[1]
        ):
            patched = new_val.clone()
            patched[:, : old_val.shape[1]] = old_val
            new_sd[key] = patched
            partial += 1
            continue
        if old_val.ndim == 1 and new_val.ndim == 1 and old_val.shape[0] < new_val.shape[0]:
            patched = new_val.clone()
            patched[: old_val.shape[0]] = old_val
            new_sd[key] = patched
            partial += 1
    new_model.policy.load_state_dict(new_sd, strict=False)
    return copied, partial


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
    set_seed(args.seed)

    cfg = get_preset(args.preset)
    cfg_overrides = parse_override_pairs(args.cfg_override)
    applied_overrides = apply_overrides(cfg, cfg_overrides) if cfg_overrides else {}
    task_key = args.task.strip().lower()
    load_path = _resolve_load_model_path(args.load_model, task_key, cfg)
    run_dir = ensure_run_dir(Path(args.run_dir))

    dump_json(
        run_dir / "cfg.json",
        {"args": vars(args), "preset": asdict(cfg), "cfg_overrides": applied_overrides},
    )

    def make_env():
        task = build_task(args.task, cfg)
        max_steps = _max_steps_for_task(args.task, cfg)
        return Quad15DEnv(task=task, cfg=cfg, max_steps=max_steps, seed=args.seed)

    vec_env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    model = None
    if load_path is not None:
        _enforce_scan_obs_aug_compat(load_path, cfg, args.task, ctx="train")
        loaded_model = PPO.load(str(load_path), device=args.device)
        # Keep action-space contract fixed at normalized [-1, 1].
        if loaded_model.action_space != vec_env.action_space:
            print(
                "[train] Loaded model action-space metadata differs from env. "
                "Overriding loaded metadata with normalized env action space."
            )
            loaded_model.action_space = vec_env.action_space
            loaded_model.policy.action_space = vec_env.action_space

        obs_mismatch = loaded_model.observation_space != vec_env.observation_space
        if obs_mismatch:
            base_lr = float(args.learning_rate) if args.learning_rate is not None else _resolve_learning_rate(loaded_model)
            base_ent = float(args.ent_coef) if args.ent_coef is not None else float(getattr(loaded_model, "ent_coef", 0.0))
            print(
                "[train] Observation-space mismatch detected. Building new policy and "
                "partially warm-starting weights from loaded checkpoint."
            )
            model = PPO(
                "MlpPolicy",
                vec_env,
                verbose=1,
                tensorboard_log=str(run_dir / "tb"),
                seed=args.seed,
                device=args.device,
                n_steps=max(8, int(args.n_steps)),
                learning_rate=base_lr,
                ent_coef=base_ent,
            )
            copied, partial = _partial_policy_warmstart(model, loaded_model)
            print(
                f"[train] Warm-start copied policy params: full={copied}, partial={partial} "
                "(obs-aug extra inputs stay randomly initialized)."
            )
        else:
            model = loaded_model
            if args.learning_rate is not None:
                lr = float(args.learning_rate)
                model.learning_rate = lr
                model.lr_schedule = lambda _: lr
                for param_group in model.policy.optimizer.param_groups:
                    param_group["lr"] = lr
                print(f"[train] Overrode learning_rate for loaded model: {lr:g}")
            if args.ent_coef is not None:
                ent = float(args.ent_coef)
                model.ent_coef = ent
                print(f"[train] Overrode ent_coef for loaded model: {ent:g}")
            model.set_env(vec_env)
        print(f"[train] Loaded model from: {load_path}")
    else:
        lr = float(args.learning_rate) if args.learning_rate is not None else 3e-4
        ent_coef = float(args.ent_coef) if args.ent_coef is not None else 0.0
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=str(run_dir / "tb"),
            seed=args.seed,
            device=args.device,
            n_steps=max(8, int(args.n_steps)),
            learning_rate=lr,
            ent_coef=ent_coef,
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir / "eval"),
        eval_freq=max(1, int(args.eval_freq)),
        deterministic=True,
        render=False,
    )

    sb3_logger = configure(str(run_dir / "sb3_logs"), ["csv", "tensorboard"])
    model.set_logger(sb3_logger)

    try:
        model.learn(total_timesteps=int(args.total_timesteps), callback=eval_callback)
        final_path = str(run_dir / "final_model")
        model.save(final_path)
        print(f"[train] Saved final model: {final_path}.zip")

        best_zip = run_dir / "best_model.zip"
        if not best_zip.exists():
            fallback = str(run_dir / "best_model")
            model.save(fallback)
            print(f"[train] Eval callback produced no best_model; saved fallback at: {fallback}.zip")
    finally:
        try:
            if getattr(model, "env", None) is not None:
                model.env.close()
        except Exception:
            pass
        try:
            vec_env.close()
        except Exception:
            pass
        try:
            eval_env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
