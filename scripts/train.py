from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from quad_rl.curriculum.presets import get_preset, list_presets
from quad_rl.envs.quad15d_env import Quad15DEnv
from quad_rl.tasks import build_task
from quad_rl.utils.config_overrides import apply_overrides, parse_override_pairs
from quad_rl.utils.logging import dump_json, ensure_run_dir
from quad_rl.utils.paths import normalize_model_path
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
        return int(getattr(cfg, "scan_max_steps", cfg.max_steps))
    return int(cfg.max_steps)


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = get_preset(args.preset)
    cfg_overrides = parse_override_pairs(args.cfg_override)
    applied_overrides = apply_overrides(cfg, cfg_overrides) if cfg_overrides else {}
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
    if args.load_model.strip():
        load_path = normalize_model_path(args.load_model)
        model = PPO.load(load_path, device=args.device)
        # Keep action-space contract fixed at normalized [-1, 1].
        if model.action_space != vec_env.action_space:
            print(
                "[train] Loaded model action-space metadata differs from env. "
                "Overriding loaded metadata with normalized env action space."
            )
            model.action_space = vec_env.action_space
            model.policy.action_space = vec_env.action_space
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

    model.learn(total_timesteps=int(args.total_timesteps), callback=eval_callback)
    final_path = str(run_dir / "final_model")
    model.save(final_path)
    print(f"[train] Saved final model: {final_path}.zip")

    best_zip = run_dir / "best_model.zip"
    if not best_zip.exists():
        fallback = str(run_dir / "best_model")
        model.save(fallback)
        print(f"[train] Eval callback produced no best_model; saved fallback at: {fallback}.zip")


if __name__ == "__main__":
    main()
