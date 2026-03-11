from __future__ import annotations

import argparse
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from quad_rl.curriculum.presets import get_preset, list_presets
from quad_rl.envs.quad15d_env import Quad15DEnv
from quad_rl.tasks import build_task
from quad_rl.utils.paths import normalize_model_path
from quad_rl.utils.scan_scale_profile import (
    apply_scan_obs_profile,
    assert_scan_obs_profile,
    effective_scan_max_steps,
    get_scan_path_scale_upper,
    resolve_scan_production_model_path,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Play/evaluate a policy step-by-step.")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--task", type=str, default="hover")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--preset", type=str, default="A0", choices=list_presets())
    parser.add_argument("--sleep", type=float, default=0.05, help="Realtime pause in seconds.")
    return parser.parse_args()


def _max_steps_for_task(task_name: str, cfg) -> int:
    key = task_name.strip().lower()
    if key == "sequence":
        return int(cfg.seq_max_steps)
    if key == "scan":
        return int(effective_scan_max_steps(cfg))
    return int(cfg.max_steps)


def main():
    args = parse_args()
    cfg = get_preset(args.preset)
    task_key = args.task.strip().lower()
    model = None
    raw_model = (args.model or "").strip()
    auto_tokens = {"", "auto", "production", "production_scan"}
    if task_key == "scan" and raw_model.lower() in auto_tokens:
        model_path_raw, profile = resolve_scan_production_model_path(cfg)
        apply_scan_obs_profile(cfg, profile)
        assert_scan_obs_profile(cfg, profile, ctx="play:auto")
        if not model_path_raw.exists():
            raise FileNotFoundError(
                f"[play] Auto-selected scan model does not exist: {model_path_raw}. "
                f"Checked profile '{profile.name}' candidates: {profile.model_candidates}"
            )
        model_path = str(model_path_raw)
        print(
            f"[play] auto-scan profile={profile.name} path_scale_upper={get_scan_path_scale_upper(cfg):.3f} "
            f"scan_max_steps_eff={effective_scan_max_steps(cfg)} model={model_path}"
        )
    elif raw_model.lower() in auto_tokens:
        model_path = ""
        print("[play] No model provided. Running zero-action policy.")
    elif raw_model:
        model_path = normalize_model_path(raw_model)
    else:
        print("[play] No model provided. Running zero-action policy.")
        model_path = ""

    if model_path:
        model = PPO.load(model_path, device=args.device)
        print(f"[play] Loaded model: {model_path}")

    def make_env():
        task = build_task(args.task, cfg)
        return Quad15DEnv(task=task, cfg=cfg, max_steps=_max_steps_for_task(args.task, cfg))

    env = DummyVecEnv([make_env])

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        steps = 0
        last_info = {}

        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = np.zeros((1, 4), dtype=np.float32)

            obs, rewards, dones, infos = env.step(action)
            done = bool(dones[0])  # DummyVecEnv: use dones[0]
            steps += 1
            last_info = infos[0]

            print(
                f"[play] ep={ep + 1} step={steps} "
                f"reward={float(rewards[0]):+.3f} "
                f"success={int(bool(last_info.get('success', False)))} "
                f"crash={int(bool(last_info.get('crash', False)))} "
                f"dist={last_info.get('dist', 0.0):.3f} "
                f"wp_idx={last_info.get('wp_idx', '-')}"
            )
            if args.sleep > 0:
                time.sleep(args.sleep)

        print(
            f"[play] episode={ep + 1} done "
            f"success={int(bool(last_info.get('success', False)))} "
            f"crash={int(bool(last_info.get('crash', False)))} "
            f"steps={steps}"
        )


if __name__ == "__main__":
    main()
