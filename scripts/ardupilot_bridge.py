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


def parse_args():
    parser = argparse.ArgumentParser(description="Send RL velocity setpoints to ArduPilot via MAVLink.")
    parser.add_argument("--dry-run", type=int, default=1, help="1: print setpoints only, 0: send MAVLink")
    parser.add_argument("--model", type=str, default="", help="Optional in dry-run mode")
    parser.add_argument("--task", type=str, default="hover")
    parser.add_argument("--preset", type=str, default="A0", choices=list_presets())
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--connection", type=str, default="udp:127.0.0.1:14550")
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=200)
    return parser.parse_args()


def _max_steps_for_task(task_name: str, cfg) -> int:
    if task_name.strip().lower() == "sequence":
        return int(cfg.seq_max_steps)
    return int(cfg.max_steps)


def _connect_mavlink(connection: str):
    from pymavlink import mavutil

    mav = mavutil.mavlink_connection(connection)
    mav.wait_heartbeat(timeout=30)
    return mav, mavutil


def _send_velocity_ned(mav, mavutil, vx: float, vy: float, vz_up: float, yaw_rate: float):
    # ArduPilot LOCAL_NED uses +z down; convert from +z up command.
    vz_ned = -vz_up
    type_mask = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
    )
    mav.mav.set_position_target_local_ned_send(
        int(time.time() * 1000),
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        0.0,
        0.0,
        0.0,
        float(vx),
        float(vy),
        float(vz_ned),
        0.0,
        0.0,
        0.0,
        0.0,
        float(yaw_rate),
    )


def main():
    args = parse_args()
    cfg = get_preset(args.preset)

    def make_env():
        task = build_task(args.task, cfg)
        return Quad15DEnv(task=task, cfg=cfg, max_steps=_max_steps_for_task(args.task, cfg))

    env = DummyVecEnv([make_env])
    obs = env.reset()

    model = None
    if args.model.strip():
        model_path = normalize_model_path(args.model)
        model = PPO.load(model_path, device=args.device)
        print(f"[bridge] Loaded model: {model_path}")
    elif int(args.dry_run) == 0:
        raise ValueError("--model is required when --dry-run 0")

    mav = None
    mavutil = None
    if int(args.dry_run) == 0:
        mav, mavutil = _connect_mavlink(args.connection)
        print(f"[bridge] MAVLink connected: {args.connection}")
    else:
        print("[bridge] Dry-run mode enabled. No MAVLink required.")

    dt = 1.0 / max(1e-3, float(args.hz))
    for step in range(int(args.steps)):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.zeros((1, 4), dtype=np.float32)

        cmd = action[0]
        vx, vy, vz, yaw_rate = float(cmd[0]), float(cmd[1]), float(cmd[2]), float(cmd[3])

        if int(args.dry_run) == 1:
            print(
                f"[bridge][dry] step={step + 1} "
                f"vx={vx:+.3f} vy={vy:+.3f} vz={vz:+.3f} yaw_rate={yaw_rate:+.3f}"
            )
        else:
            _send_velocity_ned(mav, mavutil, vx, vy, vz, yaw_rate)

        obs, rewards, dones, infos = env.step(action)
        if bool(dones[0]):
            obs = env.reset()

        time.sleep(dt)


if __name__ == "__main__":
    main()
