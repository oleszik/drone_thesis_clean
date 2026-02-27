# Curriculum RL Quadrotor Thesis Project

This repository implements:

Curriculum-Based Reinforcement Learning for Autonomous Quadrotor Flight Control in a 15-Dimensional Simulation Environment.

## Observation and Action Spaces

Observation is always 15D, for all tasks:

`[e_x,e_y,e_z, v_x,v_y,v_z, roll,pitch, yaw_err, p,q,r, a_prev_x,a_prev_y,a_prev_z]`

Action is always 4D, for all tasks:

`[vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd]`

Env action space is normalized `Box(-1,1,4)`; physical scaling is preset-driven.

## Quickstart (Windows)

Create venv + install deps:

```powershell
cd /d "D:\drone_thesis_clean" && python -m venv ".venv" && "D:\drone_thesis_clean\.venv\Scripts\python.exe" -m pip install --upgrade pip && "D:\drone_thesis_clean\.venv\Scripts\python.exe" -m pip install -r "requirements.txt"
```

Syntax check:

```powershell
"D:\drone_thesis_clean\.venv\Scripts\python.exe" -m compileall "D:\drone_thesis_clean\quad_rl" "D:\drone_thesis_clean\scripts"
```

## Current baselines

- Sequence A2: `runs\sequence_a2_v9_oobtouch\best_model.zip`
- Scan A2 production: `runs\production_scan\best_model.zip`
- Scan config note: `allow_oob_touch_scan=False`, `scan_max_steps=2400`

Eval sequence (200 eps gate):

```powershell
"D:\drone_thesis_clean\.venv\Scripts\python.exe" -m scripts.eval --model "runs\sequence_a2_v9_oobtouch\best_model.zip" --task sequence --preset A2 --episodes 200 --seed 456 --device cpu
```

Eval scan (50 eps gate):

```powershell
"D:\drone_thesis_clean\.venv\Scripts\python.exe" -m scripts.eval --model "runs\production_scan\best_model.zip" --task scan --preset A2 --episodes 50 --seed 456 --device cpu
```

Play (sequence):

```powershell
"D:\drone_thesis_clean\.venv\Scripts\python.exe" -m scripts.play --model "runs\sequence_a2_v9_oobtouch\best_model.zip" --task sequence --preset A2 --device cpu
```

Play (scan):

```powershell
"D:\drone_thesis_clean\.venv\Scripts\python.exe" -m scripts.play --model "runs\production_scan\best_model.zip" --task scan --preset A2 --device cpu
```

Sanity gates:

- sequence (seed 456, 200 eps): expect `crash_count=0`, `success_count~187/200`
- scan production (seed 456, 100 eps): expect `success_count~70/100`, `coverage_mean~0.87`, `crash_count=0`

## Current production scan model

- Model path: `runs\production_scan\best_model.zip`
- Manifest: `runs\production_scan\production_manifest.json`
- Metrics dir: `runs\production_scan\metrics`

Gate results pinned in production manifest:

- Seed 456 (100 eps): `success_count=70`, `crash_count=0`, `coverage_mean=0.8701`
- Robust seeds 1/2/3 (100 eps each): `success_count=74/100` each, `crash_count=0`, `coverage_mean~0.878`

Reproduce the winning finetune (from v35 baseline):

```powershell
"D:\drone_thesis_clean\.venv\Scripts\python.exe" -m scripts.train --run-dir "runs\scan_v37_short_sweeps_from_v35\gain_0p012" --load-model "runs\scan_prod_v35_best_by_gate\best_model.zip" --total-timesteps 300000 --seed 123 --task scan --preset A2_ABL_B125_T85 --device cpu --n-steps 4096 --eval-freq 200000 --learning-rate 5e-5 --ent-coef 0.0 --cfg-override scan_cov_late_thresh=0.85 --cfg-override scan_k_cov_gain_late=0.01 --cfg-override scan_k_cov_stall=0.0 --cfg-override scan_oob_grace_steps=3 --cfg-override scan_k_cov_gain=0.012 --cfg-override scan_debug_oob=false
```

Reproduce seed-456 gate (100 eps):

```powershell
"D:\drone_thesis_clean\.venv\Scripts\python.exe" -m scripts.eval --model "runs\production_scan\best_model.zip" --task scan --preset A2 --episodes 100 --seed 456 --device cpu --json-out "runs\production_scan\metrics\gate_seed456_100eps_repro.json"
```

Reproduce 3-seed robust gates:

```powershell
for %S in (1 2 3) do "D:\drone_thesis_clean\.venv\Scripts\python.exe" -m scripts.eval --model "runs\production_scan\best_model.zip" --task scan --preset A2 --episodes 100 --seed %S --device cpu --json-out "runs\production_scan\metrics\gate_seed_%S_100eps_repro.json"
```

Reproduce lawnmower feasibility baseline:

```powershell
"D:\drone_thesis_clean\.venv\Scripts\python.exe" -m scripts.scan_lawnmower_baseline --preset A2 --episodes 100 --seed 456 --json-out "runs\production_scan\metrics\lawnmower_feasibility_seed456_e100_repro.json"
```

Sweep and override utilities:

- `--cfg-override key=value` can be passed repeatedly to `scripts.train` and `scripts.eval` for preset-field sweeps.
- `scripts.scan_short_sweeps` runs short single-knob scans from a base model and saves `best_by_gate_model.zip`.
- Quick sanity check command:
```powershell
"D:\drone_thesis_clean\.venv\Scripts\python.exe" -m scripts.verify_production_scan --episodes 10 --seed 456 --device cpu
```

## Train

```powershell
"D:\drone_thesis_clean\.venv\Scripts\python.exe" -m scripts.train --run-dir "runs\sequence_a1s3_rescue" --total-timesteps 800000 --seed 123 --task sequence --device cpu --preset A1_S3b --n-steps 4096 --eval-freq 200000
```

Warm-start:

```powershell
"D:\drone_thesis_clean\.venv\Scripts\python.exe" -m scripts.train --run-dir "runs\sequence_a2_v9_oobtouch_continue" --total-timesteps 200000 --seed 123 --task sequence --device cpu --preset A2 --n-steps 4096 --eval-freq 200000 --load-model "runs\sequence_a2_v9_oobtouch\best_model.zip"
```

## Evaluate

```powershell
"D:\drone_thesis_clean\.venv\Scripts\python.exe" -m scripts.eval --model "runs\sequence_a2_v9_oobtouch\best_model.zip" --episodes 200 --task sequence --device cpu --preset A2 --seed 456 --debug 0
```

`scripts.eval` uses `DummyVecEnv`, so episode termination uses `dones[0]`.

## Play

```powershell
"D:\drone_thesis_clean\.venv\Scripts\python.exe" -m scripts.play --model "runs\sequence_a2_v9_oobtouch\best_model.zip" --episodes 1 --task sequence --preset A2 --device cpu --sleep 0.05
```

## ArduPilot Bridge

Dry-run without SITL or model:

```bash
python -m scripts.ardupilot_bridge --dry-run 1 --steps 50 --hz 10
```

With model and MAVLink:

```bash
python -m scripts.ardupilot_bridge \
  --dry-run 0 \
  --model runs/sequence_a2_v9_oobtouch/best_model.zip \
  --task sequence \
  --preset A2 \
  --connection udp:127.0.0.1:14550 \
  --hz 10 \
  --steps 300
```

## Tasks

Supported task names:

- `hover`
- `yaw`
- `landing` (alias: `land`)
- `waypoint`
- `sequence`
- `scan`
- `mission`
- `stage_a` (mixed hover/yaw/landing)

## Presets

Available presets:

- `A0`
- `A0_S2`
- `A1`
- `A1_S3`
- `A1_S3b`
- `A2`
- `A2_S4`

Use:

```bash
python -m scripts.train --task hover --preset A0
```

## Notes

- The dynamics model is lightweight point-mass 3D (no PyBullet).
- Gravity compensation is included so `vz_cmd=0` can hover.
- Success always sets `terminated=True` for early episode end.
- `quad_rl.utils.paths.normalize_model_path()` prevents `.zip.zip` path issues.
