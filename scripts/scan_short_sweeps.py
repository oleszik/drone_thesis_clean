from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from quad_rl.curriculum.presets import get_preset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run short single-knob scan finetune sweeps from a base model.")
    parser.add_argument(
        "--base-model",
        type=str,
        default="runs/scan_prod_v35_best_by_gate/best_model.zip",
        help="Base model path to finetune from (typically frozen production v35).",
    )
    parser.add_argument("--preset", type=str, default="A2_ABL_B125_T85")
    parser.add_argument("--run-root", type=str, default="runs/scan_v37_short_sweeps_from_v35")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-train", type=int, default=123)
    parser.add_argument("--seed-gate", type=int, default=456)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-steps", type=int, default=4096)
    parser.add_argument("--eval-freq", type=int, default=200_000)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--cov-gain-values", type=str, default="0.011,0.012,0.013,0.014")
    parser.add_argument("--cov-gain-late-values", type=str, default="0.008,0.012")
    parser.add_argument("--oob-grace-values", type=str, default="2,3,4")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only; do not run.")
    return parser.parse_args()


def _parse_float_list(csv_text: str) -> list[float]:
    if (csv_text or "").strip().lower() in {"", "none", "null"}:
        return []
    vals = []
    for tok in csv_text.split(","):
        t = tok.strip()
        if not t:
            continue
        vals.append(float(t))
    return vals


def _parse_int_list(csv_text: str) -> list[int]:
    if (csv_text or "").strip().lower() in {"", "none", "null"}:
        return []
    vals = []
    for tok in csv_text.split(","):
        t = tok.strip()
        if not t:
            continue
        vals.append(int(t))
    return vals


def _fmt_num(x: float) -> str:
    txt = f"{x:.6f}".rstrip("0").rstrip(".")
    return txt.replace("-", "m").replace(".", "p")


def _run(cmd: list[str], dry_run: bool) -> None:
    print("[sweep]$", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _gate_score(row: dict) -> tuple[float, float, float, float]:
    crash_count = float(row.get("crash_count", 0))
    success_count = float(row.get("success_count", 0))
    coverage_mean = float(row.get("coverage_mean", 0.0))
    time95 = float(row.get("time_to_95_mean", 1e9))
    return (-crash_count, success_count, coverage_mean, -time95)


def main() -> None:
    args = parse_args()
    base_model = Path(args.base_model)
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    cfg = get_preset(args.preset)
    base_overrides = {
        "scan_cov_late_thresh": float(getattr(cfg, "scan_cov_late_thresh", 0.85)),
        "scan_k_cov_gain_late": float(getattr(cfg, "scan_k_cov_gain_late", 0.01)),
        "scan_k_cov_stall": 0.0,
        "scan_oob_grace_steps": int(getattr(cfg, "scan_oob_grace_steps", 3)),
        "scan_k_cov_gain": float(getattr(cfg, "scan_k_cov_gain", 0.0125)),
        "scan_debug_oob": False,
    }

    candidates: list[dict] = []
    candidates.append({"name": "baseline_nofinetune", "overrides": {}})
    for v in _parse_float_list(args.cov_gain_values):
        candidates.append({"name": f"gain_{_fmt_num(v)}", "overrides": {"scan_k_cov_gain": float(v)}})
    for v in _parse_float_list(args.cov_gain_late_values):
        candidates.append({"name": f"late_{_fmt_num(v)}", "overrides": {"scan_k_cov_gain_late": float(v)}})
    for v in _parse_int_list(args.oob_grace_values):
        candidates.append({"name": f"grace_{v}", "overrides": {"scan_oob_grace_steps": int(v)}})

    results: list[dict] = []
    best_row: dict | None = None
    best_model_path = base_model

    for cand in candidates:
        name = str(cand["name"])
        merged = dict(base_overrides)
        merged.update(cand["overrides"])
        run_dir = run_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        model_for_gate = base_model

        if name != "baseline_nofinetune":
            train_cmd = [
                sys.executable,
                "-m",
                "scripts.train",
                "--run-dir",
                str(run_dir),
                "--load-model",
                str(base_model),
                "--total-timesteps",
                str(int(args.timesteps)),
                "--seed",
                str(int(args.seed_train)),
                "--task",
                "scan",
                "--preset",
                args.preset,
                "--device",
                args.device,
                "--n-steps",
                str(int(args.n_steps)),
                "--eval-freq",
                str(int(args.eval_freq)),
                "--learning-rate",
                str(float(args.learning_rate)),
                "--ent-coef",
                str(float(args.ent_coef)),
            ]
            for k, v in merged.items():
                train_cmd.extend(["--cfg-override", f"{k}={v}"])
            _run(train_cmd, args.dry_run)
            model_for_gate = run_dir / "best_model.zip"

        gate_json = run_dir / "gates" / f"{name}_gate_seed{int(args.seed_gate)}.json"
        gate_cmd = [
            sys.executable,
            "-m",
            "scripts.eval",
            "--model",
            str(model_for_gate),
            "--task",
            "scan",
            "--preset",
            args.preset,
            "--episodes",
            str(int(args.episodes)),
            "--seed",
            str(int(args.seed_gate)),
            "--device",
            args.device,
            "--json-out",
            str(gate_json),
        ]
        for k, v in merged.items():
            gate_cmd.extend(["--cfg-override", f"{k}={v}"])
        _run(gate_cmd, args.dry_run)

        if args.dry_run:
            continue
        gate = json.loads(gate_json.read_text(encoding="utf-8"))
        row = {
            "candidate": name,
            "model": str(model_for_gate),
            "gate_json": str(gate_json),
            "overrides": merged,
            "success_count": int(gate.get("success_count", 0)),
            "crash_count": int(gate.get("crash_count", 0)),
            "coverage_mean": float(gate.get("coverage_mean", 0.0)),
            "coverage_min": float(gate.get("coverage_min", 0.0)),
            "time_to_95_mean": float(gate.get("time_to_95_mean", 0.0)),
        }
        results.append(row)
        if best_row is None or _gate_score(row) > _gate_score(best_row):
            best_row = row
            best_model_path = model_for_gate

    if args.dry_run:
        return

    results_path = run_root / "sweep_gate_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if best_row is not None:
        best_out = run_root / "best_by_gate_model.zip"
        shutil.copy2(best_model_path, best_out)
        best_summary = {
            "base_model": str(base_model),
            "preset": args.preset,
            "timesteps": int(args.timesteps),
            "episodes": int(args.episodes),
            "seed_train": int(args.seed_train),
            "seed_gate": int(args.seed_gate),
            "best_row": best_row,
            "best_model": str(best_out),
            "all_results": str(results_path),
        }
        (run_root / "best_by_gate_summary.json").write_text(json.dumps(best_summary, indent=2), encoding="utf-8")
        print("[sweep] best candidate:", json.dumps(best_row, indent=2))
        print(f"[sweep] wrote best model: {best_out}")


if __name__ == "__main__":
    main()
