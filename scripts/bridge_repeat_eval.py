from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeat one ArduPilot bridge config N times and summarize stability.")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--dry-run", type=int, default=0, choices=(0, 1))
    parser.add_argument("--conn", type=str, default="udpin:0.0.0.0:14551")
    parser.add_argument("--model", type=str, default="auto")
    parser.add_argument("--task", type=str, default="scan")
    parser.add_argument("--preset", type=str, default="A2")
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--bounds", type=float, nargs=2, default=(40.0, 40.0), metavar=("W", "H"))
    parser.add_argument("--alt", type=float, default=8.0)
    parser.add_argument("--auto-guided", type=int, default=1, choices=(0, 1))
    parser.add_argument("--auto-arm", type=int, default=1, choices=(0, 1))
    parser.add_argument("--auto-takeoff-m", type=float, default=8.0)
    parser.add_argument("--strict-lawnmower", type=int, default=1, choices=(0, 1))
    parser.add_argument("--return-land-on-complete", type=int, default=1, choices=(0, 1))
    parser.add_argument("--yaw-mode", type=str, default="fixed", choices=("fixed", "face-vel", "none"))
    parser.add_argument("--yaw-rate-max-dps", type=float, default=90.0)
    parser.add_argument("--use-vel-caps", type=int, default=0, choices=(0, 1))
    parser.add_argument("--step-len-m", type=float, default=7.0)
    parser.add_argument("--accept-radius-m", type=float, default=0.9)
    parser.add_argument("--policy-hz", type=float, default=2.0)
    parser.add_argument("--coverage-hz", type=float, default=1.0)
    parser.add_argument("--camera-hfov-deg", type=float, default=151.5)
    parser.add_argument("--camera-vfov-deg", type=float, default=131.3)
    parser.add_argument("--footprint-model", type=str, default="circle_min", choices=("ellipse", "circle_min", "circle_area"))
    parser.add_argument("--footprint-fov-scale", type=float, default=0.35)
    parser.add_argument("--footprint-min-passes", type=int, default=1)
    parser.add_argument("--footprint-count-dist-factor", type=float, default=0.7)
    parser.add_argument("--footprint-count-dist-min-m", type=float, default=0.5)
    parser.add_argument("--bridge-arg", action="append", default=[], help="Extra raw args forwarded to ardupilot_bridge.")
    return parser.parse_args(argv)


def _num_or_nan(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _fmt(x: Any, digits: int = 4) -> str:
    v = _num_or_nan(x)
    if math.isnan(v):
        return "nan"
    return f"{v:.{digits}f}"


def _extract_bridge_summary_path(stdout_text: str) -> Path | None:
    marker = "[bridge] summary:"
    for line in stdout_text.splitlines():
        if marker not in line:
            continue
        raw = line.split(marker, 1)[1].strip()
        if raw:
            return Path(raw)
    return None


def _resolve_out_dir(args: argparse.Namespace) -> Path:
    if (args.out_dir or "").strip():
        out = Path(args.out_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        return out
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path("runs") / "bridge_repeat_eval" / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_bridge_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "scripts.ardupilot_bridge",
        "--dry-run",
        str(int(args.dry_run)),
        "--model",
        str(args.model),
        "--task",
        str(args.task),
        "--preset",
        str(args.preset),
        "--scan-path-len-scale",
        str(float(args.scale)),
        "--bounds-m",
        str(float(args.bounds[0])),
        str(float(args.bounds[1])),
        "--alt-m",
        str(float(args.alt)),
        "--connection",
        str(args.conn),
        "--auto-guided",
        str(int(args.auto_guided)),
        "--auto-arm",
        str(int(args.auto_arm)),
        "--auto-takeoff-m",
        str(float(args.auto_takeoff_m)),
        "--strict-lawnmower",
        str(int(args.strict_lawnmower)),
        "--return-land-on-complete",
        str(int(args.return_land_on_complete)),
        "--yaw-mode",
        str(args.yaw_mode),
        "--yaw-rate-max-dps",
        str(float(args.yaw_rate_max_dps)),
        "--use-vel-caps",
        str(int(args.use_vel_caps)),
        "--step-len-m",
        str(float(args.step_len_m)),
        "--accept-radius-m",
        str(float(args.accept_radius_m)),
        "--policy-hz",
        str(float(args.policy_hz)),
        "--coverage-hz",
        str(float(args.coverage_hz)),
        "--camera-hfov-deg",
        str(float(args.camera_hfov_deg)),
        "--camera-vfov-deg",
        str(float(args.camera_vfov_deg)),
        "--footprint-model",
        str(args.footprint_model),
        "--footprint-fov-scale",
        str(float(args.footprint_fov_scale)),
        "--footprint-min-passes",
        str(int(args.footprint_min_passes)),
        "--footprint-count-dist-factor",
        str(float(args.footprint_count_dist_factor)),
        "--footprint-count-dist-min-m",
        str(float(args.footprint_count_dist_min_m)),
    ]
    for extra in args.bridge_arg:
        tok = str(extra).strip()
        if tok:
            cmd.append(tok)
    return cmd


def _run_one(args: argparse.Namespace, run_idx: int, out_dir: Path) -> dict[str, Any]:
    cmd = _build_bridge_cmd(args)
    print(f"[repeat_eval] run {run_idx}/{int(args.runs)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / f"run_{run_idx:02d}_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (logs_dir / f"run_{run_idx:02d}_stderr.txt").write_text(proc.stderr, encoding="utf-8")

    summary_path = _extract_bridge_summary_path(proc.stdout)
    row: dict[str, Any] = {
        "run_idx": int(run_idx),
        "bridge_cmd": cmd,
        "bridge_returncode": int(proc.returncode),
        "summary_path": "" if summary_path is None else str(summary_path),
        "run_dir": "" if summary_path is None else str(summary_path.parent),
        "ok": 0,
    }
    if summary_path is None:
        row["error"] = "summary_path_not_found"
        return row
    if not summary_path.exists():
        row["error"] = "summary_path_missing_on_disk"
        return row
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        row["error"] = f"summary_unreadable:{exc.__class__.__name__}"
        return row

    row["ok"] = int(proc.returncode == 0)
    keys = [
        "duration_s",
        "final_coverage_footprint_1x",
        "final_coverage_footprint_kx",
        "footprint_pass_count_mean_on_covered",
        "footprint_overlap_excess_ratio",
        "clamp_rate_per_min",
        "oob_events",
    ]
    for k in keys:
        row[k] = _num_or_nan(summary.get(k))
    row["exit_reason"] = str(summary.get("exit_reason", ""))
    return row


def _stats(rows: list[dict[str, Any]], key: str) -> tuple[float, float]:
    vals = [_num_or_nan(r.get(key)) for r in rows if int(r.get("ok", 0)) == 1]
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return float("nan"), float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = _resolve_out_dir(args)

    rows: list[dict[str, Any]] = []
    for i in range(1, int(args.runs) + 1):
        rows.append(_run_one(args, run_idx=i, out_dir=out_dir))

    print("\n[repeat_eval] per-run")
    print("run ok dur_s foot1x pass_mean overlap_excess clamp/min oob exit")
    for r in rows:
        print(
            f"{int(r.get('run_idx', 0)):>3d} "
            f"{int(r.get('ok', 0)):>2d} "
            f"{_fmt(r.get('duration_s'), 2):>6} "
            f"{_fmt(r.get('final_coverage_footprint_1x'), 4):>7} "
            f"{_fmt(r.get('footprint_pass_count_mean_on_covered'), 3):>9} "
            f"{_fmt(r.get('footprint_overlap_excess_ratio'), 3):>13} "
            f"{_fmt(r.get('clamp_rate_per_min'), 3):>9} "
            f"{_fmt(r.get('oob_events'), 0):>3} "
            f"{str(r.get('exit_reason', ''))}"
        )

    metric_keys = [
        "duration_s",
        "final_coverage_footprint_1x",
        "final_coverage_footprint_kx",
        "footprint_pass_count_mean_on_covered",
        "footprint_overlap_excess_ratio",
        "clamp_rate_per_min",
        "oob_events",
    ]
    summary_stats: dict[str, dict[str, float]] = {}
    print("\n[repeat_eval] mean/std over successful runs")
    for k in metric_keys:
        m, s = _stats(rows, k)
        summary_stats[k] = {"mean": m, "std": s}
        print(f"{k}: mean={_fmt(m, 4)} std={_fmt(s, 4)}")

    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": vars(args),
        "rows": rows,
        "stats": summary_stats,
    }
    out_path = out_dir / "repeat_summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[repeat_eval] wrote: {out_path}")

    failures = sum(1 for r in rows if int(r.get("ok", 0)) != 1)
    return 1 if failures > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())

