from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PAIRED_CONFIGS: list[tuple[float, float]] = [
    (4.0, 1.2),
    (4.5, 1.35),
    (5.0, 1.5),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 3-arm bridge speed sweep (paired step_len_m + vxy_cap) and rank results."
    )
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--conn", type=str, default="udp:127.0.0.1:14550")
    parser.add_argument("--model", type=str, default="auto")
    parser.add_argument("--task", type=str, default="scan")
    parser.add_argument("--preset", type=str, default="A2")
    parser.add_argument("--bounds", type=float, nargs=2, default=(40.0, 40.0), metavar=("W", "H"))
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--alt", type=float, default=10.0)
    parser.add_argument("--duration-s", type=float, default=180.0)
    parser.add_argument("--rate-hz", type=float, default=5.0)
    parser.add_argument("--policy-hz", type=float, default=2.0)
    parser.add_argument("--policy-hz-max", type=float, default=3.0)
    parser.add_argument("--dry-run", type=int, default=0, choices=(0, 1))
    parser.add_argument("--sitl-recommended", type=int, default=1, choices=(0, 1))
    parser.add_argument("--sitl-recommended-source", type=str, default="latest")
    parser.add_argument("--auto-pin", type=int, default=0, choices=(0, 1))
    parser.add_argument("--scale-bucket", type=str, default="scale2", choices=("scale1", "scale2"))
    parser.add_argument("--out-dir", type=str, default="")
    return parser.parse_args(argv)


def _num_or_nan(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


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
    out = Path("runs") / "bridge_speed_sweep" / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def _selected_profile_for_scale(scale: float) -> str:
    return "patch7_scale2" if float(scale) >= 2.0 else "patch5_default"


def _build_bridge_cmd(args: argparse.Namespace, step_len_m: float, vxy_cap: float) -> list[str]:
    rate_hz = max(1e-3, float(args.rate_hz))
    step_budget = int(max(1, math.ceil(float(args.duration_s) * rate_hz)))
    cmd = [
        sys.executable,
        "-m",
        "scripts.ardupilot_bridge",
        "--dry-run",
        str(int(args.dry_run)),
        "--sitl-recommended",
        str(int(args.sitl_recommended)),
        "--sitl-recommended-source",
        str(args.sitl_recommended_source),
        "--model",
        str(args.model),
        "--task",
        str(args.task),
        "--preset",
        str(args.preset),
        "--scan-path-len-scale",
        str(float(args.scale)),
        "--connection",
        str(args.conn),
        "--bounds-m",
        str(float(args.bounds[0])),
        str(float(args.bounds[1])),
        "--margin-m",
        str(float(args.margin)),
        "--alt-m",
        str(float(args.alt)),
        "--rate-hz",
        str(float(args.rate_hz)),
        "--policy-hz",
        str(float(args.policy_hz)),
        "--policy-hz-max",
        str(float(args.policy_hz_max)),
        "--steps",
        str(step_budget),
        "--step-len-m",
        str(float(step_len_m)),
        "--vxy-cap",
        str(float(vxy_cap)),
        "--cfg-override",
        f"scan_max_steps={step_budget}",
    ]
    return cmd


def _rank_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    # Primary constraint proxy: avoid raising OOB/clamps. Then maximize coverage speed proxy.
    oob = _num_or_nan(row.get("oob_events"))
    clamp_rate = _num_or_nan(row.get("clamp_rate_per_min"))
    cov = _num_or_nan(row.get("final_coverage"))
    p95_dist = _num_or_nan(row.get("p95_dist_to_target"))

    if math.isnan(oob):
        oob = 1e9
    if math.isnan(clamp_rate):
        clamp_rate = 1e9
    if math.isnan(cov):
        cov = -1e9
    if math.isnan(p95_dist):
        p95_dist = 1e9

    return (-oob, -clamp_rate, cov, -p95_dist)


def _fmt(x: Any, digits: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return "nan"
    if math.isnan(v):
        return "nan"
    return f"{v:.{digits}f}"


def _validate_arm(row: dict[str, Any]) -> tuple[int, str]:
    if not str(row.get("summary_path", "")).strip():
        return 0, "missing_summary_path_in_stdout"
    if int(row.get("ok", 0)) != 1:
        return 0, f"returncode_nonzero:{int(row.get('bridge_returncode', -1))}"
    if int(row.get("valid_metrics", 0)) != 1:
        return 0, "metrics_invalid"
    return 1, ""


def _run_one(args: argparse.Namespace, run_idx: int, step_len_m: float, vxy_cap: float, out_dir: Path) -> dict[str, Any]:
    cmd = _build_bridge_cmd(args, step_len_m=step_len_m, vxy_cap=vxy_cap)
    print(
        f"[speed_sweep] run {run_idx}/3 step_len_m={step_len_m:.2f} vxy_cap={vxy_cap:.2f} duration_s={float(args.duration_s):.1f}"
    )
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    summary_path = _extract_bridge_summary_path(proc.stdout)
    run_dir = summary_path.parent if summary_path is not None else None

    row: dict[str, Any] = {
        "run_idx": int(run_idx),
        "params": {"step_len_m": float(step_len_m), "vxy_cap": float(vxy_cap)},
        "bridge_cmd": cmd,
        "bridge_returncode": int(proc.returncode),
        "run_dir": "" if run_dir is None else str(run_dir),
        "summary_path": "" if summary_path is None else str(summary_path),
        "ok": 0,
        "valid_metrics": 0,
        "valid_for_ranking": 0,
        "final_coverage": float("nan"),
        "clamp_rate_per_min": float("nan"),
        "oob_events": float("nan"),
        "p95_dist_to_target": float("nan"),
        "step_len": float("nan"),
        "vxy_cap": float("nan"),
        "accept_radius": float("nan"),
        "accept_radius_m": float("nan"),
        "margin_m": float("nan"),
        "policy_hz": float("nan"),
        "policy_hz_max": float("nan"),
        "bounds": [float(args.bounds[0]), float(args.bounds[1])],
        "scale": float(args.scale),
    }

    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / f"run_{run_idx:02d}_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (logs_dir / f"run_{run_idx:02d}_stderr.txt").write_text(proc.stderr, encoding="utf-8")

    if summary_path is None:
        row["error"] = "bridge_summary_path_not_found_in_stdout"
        row["bridge_stdout_tail"] = proc.stdout.splitlines()[-40:]
        row["bridge_stderr_tail"] = proc.stderr.splitlines()[-40:]
        print(f"[speed_sweep] run {run_idx} failed: summary path not found in bridge output")
        return row

    if not summary_path.exists():
        row["error"] = "bridge_summary_path_missing_on_disk"
        row["bridge_stdout_tail"] = proc.stdout.splitlines()[-40:]
        row["bridge_stderr_tail"] = proc.stderr.splitlines()[-40:]
        print(f"[speed_sweep] run {run_idx} failed: summary.json missing at {summary_path}")
        return row

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        row["error"] = f"bridge_summary_unreadable:{exc.__class__.__name__}"
        row["bridge_stdout_tail"] = proc.stdout.splitlines()[-40:]
        row["bridge_stderr_tail"] = proc.stderr.splitlines()[-40:]
        print(f"[speed_sweep] run {run_idx} failed: summary.json unreadable at {summary_path}")
        return row

    row["ok"] = int(proc.returncode == 0)
    row["final_coverage"] = _num_or_nan(summary.get("final_coverage", summary.get("coverage_mean")))
    row["clamp_rate_per_min"] = _num_or_nan(summary.get("clamp_rate_per_min"))
    row["oob_events"] = _num_or_nan(summary.get("oob_events"))
    row["p95_dist_to_target"] = _num_or_nan(summary.get("p95_dist_to_target"))
    row["step_len"] = _num_or_nan(summary.get("step_len", summary.get("final_step_len_m")))
    row["vxy_cap"] = _num_or_nan(summary.get("vxy_cap", summary.get("final_vxy_cap")))
    row["accept_radius"] = _num_or_nan(summary.get("accept_radius", summary.get("final_accept_radius_m")))
    row["accept_radius_m"] = _num_or_nan(summary.get("final_accept_radius_m", summary.get("accept_radius")))
    row["margin_m"] = _num_or_nan(summary.get("margin_m", args.margin))
    row["policy_hz"] = _num_or_nan(summary.get("policy_hz", args.policy_hz))
    row["policy_hz_max"] = _num_or_nan(summary.get("policy_hz_max", args.policy_hz_max))
    if isinstance(summary.get("bounds"), list) and len(summary["bounds"]) == 2:
        row["bounds"] = [float(summary["bounds"][0]), float(summary["bounds"][1])]
    row["scale"] = _num_or_nan(summary.get("scale", args.scale))
    row["exit_reason"] = str(summary.get("exit_reason", ""))
    row["duration_s"] = _num_or_nan(summary.get("duration_s"))
    row["valid_metrics"] = int((_num_or_nan(row.get("final_coverage")) > 0.0) or (_num_or_nan(row.get("duration_s")) > 0.0))
    row["valid_for_ranking"], reason = _validate_arm(row)
    if int(row["valid_for_ranking"]) == 0 and "error" not in row:
        row["error"] = str(reason or "invalid_arm")
    print(
        "[speed_sweep] run "
        f"{run_idx} metrics: cov={_fmt(row['final_coverage'])} clamp/min={_fmt(row['clamp_rate_per_min'])} "
        f"oob={_fmt(row['oob_events'], digits=0)} p95d={_fmt(row['p95_dist_to_target'])} "
        f"valid={int(row['valid_for_ranking'])}"
    )
    return row


def _write_winner_opt_summary(args: argparse.Namespace, winner: dict[str, Any], out_dir: Path) -> Path:
    step_len_m = _num_or_nan(winner.get("step_len"))
    if math.isnan(step_len_m):
        params = winner.get("params", {}) if isinstance(winner.get("params", {}), dict) else {}
        step_len_m = _num_or_nan(params.get("step_len_m"))
    vxy_cap = _num_or_nan(winner.get("vxy_cap"))
    if math.isnan(vxy_cap):
        params = winner.get("params", {}) if isinstance(winner.get("params", {}), dict) else {}
        vxy_cap = _num_or_nan(params.get("vxy_cap"))
    accept_radius_m = _num_or_nan(winner.get("accept_radius_m"))
    if math.isnan(accept_radius_m):
        accept_radius_m = 1.25 if float(args.scale) >= 2.0 else 0.75
    margin_m = _num_or_nan(winner.get("margin_m"))
    if math.isnan(margin_m):
        margin_m = float(args.margin)
    policy_hz = _num_or_nan(winner.get("policy_hz"))
    if math.isnan(policy_hz):
        policy_hz = float(args.policy_hz)
    policy_hz_max = _num_or_nan(winner.get("policy_hz_max"))
    if math.isnan(policy_hz_max):
        policy_hz_max = float(args.policy_hz_max)
    bounds = winner.get("bounds")
    if not (isinstance(bounds, list) and len(bounds) == 2):
        bounds = [float(args.bounds[0]), float(args.bounds[1])]
    scale = _num_or_nan(winner.get("scale"))
    if math.isnan(scale):
        scale = float(args.scale)
    selected_profile = _selected_profile_for_scale(scale)

    config = {
        "step_len_m": float(step_len_m),
        "accept_radius_m": float(accept_radius_m),
        "vxy_cap": float(vxy_cap),
        "margin_m": float(margin_m),
        "policy_hz": float(policy_hz),
        "policy_hz_max": float(policy_hz_max),
        "bounds": [float(bounds[0]), float(bounds[1])],
        "scale": float(scale),
        "scan_path_len_scale": float(scale),
    }
    winner_metrics = {
        "final_coverage": _num_or_nan(winner.get("final_coverage")),
        "clamp_rate_per_min": _num_or_nan(winner.get("clamp_rate_per_min")),
        "oob_events": _num_or_nan(winner.get("oob_events")),
        "p95_dist_to_target": _num_or_nan(winner.get("p95_dist_to_target")),
    }
    payload = {
        "metadata": {
            "selected_profile": str(selected_profile),
            "scan_path_len_scale": float(scale),
            "source": "bridge_speed_sweep",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "ranked": [
            {
                "rank": 1,
                "arm": "S_speed_winner",
                "config": config,
                "metrics": winner_metrics,
            }
        ],
        "config": {"scan_path_len_scale": float(scale)},
    }
    winner_path = out_dir / "winner_opt_summary.json"
    winner_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return winner_path


def _auto_pin_winner(args: argparse.Namespace, winner: dict[str, Any], out_dir: Path) -> None:
    winner_path = _write_winner_opt_summary(args=args, winner=winner, out_dir=out_dir)
    bless_cmd = [
        sys.executable,
        "-m",
        "scripts.bless_ardupilot_defaults",
        "--scale-bucket",
        str(args.scale_bucket),
        "--source",
        f"path:{winner_path}",
    ]
    proc = subprocess.run(bless_cmd, capture_output=True, text=True, check=False)
    (out_dir / "auto_pin_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (out_dir / "auto_pin_stderr.txt").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(
            "[sweep] auto-pin failed. "
            f"cmd={' '.join(bless_cmd)} returncode={proc.returncode}"
        )
    pinned_path = Path("runs") / "production_ardupilot_defaults" / f"{args.scale_bucket}_opt_summary.json"
    print(f"[sweep] pinned winner to {pinned_path}")


def _print_ranked(rows: list[dict[str, Any]]) -> None:
    print("\n[speed_sweep] ranked results (best first)")
    print("rank run step_len_m vxy_cap final_cov clamp/min oob p95_dist ok")
    for rank, row in enumerate(rows, start=1):
        params = row.get("params", {}) if isinstance(row.get("params", {}), dict) else {}
        print(
            f"{rank:>4d} "
            f"{int(row.get('run_idx', 0)):>3d} "
            f"{float(params.get('step_len_m', float('nan'))):>10.2f} "
            f"{float(params.get('vxy_cap', float('nan'))):>7.2f} "
            f"{_fmt(row.get('final_coverage'), digits=4):>9} "
            f"{_fmt(row.get('clamp_rate_per_min'), digits=4):>9} "
            f"{_fmt(row.get('oob_events'), digits=0):>3} "
            f"{_fmt(row.get('p95_dist_to_target'), digits=4):>8} "
            f"{int(row.get('ok', 0)):>2d}"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = _resolve_out_dir(args)

    rows: list[dict[str, Any]] = []
    for run_idx, (step_len_m, vxy_cap) in enumerate(PAIRED_CONFIGS, start=1):
        rows.append(_run_one(args=args, run_idx=run_idx, step_len_m=step_len_m, vxy_cap=vxy_cap, out_dir=out_dir))

    invalid_rows = [r for r in rows if int(r.get("valid_for_ranking", 0)) != 1]
    ranked: list[dict[str, Any]] = []
    ranking_performed = False
    if invalid_rows:
        print("[speed_sweep] ERROR: one or more arms are missing/invalid; ranking and pinning are disabled.")
        for bad in invalid_rows:
            reason = str(bad.get("error", "")).strip() or "unknown"
            print(
                f"[speed_sweep] invalid arm run={int(bad.get('run_idx', 0))} "
                f"rc={int(bad.get('bridge_returncode', -1))} "
                f"run_dir={bad.get('run_dir', '')} "
                f"summary_path={bad.get('summary_path', '')} "
                f"error={reason}"
            )
    else:
        ranked = sorted(rows, key=_rank_key, reverse=True)
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = int(rank)
        ranking_performed = True
        _print_ranked(ranked)

    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "out_dir": str(out_dir),
        "scale": float(args.scale),
        "conn": str(args.conn),
        "bounds": [float(args.bounds[0]), float(args.bounds[1])],
        "alt": float(args.alt),
        "duration_s": float(args.duration_s),
        "dry_run": int(args.dry_run),
        "paired_configs": [{"step_len_m": s, "vxy_cap": v} for (s, v) in PAIRED_CONFIGS],
        "rank_policy": "oob_events asc, clamp_rate_per_min asc, final_coverage desc, p95_dist_to_target asc",
        "results": rows,
        "ranked": ranked,
        "best": ranked[0] if ranked else {},
        "ranking_performed": int(ranking_performed),
        "invalid_arm_count": int(len(invalid_rows)),
    }

    out_path = out_dir / "sweep_summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[speed_sweep] wrote: {out_path}")

    if int(args.auto_pin) == 1 and ranked and (not invalid_rows):
        _auto_pin_winner(args=args, winner=ranked[0], out_dir=out_dir)
    elif int(args.auto_pin) == 1:
        print("[sweep] auto-pin skipped because not all 3 arms succeeded with valid metrics.")

    failures = sum(1 for r in rows if int(r.get("valid_for_ranking", 0)) != 1)
    return 1 if failures > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
