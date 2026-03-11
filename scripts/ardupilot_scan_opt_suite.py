from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from scripts import ardupilot_scan_gate_suite as gate_suite


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 4-arm ArduPilot bridge optimization suites (A/B/C/D).")
    parser.add_argument("--sitl-recommended", type=int, default=1, choices=(0, 1))
    parser.add_argument("--sitl-recommended-source", type=str, default="latest")
    parser.add_argument("--conn", type=str, default="udp:127.0.0.1:14550")
    parser.add_argument("--ack-verbosity", type=str, default="important", choices=("all", "important"))
    parser.add_argument("--model", type=str, default="auto")
    parser.add_argument("--task", type=str, default="scan")
    parser.add_argument("--preset", type=str, default="A2")
    parser.add_argument("--scan-path-len-scale", type=float, default=1.0)
    parser.add_argument("--bounds-m", type=float, nargs=2, default=(40.0, 40.0), metavar=("W", "H"))
    parser.add_argument("--margin-m", type=float, default=2.0)
    parser.add_argument("--alt-m", type=float, default=10.0)
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--rate-hz", type=float, default=5.0)
    parser.add_argument("--policy-hz", type=float, default=2.0)
    parser.add_argument("--coverage-hz", type=float, default=5.0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--dry-run", type=int, default=1, choices=(0, 1))
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--stop-cov", type=float, default=0.95)
    parser.add_argument("--step-len-m", type=float, default=None)
    parser.add_argument("--accept-radius-m", type=float, default=None)
    parser.add_argument("--vxy-cap", type=float, default=None)
    parser.add_argument("--lane-kp", type=float, default=0.4)
    parser.add_argument("--lane-dir-mode", type=str, default="auto", choices=("auto", "fixed"))
    parser.add_argument("--cfg-override", action="append", default=[])
    return parser.parse_args(argv)


def _resolve_out_dir(args: argparse.Namespace) -> Path:
    if (args.out_dir or "").strip():
        out = Path(args.out_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        return out
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path("runs") / "ardupilot_scan_opt_suite" / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def _arm_specs() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("A_baseline", {"lookahead_enable": 1, "step_sched_enable": 1, "lane_keep_enable": 0, "adaptive_tracking": 0}),
        ("B_adaptive", {"lookahead_enable": 1, "step_sched_enable": 1, "lane_keep_enable": 0, "adaptive_tracking": 1}),
        ("C_lane", {"lookahead_enable": 1, "step_sched_enable": 1, "lane_keep_enable": 1, "adaptive_tracking": 0}),
        ("D_all", {"lookahead_enable": 1, "step_sched_enable": 1, "lane_keep_enable": 1, "adaptive_tracking": 1}),
    ]


def _build_suite_argv(args: argparse.Namespace, arm_out_dir: Path, overrides: dict[str, Any]) -> list[str]:
    argv = [
        "--sitl-recommended",
        str(int(args.sitl_recommended)),
        "--sitl-recommended-source",
        str(args.sitl_recommended_source),
        "--conn",
        str(args.conn),
        "--ack-verbosity",
        str(args.ack_verbosity),
        "--model",
        str(args.model),
        "--task",
        str(args.task),
        "--preset",
        str(args.preset),
        "--scan-path-len-scale",
        str(float(args.scan_path_len_scale)),
        "--bounds-m",
        str(float(args.bounds_m[0])),
        str(float(args.bounds_m[1])),
        "--margin-m",
        str(float(args.margin_m)),
        "--alt-m",
        str(float(args.alt_m)),
        "--duration-s",
        str(float(args.duration_s)),
        "--rate-hz",
        str(float(args.rate_hz)),
        "--policy-hz",
        str(float(args.policy_hz)),
        "--coverage-hz",
        str(float(args.coverage_hz)),
        "--stop-cov",
        str(float(args.stop_cov)),
        "--runs",
        str(int(args.runs)),
        "--ab",
        "0",
        "--adaptive-tracking",
        str(int(overrides["adaptive_tracking"])),
        "--lookahead-enable",
        str(int(overrides["lookahead_enable"])),
        "--step-sched-enable",
        str(int(overrides["step_sched_enable"])),
        "--lane-keep-enable",
        str(int(overrides["lane_keep_enable"])),
        "--lane-kp",
        str(float(args.lane_kp)),
        "--lane-dir-mode",
        str(args.lane_dir_mode),
        "--dry-run",
        str(int(args.dry_run)),
        "--out-dir",
        str(arm_out_dir),
    ]
    if args.step_len_m is not None:
        argv.extend(["--step-len-m", str(float(args.step_len_m))])
    if args.accept_radius_m is not None:
        argv.extend(["--accept-radius-m", str(float(args.accept_radius_m))])
    if args.vxy_cap is not None:
        argv.extend(["--vxy-cap", str(float(args.vxy_cap))])
    for ov in args.cfg_override:
        argv.extend(["--cfg-override", str(ov)])
    return argv


def _num_or_nan(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def _arm_stats(summary: dict[str, Any]) -> dict[str, Any]:
    stats_by_arm = summary.get("stats_by_arm", {})
    if "S" in stats_by_arm:
        return dict(stats_by_arm["S"])
    if stats_by_arm:
        return dict(next(iter(stats_by_arm.values())))
    return {}


def _rank_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    cov95 = _num_or_nan(row.get("cov95_hit_rate"))
    cov = _num_or_nan(row.get("coverage_mean"))
    primary = cov95 if not math.isnan(cov95) else cov
    p95_dist = _num_or_nan(row.get("p95_dist_to_target"))
    clamp_rate = _num_or_nan(row.get("clamp_rate_per_min"))
    oob = _num_or_nan(row.get("oob_events"))
    if math.isnan(primary):
        primary = -1e9
    if math.isnan(p95_dist):
        p95_dist = 1e9
    if math.isnan(clamp_rate):
        clamp_rate = 1e9
    if math.isnan(oob):
        oob = 1e9
    return (primary, -p95_dist, -clamp_rate, -oob)


def _fmt(x: Any) -> str:
    v = _num_or_nan(x)
    if math.isnan(v):
        return "nan"
    return f"{v:.3f}"


def _selected_profile_for_scale(scale: float) -> str:
    return "patch7_scale2" if float(scale) >= 2.0 else "patch5_default"


def main() -> None:
    args = parse_args()
    out_dir = _resolve_out_dir(args)
    arm_results: list[dict[str, Any]] = []
    failures = 0

    for arm_name, ov in _arm_specs():
        arm_out = out_dir / arm_name
        arm_out.mkdir(parents=True, exist_ok=True)
        suite_args = gate_suite.parse_args(_build_suite_argv(args, arm_out, ov))
        suite_summary, suite_path, rc = gate_suite.run_suite(suite_args)
        if int(rc) != 0:
            failures += 1
        stats = _arm_stats(suite_summary)
        arm_results.append(
            {
                "arm": arm_name,
                "config": ov,
                "returncode": int(rc),
                "suite_summary_path": str(suite_path),
                "metrics": {
                    "coverage_mean": _num_or_nan(stats.get("coverage_mean")),
                    "cov95_hit_rate": _num_or_nan(stats.get("cov95_hit_rate")),
                    "p95_dist_to_target": _num_or_nan(stats.get("p95_dist_to_target")),
                    "clamp_rate_per_min": _num_or_nan(stats.get("clamp_rate_per_min")),
                    "oob_events": _num_or_nan(stats.get("oob_events")),
                    "mean_abs_cross_track": _num_or_nan(stats.get("mean_abs_cross_track")),
                },
            }
        )

    ranked = sorted(arm_results, key=lambda r: _rank_key(r["metrics"]), reverse=True)
    for i, row in enumerate(ranked, start=1):
        row["rank"] = int(i)

    opt_summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "out_dir": str(out_dir),
        "metadata": {
            "scan_path_len_scale": float(args.scan_path_len_scale),
            "selected_profile": _selected_profile_for_scale(float(args.scan_path_len_scale)),
            "bounds_m": [float(args.bounds_m[0]), float(args.bounds_m[1])],
            "bounds-m": [float(args.bounds_m[0]), float(args.bounds_m[1])],
            "duration_s": float(args.duration_s),
            "duration-s": float(args.duration_s),
            "runs": int(args.runs),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "config": {
            "conn": str(args.conn),
            "model": str(args.model),
            "preset": str(args.preset),
            "scan_path_len_scale": float(args.scan_path_len_scale),
            "bounds_m": [float(args.bounds_m[0]), float(args.bounds_m[1])],
            "duration_s": float(args.duration_s),
            "runs": int(args.runs),
            "dry_run": int(args.dry_run),
        },
        "arms": arm_results,
        "ranked": ranked,
        "failures": int(failures),
    }
    out_path = out_dir / "opt_summary.json"
    out_path.write_text(json.dumps(opt_summary, indent=2), encoding="utf-8")

    print("rank,arm,cov95,coverage,p95_dist,clamp_rate,oob,cross_track")
    for row in ranked:
        m = row["metrics"]
        print(
            f"{row['rank']},{row['arm']},{_fmt(m.get('cov95_hit_rate'))},"
            f"{_fmt(m.get('coverage_mean'))},{_fmt(m.get('p95_dist_to_target'))},"
            f"{_fmt(m.get('clamp_rate_per_min'))},{_fmt(m.get('oob_events'))},"
            f"{_fmt(m.get('mean_abs_cross_track'))}"
        )
    print(f"[scan_opt_suite] opt_summary: {out_path}")

    raise SystemExit(1 if failures > 0 else 0)


if __name__ == "__main__":
    main()
