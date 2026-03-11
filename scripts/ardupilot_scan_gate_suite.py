from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from scripts import ardupilot_scan_gate as gate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A/B + multi-run ArduPilot scan gate suites.")
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
    parser.add_argument("--lookahead-enable", type=int, default=1, choices=(0, 1))
    parser.add_argument("--lookahead-time-s", type=float, default=0.6)
    parser.add_argument("--lookahead-cap-m", type=float, default=2.0)
    parser.add_argument("--lane-keep-enable", type=int, default=0, choices=(0, 1))
    parser.add_argument("--lane-dir-mode", type=str, default="auto", choices=("auto", "fixed"))
    parser.add_argument("--lane-kp", type=float, default=0.4)
    parser.add_argument("--lane-max-corr-m", type=float, default=1.5)
    parser.add_argument("--step-sched-enable", type=int, default=1, choices=(0, 1))
    parser.add_argument("--step-len-corner-mult", type=float, default=0.7)
    parser.add_argument("--step-len-boundary-mult", type=float, default=0.8)
    parser.add_argument("--boundary-near-m", type=float, default=3.0)
    parser.add_argument("--stop-cov", type=float, default=0.95)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--ab", type=int, default=1, choices=(0, 1))
    parser.add_argument("--adaptive-tracking", type=int, default=0, choices=(0, 1))
    parser.add_argument("--dry-run", type=int, default=1, choices=(0, 1))
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--target-refresh-mode", type=str, default="hold", choices=("hold", "always"))
    parser.add_argument("--use-vel-caps", type=int, default=1, choices=(0, 1))
    parser.add_argument("--vxy-cap", type=float, default=None)
    parser.add_argument("--vz-cap", type=float, default=0.5)
    parser.add_argument("--corner-angle-deg", type=float, default=60.0)
    parser.add_argument("--corner-slow-seconds", type=float, default=2.0)
    parser.add_argument("--corner-vxy-cap", type=float, default=0.8)
    parser.add_argument("--step-len-m", type=float, default=None)
    parser.add_argument("--accept-radius-m", type=float, default=None)
    parser.add_argument("--max-hold-s", type=float, default=3.0)
    parser.add_argument("--ignore-ekf", type=int, default=0, choices=(0, 1))
    parser.add_argument("--clamp-stop-count", type=int, default=100)
    parser.add_argument("--adapt-interval-s", type=float, default=5.0)
    parser.add_argument("--dist-p95-high", type=float, default=2.5)
    parser.add_argument("--progress-low", type=float, default=0.2)
    parser.add_argument("--policy-hz-min", type=float, default=1.0)
    parser.add_argument("--policy-hz-max", type=float, default=3.0)
    parser.add_argument("--policy-hz-stable-dist-p95", type=float, default=1.5)
    parser.add_argument("--policy-hz-stable-progress", type=float, default=0.4)
    parser.add_argument("--step-len-min", type=float, default=2.0)
    parser.add_argument("--step-len-max", type=float, default=8.0)
    parser.add_argument("--accept-radius-min", type=float, default=0.75)
    parser.add_argument("--accept-radius-max", type=float, default=2.0)
    parser.add_argument("--oob-recovery", type=int, default=1, choices=(0, 1))
    parser.add_argument("--oob-recovery-seconds", type=float, default=3.0)
    parser.add_argument("--oob-recovery-vxy-cap", type=float, default=0.6)
    parser.add_argument("--oob-recovery-accept-boost", type=float, default=0.5)
    parser.add_argument("--oob-clamp-fast-count", type=int, default=3)
    parser.add_argument("--oob-clamp-fast-window-s", type=float, default=5.0)
    parser.add_argument(
        "--cfg-override",
        action="append",
        default=[],
        help="Override preset fields with key=value (repeatable).",
    )
    return parser.parse_args(argv)


def _resolve_out_dir(args: argparse.Namespace) -> Path:
    if (args.out_dir or "").strip():
        out = Path(args.out_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        return out
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path("runs") / "ardupilot_scan_gate_suite" / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_gate_argv(args: argparse.Namespace, adaptive_tracking: int, out_path: Path) -> list[str]:
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
        "--lookahead-enable",
        str(int(args.lookahead_enable)),
        "--lookahead-time-s",
        str(float(args.lookahead_time_s)),
        "--lookahead-cap-m",
        str(float(args.lookahead_cap_m)),
        "--lane-keep-enable",
        str(int(args.lane_keep_enable)),
        "--lane-dir-mode",
        str(args.lane_dir_mode),
        "--lane-kp",
        str(float(args.lane_kp)),
        "--lane-max-corr-m",
        str(float(args.lane_max_corr_m)),
        "--step-sched-enable",
        str(int(args.step_sched_enable)),
        "--step-len-corner-mult",
        str(float(args.step_len_corner_mult)),
        "--step-len-boundary-mult",
        str(float(args.step_len_boundary_mult)),
        "--boundary-near-m",
        str(float(args.boundary_near_m)),
        "--stop-cov",
        str(float(args.stop_cov)),
        "--target-refresh-mode",
        str(args.target_refresh_mode),
        "--use-vel-caps",
        str(int(args.use_vel_caps)),
        "--vz-cap",
        str(float(args.vz_cap)),
        "--corner-angle-deg",
        str(float(args.corner_angle_deg)),
        "--corner-slow-seconds",
        str(float(args.corner_slow_seconds)),
        "--corner-vxy-cap",
        str(float(args.corner_vxy_cap)),
        "--max-hold-s",
        str(float(args.max_hold_s)),
        "--ignore-ekf",
        str(int(args.ignore_ekf)),
        "--clamp-stop-count",
        str(int(args.clamp_stop_count)),
        "--adaptive-tracking",
        str(int(adaptive_tracking)),
        "--adapt-interval-s",
        str(float(args.adapt_interval_s)),
        "--dist-p95-high",
        str(float(args.dist_p95_high)),
        "--progress-low",
        str(float(args.progress_low)),
        "--policy-hz-min",
        str(float(args.policy_hz_min)),
        "--policy-hz-max",
        str(float(args.policy_hz_max)),
        "--policy-hz-stable-dist-p95",
        str(float(args.policy_hz_stable_dist_p95)),
        "--policy-hz-stable-progress",
        str(float(args.policy_hz_stable_progress)),
        "--step-len-min",
        str(float(args.step_len_min)),
        "--step-len-max",
        str(float(args.step_len_max)),
        "--accept-radius-min",
        str(float(args.accept_radius_min)),
        "--accept-radius-max",
        str(float(args.accept_radius_max)),
        "--oob-recovery",
        str(int(args.oob_recovery)),
        "--oob-recovery-seconds",
        str(float(args.oob_recovery_seconds)),
        "--oob-recovery-vxy-cap",
        str(float(args.oob_recovery_vxy_cap)),
        "--oob-recovery-accept-boost",
        str(float(args.oob_recovery_accept_boost)),
        "--oob-clamp-fast-count",
        str(int(args.oob_clamp_fast_count)),
        "--oob-clamp-fast-window-s",
        str(float(args.oob_clamp_fast_window_s)),
        "--dry-run",
        str(int(args.dry_run)),
        "--out",
        str(out_path),
    ]
    if args.vxy_cap is not None:
        argv.extend(["--vxy-cap", str(float(args.vxy_cap))])
    if args.step_len_m is not None:
        argv.extend(["--step-len-m", str(float(args.step_len_m))])
    if args.accept_radius_m is not None:
        argv.extend(["--accept-radius-m", str(float(args.accept_radius_m))])
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


def _mean(vals: list[float]) -> float:
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _std(vals: list[float], mean_val: float) -> float:
    if not vals:
        return float("nan")
    return float(math.sqrt(sum((v - mean_val) * (v - mean_val) for v in vals) / len(vals)))


def _collect_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    coverage_vals = [_num_or_nan(r.get("coverage_mean")) for r in rows]
    cov95_vals = [_num_or_nan(r.get("cov95_hit")) for r in rows]
    time95_vals = [_num_or_nan(r.get("time_to_95_s")) for r in rows]
    p95_dist_vals = [_num_or_nan(r.get("p95_dist_to_target")) for r in rows]
    progress_vals = [_num_or_nan(r.get("mean_progress_per_s")) for r in rows]
    clamp_vals = [_num_or_nan(r.get("clamp_count")) for r in rows]
    clamp_rate_vals = [_num_or_nan(r.get("clamp_rate_per_min")) for r in rows]
    oob_vals = [_num_or_nan(r.get("oob_events")) for r in rows]
    cross_track_vals = [_num_or_nan(r.get("mean_abs_cross_track")) for r in rows]
    bad_vals = [_num_or_nan(r.get("adaptive_bad_count")) for r in rows]
    good_vals = [_num_or_nan(r.get("adaptive_good_count")) for r in rows]
    final_step_vals = [_num_or_nan(r.get("final_step_len_m")) for r in rows]
    final_accept_vals = [_num_or_nan(r.get("final_accept_radius_m")) for r in rows]
    final_vxy_vals = [_num_or_nan(r.get("final_vxy_cap")) for r in rows]

    def finite(arr: list[float]) -> list[float]:
        return [x for x in arr if not math.isnan(x)]

    cov_f = finite(coverage_vals)
    cov95_f = finite(cov95_vals)
    time95_f = finite(time95_vals)
    p95d_f = finite(p95_dist_vals)
    progress_f = finite(progress_vals)
    clamp_f = finite(clamp_vals)
    clamp_rate_f = finite(clamp_rate_vals)
    oob_f = finite(oob_vals)
    cross_track_f = finite(cross_track_vals)
    bad_f = finite(bad_vals)
    good_f = finite(good_vals)
    step_f = finite(final_step_vals)
    accept_f = finite(final_accept_vals)
    vxy_f = finite(final_vxy_vals)

    cov_m = _mean(cov_f)
    time95_m = _mean(time95_f)
    p95d_m = _mean(p95d_f)
    progress_m = _mean(progress_f)
    clamp_m = _mean(clamp_f)
    clamp_rate_m = _mean(clamp_rate_f)
    oob_m = _mean(oob_f)
    cross_track_m = _mean(cross_track_f)
    bad_m = _mean(bad_f)
    good_m = _mean(good_f)
    step_m = _mean(step_f)
    accept_m = _mean(accept_f)
    vxy_m = _mean(vxy_f)

    return {
        "n": int(len(rows)),
        "coverage_mean_mean": cov_m,
        "coverage_mean_std": _std(cov_f, cov_m),
        "cov95_hit_rate": _mean(cov95_f),
        "time_to_95_mean": time95_m,
        "time_to_95_std": _std(time95_f, time95_m),
        "p95_dist_to_target_mean": p95d_m,
        "p95_dist_to_target_std": _std(p95d_f, p95d_m),
        "progress_per_s_mean": progress_m,
        "progress_per_s_std": _std(progress_f, progress_m),
        "clamp_count_mean": clamp_m,
        "clamp_count_std": _std(clamp_f, clamp_m),
        "clamp_rate_per_min_mean": clamp_rate_m,
        "clamp_rate_per_min_std": _std(clamp_rate_f, clamp_rate_m),
        "oob_events_mean": oob_m,
        "oob_events_std": _std(oob_f, oob_m),
        "mean_abs_cross_track_mean": cross_track_m,
        "mean_abs_cross_track_std": _std(cross_track_f, cross_track_m),
        "adaptive_bad_count_mean": bad_m,
        "adaptive_bad_count_std": _std(bad_f, bad_m),
        "adaptive_good_count_mean": good_m,
        "adaptive_good_count_std": _std(good_f, good_m),
        "final_step_len_m_mean": step_m,
        "final_accept_radius_m_mean": accept_m,
        "final_vxy_cap_mean": vxy_m,
        # Flat aliases for consumers that expect non-suffixed names.
        "coverage_mean": cov_m,
        "p95_dist_to_target": p95d_m,
        "clamp_rate_per_min": clamp_rate_m,
        "oob_events": oob_m,
        "mean_abs_cross_track": cross_track_m,
    }


def _fmt(x: Any) -> str:
    try:
        xv = float(x)
    except Exception:
        return "nan"
    if math.isnan(xv):
        return "nan"
    return f"{xv:.3f}"


def _compute_recommendations(args: argparse.Namespace, stats_by_arm: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if "B" in stats_by_arm:
        base = stats_by_arm["B"]
    elif "S" in stats_by_arm:
        base = stats_by_arm["S"]
    elif "A" in stats_by_arm:
        base = stats_by_arm["A"]
    else:
        base = {}

    step_len = _num_or_nan(base.get("final_step_len_m_mean"))
    accept_radius = _num_or_nan(base.get("final_accept_radius_m_mean"))
    vxy_cap = _num_or_nan(base.get("final_vxy_cap_mean"))
    if math.isnan(step_len):
        step_len = float(args.step_len_m) if args.step_len_m is not None else 3.0
    if math.isnan(accept_radius):
        accept_radius = float(args.accept_radius_m) if args.accept_radius_m is not None else 1.0
    if math.isnan(vxy_cap):
        vxy_cap = float(args.vxy_cap) if args.vxy_cap is not None else 1.0

    margin_m = float(args.margin_m)
    oob_recovery = int(args.oob_recovery)

    p95_dist_mean = _num_or_nan(base.get("p95_dist_to_target_mean"))
    progress_mean = _num_or_nan(base.get("progress_per_s_mean"))
    clamp_rate_mean = _num_or_nan(base.get("clamp_rate_per_min_mean"))
    oob_events_mean = _num_or_nan(base.get("oob_events_mean"))

    if (not math.isnan(p95_dist_mean) and p95_dist_mean > 2.5) or (not math.isnan(progress_mean) and progress_mean < 0.2):
        step_len *= 0.85
        accept_radius *= 1.10
        vxy_cap *= 0.90
    if not math.isnan(clamp_rate_mean) and clamp_rate_mean > 5.0:
        margin_m += 1.0
        step_len *= 0.85
    if not math.isnan(oob_events_mean) and oob_events_mean > 0.0:
        oob_recovery = 1

    return {
        "step_len_m": float(step_len),
        "accept_radius_m": float(accept_radius),
        "vxy_cap": float(vxy_cap),
        "margin_m": float(margin_m),
        "oob_recovery": int(oob_recovery),
    }


def run_suite(args: argparse.Namespace) -> tuple[dict[str, Any], Path, int]:
    out_dir = _resolve_out_dir(args)

    run_count = max(1, int(args.runs))
    ab_mode = bool(int(args.ab))
    arms = [("A", 0), ("B", 1)] if ab_mode else [("S", int(args.adaptive_tracking))]

    records: list[dict[str, Any]] = []
    failures = 0
    for run_idx in range(1, run_count + 1):
        for arm_name, adaptive in arms:
            gate_out = out_dir / f"run_{run_idx:03d}" / f"arm_{arm_name}" / "gate_summary.json"
            gate_out.parent.mkdir(parents=True, exist_ok=True)
            gate_args = gate.parse_args(_build_gate_argv(args, adaptive, gate_out))
            gate_summary, gate_path, rc = gate.run_gate(gate_args)
            if int(rc) != 0:
                failures += 1
            records.append(
                {
                    "run_index": int(run_idx),
                    "arm": arm_name,
                    "adaptive_tracking": int(adaptive),
                    "returncode": int(rc),
                    "gate_summary_path": str(gate_path),
                    "gate_summary": gate_summary,
                }
            )

    by_arm_rows: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_arm_rows.setdefault(str(rec["arm"]), []).append(dict(rec["gate_summary"]))
    stats_by_arm = {arm: _collect_stats(rows) for arm, rows in by_arm_rows.items()}

    delta_b_minus_a: dict[str, Any] = {}
    if "A" in stats_by_arm and "B" in stats_by_arm:
        keys = [
            "coverage_mean_mean",
            "cov95_hit_rate",
            "time_to_95_mean",
            "p95_dist_to_target_mean",
            "clamp_count_mean",
            "progress_per_s_mean",
            "clamp_rate_per_min_mean",
            "oob_events_mean",
            "adaptive_bad_count_mean",
            "adaptive_good_count_mean",
        ]
        for k in keys:
            av = _num_or_nan(stats_by_arm["A"].get(k))
            bv = _num_or_nan(stats_by_arm["B"].get(k))
            delta_b_minus_a[k] = float("nan") if (math.isnan(av) or math.isnan(bv)) else float(bv - av)
    recommendations = _compute_recommendations(args, stats_by_arm)

    suite_summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "out_dir": str(out_dir),
        "config": {
            "conn": str(args.conn),
            "model": str(args.model),
            "preset": str(args.preset),
            "scan_path_len_scale": float(args.scan_path_len_scale),
            "bounds_m": [float(args.bounds_m[0]), float(args.bounds_m[1])],
            "duration_s": float(args.duration_s),
            "policy_hz": float(args.policy_hz),
            "coverage_hz": float(args.coverage_hz),
            "runs": int(run_count),
            "ab": int(args.ab),
            "dry_run": int(args.dry_run),
        },
        "records": records,
        "stats_by_arm": stats_by_arm,
        "delta_b_minus_a": delta_b_minus_a,
        "recommendations": recommendations,
        "failures": int(failures),
    }
    suite_path = out_dir / "suite_summary.json"
    suite_path.write_text(json.dumps(suite_summary, indent=2), encoding="utf-8")

    print("arm,n,cov_mean,cov95_rate,time95,p95_dist,clamp,bad,good")
    for arm in sorted(stats_by_arm.keys()):
        s = stats_by_arm[arm]
        print(
            f"{arm},{s['n']},{_fmt(s['coverage_mean_mean'])},{_fmt(s['cov95_hit_rate'])},"
            f"{_fmt(s['time_to_95_mean'])},{_fmt(s['p95_dist_to_target_mean'])},"
            f"{_fmt(s['clamp_count_mean'])},{_fmt(s['adaptive_bad_count_mean'])},"
            f"{_fmt(s['adaptive_good_count_mean'])}"
        )
    if delta_b_minus_a:
        print(
            "delta_B_minus_A,"
            f"{_fmt(delta_b_minus_a.get('coverage_mean_mean'))},"
            f"{_fmt(delta_b_minus_a.get('cov95_hit_rate'))},"
            f"{_fmt(delta_b_minus_a.get('time_to_95_mean'))},"
            f"{_fmt(delta_b_minus_a.get('p95_dist_to_target_mean'))},"
            f"{_fmt(delta_b_minus_a.get('clamp_count_mean'))},"
            f"{_fmt(delta_b_minus_a.get('adaptive_bad_count_mean'))},"
            f"{_fmt(delta_b_minus_a.get('adaptive_good_count_mean'))}"
        )
    print(
        "[scan_gate_suite] Recommended flags: "
        f"--step-len-m {recommendations['step_len_m']:.3f} "
        f"--accept-radius-m {recommendations['accept_radius_m']:.3f} "
        f"--vxy-cap {recommendations['vxy_cap']:.3f} "
        f"--margin-m {recommendations['margin_m']:.3f} "
        f"--oob-recovery {int(recommendations['oob_recovery'])}"
    )
    print(f"[scan_gate_suite] suite_summary: {suite_path}")

    rc = 1 if failures > 0 else 0
    return suite_summary, suite_path, int(rc)


def main() -> None:
    args = parse_args()
    _, _, rc = run_suite(args)
    raise SystemExit(int(rc))


if __name__ == "__main__":
    main()
