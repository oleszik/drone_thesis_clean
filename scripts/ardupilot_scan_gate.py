from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an ArduPilot SITL scan gate via scripts.ardupilot_bridge.")
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
    parser.add_argument("--oob-recovery", type=int, default=1, choices=(0, 1))
    parser.add_argument("--oob-recovery-seconds", type=float, default=3.0)
    parser.add_argument("--oob-recovery-vxy-cap", type=float, default=0.6)
    parser.add_argument("--oob-recovery-accept-boost", type=float, default=0.5)
    parser.add_argument("--oob-clamp-fast-count", type=int, default=3)
    parser.add_argument("--oob-clamp-fast-window-s", type=float, default=5.0)
    parser.add_argument("--adaptive-tracking", type=int, default=0, choices=(0, 1))
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
    parser.add_argument(
        "--cfg-override",
        action="append",
        default=[],
        help="Override preset fields with key=value (repeatable).",
    )
    parser.add_argument("--dry-run", type=int, default=1, choices=(0, 1))
    parser.add_argument("--out", type=str, default="", help="Optional output path for gate_summary.json")
    return parser.parse_args(argv)


def _extract_bridge_summary_path(stdout_text: str) -> Path | None:
    marker = "[bridge] summary:"
    for line in stdout_text.splitlines():
        if marker not in line:
            continue
        raw = line.split(marker, 1)[1].strip()
        if raw:
            return Path(raw)
    return None


def _latest_bridge_summary(min_mtime_s: float) -> Path | None:
    root = Path("runs") / "ardupilot_scan"
    if not root.exists():
        return None
    cands = sorted(root.glob("*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in cands:
        if path.stat().st_mtime >= (float(min_mtime_s) - 2.0):
            return path
    return cands[0] if cands else None


def _resolve_out_path(out_arg: str) -> Path:
    if (out_arg or "").strip():
        out_path = Path(out_arg).expanduser()
        if out_path.suffix.lower() != ".json":
            out_path = out_path / "gate_summary.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / "ardupilot_scan_gate" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / "gate_summary.json"


def _fmt_time_to_95(x: Any) -> str:
    if x is None:
        return "nan"
    try:
        if math.isnan(float(x)):
            return "nan"
    except Exception:
        return "nan"
    return f"{float(x):.3f}"


def _build_bridge_cmd(args: argparse.Namespace) -> list[str]:
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
        "--connection",
        str(args.conn),
        "--ack-verbosity",
        str(args.ack_verbosity),
        "--steps",
        str(step_budget),
        "--scan-path-len-scale",
        str(float(args.scan_path_len_scale)),
        "--bounds-m",
        str(float(args.bounds_m[0])),
        str(float(args.bounds_m[1])),
        "--margin-m",
        str(float(args.margin_m)),
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
        "--alt-m",
        str(float(args.alt_m)),
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
        "--adaptive-tracking",
        str(int(args.adaptive_tracking)),
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
    ]
    if args.vxy_cap is not None:
        cmd.extend(["--vxy-cap", str(float(args.vxy_cap))])
    if args.step_len_m is not None:
        cmd.extend(["--step-len-m", str(float(args.step_len_m))])
    if args.accept_radius_m is not None:
        cmd.extend(["--accept-radius-m", str(float(args.accept_radius_m))])
    for ov in args.cfg_override:
        cmd.extend(["--cfg-override", str(ov)])
    cmd.extend(["--cfg-override", f"scan_max_steps={step_budget}"])
    return cmd


def run_gate(args: argparse.Namespace) -> tuple[dict[str, Any], Path, int]:
    out_path = _resolve_out_path(args.out)
    launch_wall_s = time.time()
    bridge_cmd = _build_bridge_cmd(args)
    proc = subprocess.run(bridge_cmd, capture_output=True, text=True, check=False)

    summary_path = _extract_bridge_summary_path(proc.stdout)
    if summary_path is None or not summary_path.exists():
        summary_path = _latest_bridge_summary(launch_wall_s)
    if summary_path is None or not summary_path.exists():
        gate_summary = {
            "bridge_cmd": bridge_cmd,
            "bridge_returncode": int(proc.returncode),
            "bridge_stdout_tail": proc.stdout.splitlines()[-40:],
            "bridge_stderr_tail": proc.stderr.splitlines()[-40:],
            "error": "bridge_summary_not_found",
        }
        out_path.write_text(json.dumps(gate_summary, indent=2), encoding="utf-8")
        print(f"[scan_gate] failed to find bridge summary. gate_summary={out_path}")
        return gate_summary, out_path, int(max(1, int(proc.returncode)))

    bridge_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    coverage_mean = float(bridge_summary.get("coverage_mean", bridge_summary.get("final_coverage", 0.0)))
    final_coverage = float(bridge_summary.get("final_coverage", coverage_mean))
    cov95_hit = int(final_coverage >= 0.95)
    time_to_95_s = bridge_summary.get("time_to_95_s", None)
    clamp_count = int(bridge_summary.get("clamp_count", 0))
    p95_dist_to_target = float(bridge_summary.get("p95_dist_to_target", 0.0))

    gate_summary = {
        "bridge_cmd": bridge_cmd,
        "bridge_returncode": int(proc.returncode),
        "bridge_summary_path": str(summary_path),
        "coverage_mean": float(coverage_mean),
        "cov95_hit": int(cov95_hit),
        "time_to_95_s": time_to_95_s,
        "clamp_count": int(clamp_count),
        "p95_dist_to_target": float(p95_dist_to_target),
        "final_coverage": float(final_coverage),
        "exit_reason": str(bridge_summary.get("exit_reason", "")),
        "duration_s": float(bridge_summary.get("duration_s", 0.0)),
        "policy_hz": float(bridge_summary.get("policy_hz", 0.0)),
        "num_policy_ticks": int(bridge_summary.get("num_policy_ticks", 0)),
        "adaptive_tracking": int(bridge_summary.get("adaptive_tracking", 0)),
        "adaptive_event_count": int(bridge_summary.get("adaptive_event_count", 0)),
        "adaptive_bad_count": int(bridge_summary.get("adaptive_bad_count", 0)),
        "adaptive_good_count": int(bridge_summary.get("adaptive_good_count", 0)),
        "mean_progress_per_s": float(bridge_summary.get("mean_progress_per_s", 0.0)),
        "clamp_rate_per_min": float(bridge_summary.get("clamp_rate_per_min", 0.0)),
        "oob_events": int(bridge_summary.get("oob_events", 0)),
        "mean_abs_cross_track": float(bridge_summary.get("mean_abs_cross_track", 0.0)),
        "final_step_len_m": float(bridge_summary.get("final_step_len_m", bridge_summary.get("step_len", 0.0))),
        "final_accept_radius_m": float(
            bridge_summary.get("final_accept_radius_m", bridge_summary.get("accept_radius", 0.0))
        ),
        "final_vxy_cap": float(bridge_summary.get("final_vxy_cap", bridge_summary.get("vxy_cap", 0.0))),
    }
    out_path.write_text(json.dumps(gate_summary, indent=2), encoding="utf-8")

    print(
        f"coverage_mean={coverage_mean:.3f} cov95_hit={cov95_hit} "
        f"time_to_95={_fmt_time_to_95(time_to_95_s)} clamp_count={clamp_count} "
        f"p95_dist_to_target={p95_dist_to_target:.3f}"
    )
    print(f"[scan_gate] gate_summary: {out_path}")

    if proc.returncode != 0:
        print("[scan_gate] bridge returned non-zero. stderr tail:")
        for line in proc.stderr.splitlines()[-20:]:
            print(line)
    return gate_summary, out_path, int(proc.returncode)


def main() -> None:
    args = parse_args()
    _, _, rc = run_gate(args)
    raise SystemExit(int(rc))


if __name__ == "__main__":
    main()
