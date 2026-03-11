from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from scripts import ardupilot_scan_gate_suite as gate_suite


PRIMARY_IMPROVE_THRESHOLD = 1e-3
EPS = 1e-9


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep ArduPilot bridge (step_len, accept_radius, vxy_cap) around pinned defaults.")
    parser.add_argument("--scale-bucket", type=str, required=True, choices=("scale1", "scale2"))
    parser.add_argument("--conn", type=str, default="udp:127.0.0.1:14550")
    parser.add_argument("--model", type=str, default="auto")
    parser.add_argument("--task", type=str, default="scan")
    parser.add_argument("--preset", type=str, default="A2")
    parser.add_argument("--scan-path-len-scale", type=float, default=None)
    parser.add_argument("--bounds-m", type=float, nargs=2, default=(40.0, 40.0), metavar=("W", "H"))
    parser.add_argument("--margin-m", type=float, default=2.0)
    parser.add_argument("--alt-m", type=float, default=10.0)
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--grid",
        type=str,
        default="step=3,4,5;accept=0.75,1.0,1.25;vxy=1.0,1.2,1.5",
        help="Grid spec, e.g. step=3,4,5;accept=0.75,1.0;vxy=1.0,1.2",
    )
    parser.add_argument("--dry-run", type=int, default=1, choices=(0, 1))
    parser.add_argument("--two-stage", type=int, default=0, choices=(0, 1))
    parser.add_argument("--auto-bless", type=int, default=0, choices=(0, 1))
    parser.add_argument("--out-dir", type=str, default="")
    return parser.parse_args(argv)


def _scale_for_bucket(bucket: str) -> float:
    return 2.0 if str(bucket) == "scale2" else 1.0


def _selected_profile_for_scale(scale: float) -> str:
    return "patch7_scale2" if float(scale) >= 2.0 else "patch5_default"


def _resolve_out_dir(args: argparse.Namespace) -> Path:
    if (args.out_dir or "").strip():
        out = Path(args.out_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        return out
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path("runs") / "ardupilot_param_sweep" / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def _parse_float_list(csv_text: str) -> list[float]:
    vals: list[float] = []
    for tok in str(csv_text).split(","):
        t = tok.strip()
        if not t:
            continue
        vals.append(float(t))
    return vals


def _parse_grid_spec(text: str) -> dict[str, list[float]]:
    spec: dict[str, list[float]] = {}
    alias = {
        "step": "step",
        "step_len": "step",
        "step_len_m": "step",
        "accept": "accept",
        "accept_radius": "accept",
        "accept_radius_m": "accept",
        "vxy": "vxy",
        "vxy_cap": "vxy",
    }
    for part in str(text).split(";"):
        p = part.strip()
        if not p:
            continue
        if "=" not in p:
            raise ValueError(f"Invalid grid token '{p}', expected key=v1,v2")
        k_raw, v_raw = p.split("=", 1)
        key = alias.get(k_raw.strip().lower(), "")
        if not key:
            raise ValueError(f"Unknown grid key '{k_raw}'")
        spec[key] = _parse_float_list(v_raw)
    for req in ("step", "accept", "vxy"):
        if req not in spec or len(spec[req]) == 0:
            raise ValueError(f"Missing grid values for '{req}'")
    return spec


def _expand_grid(spec: dict[str, list[float]]) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for step in spec["step"]:
        for accept in spec["accept"]:
            for vxy in spec["vxy"]:
                out.append(
                    {
                        "step_len_m": float(step),
                        "accept_radius_m": float(accept),
                        "vxy_cap": float(vxy),
                    }
                )
    return out


def _mid_list_value(vals: list[float]) -> float:
    if len(vals) <= 0:
        raise ValueError("Cannot select middle from empty list.")
    return float(vals[len(vals) // 2])


def _load_pinned_summary(scale_bucket: str) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    path = Path("runs") / "production_ardupilot_defaults" / f"{scale_bucket}_opt_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Pinned defaults not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    ranked = data.get("ranked", [])
    if not isinstance(ranked, list) or len(ranked) <= 0 or not isinstance(ranked[0], dict):
        raise ValueError(f"Pinned opt summary has no ranked winner: {path}")
    top = dict(ranked[0])
    top_cfg = dict(top.get("config", {})) if isinstance(top.get("config", {}), dict) else {}
    top_metrics = dict(top.get("metrics", {})) if isinstance(top.get("metrics", {}), dict) else {}
    return path, data, top_cfg, top_metrics


def _num_or_nan(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def _suite_single_arm_stats(suite_summary: dict[str, Any]) -> dict[str, Any]:
    stats_by_arm = suite_summary.get("stats_by_arm", {})
    if isinstance(stats_by_arm, dict) and "S" in stats_by_arm and isinstance(stats_by_arm["S"], dict):
        return dict(stats_by_arm["S"])
    if isinstance(stats_by_arm, dict) and stats_by_arm:
        first = next(iter(stats_by_arm.values()))
        if isinstance(first, dict):
            return dict(first)
    return {}


def _rank_tuple(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    cov95 = _num_or_nan(metrics.get("cov95_hit_rate"))
    cov = _num_or_nan(metrics.get("coverage_mean"))
    primary = cov95 if not math.isnan(cov95) else cov
    p95d = _num_or_nan(metrics.get("p95_dist_to_target"))
    clamp_rate = _num_or_nan(metrics.get("clamp_rate_per_min"))
    oob = _num_or_nan(metrics.get("oob_events"))
    if math.isnan(primary):
        primary = -1e9
    if math.isnan(p95d):
        p95d = 1e9
    if math.isnan(clamp_rate):
        clamp_rate = 1e9
    if math.isnan(oob):
        oob = 1e9
    return (primary, -p95d, -clamp_rate, -oob)


def _build_suite_argv(
    args: argparse.Namespace,
    out_dir: Path,
    scale: float,
    toggles: dict[str, Any],
    combo: dict[str, float],
) -> list[str]:
    argv = [
        "--sitl-recommended",
        "0",
        "--conn",
        str(args.conn),
        "--model",
        str(args.model),
        "--task",
        str(args.task),
        "--preset",
        str(args.preset),
        "--scan-path-len-scale",
        str(float(scale)),
        "--bounds-m",
        str(float(args.bounds_m[0])),
        str(float(args.bounds_m[1])),
        "--margin-m",
        str(float(args.margin_m)),
        "--alt-m",
        str(float(args.alt_m)),
        "--duration-s",
        str(float(args.duration_s)),
        "--runs",
        str(int(args.runs)),
        "--ab",
        "0",
        "--dry-run",
        str(int(args.dry_run)),
        "--out-dir",
        str(out_dir),
        "--step-len-m",
        str(float(combo["step_len_m"])),
        "--accept-radius-m",
        str(float(combo["accept_radius_m"])),
        "--vxy-cap",
        str(float(combo["vxy_cap"])),
        "--lookahead-enable",
        str(int(toggles.get("lookahead_enable", 1))),
        "--step-sched-enable",
        str(int(toggles.get("step_sched_enable", 1))),
        "--lane-keep-enable",
        str(int(toggles.get("lane_keep_enable", 0))),
        "--adaptive-tracking",
        str(int(toggles.get("adaptive_tracking", 0))),
    ]
    if "lane_kp" in toggles:
        argv.extend(["--lane-kp", str(float(toggles["lane_kp"]))])
    if "corner_vxy_cap" in toggles:
        argv.extend(["--corner-vxy-cap", str(float(toggles["corner_vxy_cap"]))])
    return argv


def _run_candidates(
    args: argparse.Namespace,
    out_dir: Path,
    scale: float,
    toggles: dict[str, Any],
    combos: list[dict[str, float]],
    candidate_start: int = 1,
    stage_label: str = "",
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    failures = 0
    for offset, combo in enumerate(combos):
        candidate_id = int(candidate_start + offset)
        cand_dir = out_dir / f"cand_{candidate_id:03d}"
        cand_dir.mkdir(parents=True, exist_ok=True)
        suite_args = gate_suite.parse_args(_build_suite_argv(args, cand_dir, scale, toggles, combo))
        suite_summary, suite_path, rc = gate_suite.run_suite(suite_args)
        stats = _suite_single_arm_stats(suite_summary)
        metrics = {
            "coverage_mean": _num_or_nan(stats.get("coverage_mean")),
            "cov95_hit_rate": _num_or_nan(stats.get("cov95_hit_rate")),
            "p95_dist_to_target": _num_or_nan(stats.get("p95_dist_to_target")),
            "clamp_rate_per_min": _num_or_nan(stats.get("clamp_rate_per_min")),
            "oob_events": _num_or_nan(stats.get("oob_events")),
            "mean_abs_cross_track": _num_or_nan(stats.get("mean_abs_cross_track")),
        }
        rows.append(
            {
                "stage": str(stage_label),
                "candidate_id": int(candidate_id),
                "params": combo,
                "returncode": int(rc),
                "suite_summary_path": str(suite_path),
                "metrics": metrics,
            }
        )
        if int(rc) != 0:
            failures += 1
    return rows, int(failures)


def _beats_baseline(best_metrics: dict[str, Any], base_metrics: dict[str, Any]) -> bool:
    best_cov95 = _num_or_nan(best_metrics.get("cov95_hit_rate"))
    base_cov95 = _num_or_nan(base_metrics.get("cov95_hit_rate"))
    best_cov = _num_or_nan(best_metrics.get("coverage_mean"))
    base_cov = _num_or_nan(base_metrics.get("coverage_mean"))
    best_primary = best_cov95 if not math.isnan(best_cov95) else best_cov
    base_primary = base_cov95 if not math.isnan(base_cov95) else base_cov
    if math.isnan(best_primary) or math.isnan(base_primary):
        return False
    if best_primary > (base_primary + PRIMARY_IMPROVE_THRESHOLD):
        return True
    if abs(best_primary - base_primary) <= PRIMARY_IMPROVE_THRESHOLD:
        best_sec = _rank_tuple(best_metrics)[1:]
        base_sec = _rank_tuple(base_metrics)[1:]
        return best_sec > base_sec
    return False


def _make_blessable_opt_summary(
    out_path: Path,
    scale: float,
    bounds_m: tuple[float, float],
    duration_s: float,
    runs: int,
    winner_cfg: dict[str, Any],
    winner_metrics: dict[str, Any],
) -> Path:
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metadata": {
            "scan_path_len_scale": float(scale),
            "selected_profile": _selected_profile_for_scale(float(scale)),
            "bounds_m": [float(bounds_m[0]), float(bounds_m[1])],
            "bounds-m": [float(bounds_m[0]), float(bounds_m[1])],
            "duration_s": float(duration_s),
            "duration-s": float(duration_s),
            "runs": int(runs),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "ranked": [
            {
                "arm": "A_baseline",
                "config": winner_cfg,
                "metrics": winner_metrics,
                "rank": 1,
            }
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _fmt(x: Any) -> str:
    v = _num_or_nan(x)
    if math.isnan(v):
        return "nan"
    return f"{v:.3f}"


def main() -> None:
    args = parse_args()
    out_dir = _resolve_out_dir(args)
    scale = float(args.scan_path_len_scale) if args.scan_path_len_scale is not None else _scale_for_bucket(args.scale_bucket)
    grid_spec = _parse_grid_spec(args.grid)
    combos = _expand_grid(grid_spec)

    pinned_path, _pinned_data, pinned_cfg, pinned_metrics = _load_pinned_summary(args.scale_bucket)
    toggles = {
        "lookahead_enable": int(pinned_cfg.get("lookahead_enable", 1)),
        "step_sched_enable": int(pinned_cfg.get("step_sched_enable", 1)),
        "lane_keep_enable": int(pinned_cfg.get("lane_keep_enable", 0)),
        "adaptive_tracking": int(pinned_cfg.get("adaptive_tracking", 0)),
    }
    if "lane_kp" in pinned_cfg:
        toggles["lane_kp"] = float(pinned_cfg["lane_kp"])
    if "corner_vxy_cap" in pinned_cfg:
        toggles["corner_vxy_cap"] = float(pinned_cfg["corner_vxy_cap"])

    rows: list[dict[str, Any]] = []
    failures = 0
    stage_summaries: dict[str, Any] = {}

    if int(args.two_stage) == 1:
        stage_a_accept = _mid_list_value(grid_spec["accept"])
        stage_a_combos = [
            {
                "step_len_m": float(step),
                "accept_radius_m": float(stage_a_accept),
                "vxy_cap": float(vxy),
            }
            for step in grid_spec["step"]
            for vxy in grid_spec["vxy"]
        ]
        rows_a, fail_a = _run_candidates(
            args=args,
            out_dir=out_dir,
            scale=scale,
            toggles=toggles,
            combos=stage_a_combos,
            candidate_start=1,
            stage_label="A",
        )
        ranked_a = sorted(rows_a, key=lambda r: _rank_tuple(r["metrics"]), reverse=True)
        for i, r in enumerate(ranked_a, start=1):
            r["rank"] = int(i)
        best_a = ranked_a[0] if ranked_a else None
        rows.extend(rows_a)
        failures += int(fail_a)
        stage_summaries["A"] = {
            "mode": "coarse_step_vxy",
            "accept_fixed": float(stage_a_accept),
            "candidate_count": int(len(rows_a)),
            "ranked": ranked_a,
            "best_candidate": best_a,
        }

        if best_a is not None:
            stage_b_combos = [
                {
                    "step_len_m": float(best_a["params"]["step_len_m"]),
                    "accept_radius_m": float(accept),
                    "vxy_cap": float(best_a["params"]["vxy_cap"]),
                }
                for accept in grid_spec["accept"]
            ]
        else:
            stage_b_combos = []
        rows_b, fail_b = _run_candidates(
            args=args,
            out_dir=out_dir,
            scale=scale,
            toggles=toggles,
            combos=stage_b_combos,
            candidate_start=(len(rows) + 1),
            stage_label="B",
        )
        ranked_b = sorted(rows_b, key=lambda r: _rank_tuple(r["metrics"]), reverse=True)
        for i, r in enumerate(ranked_b, start=1):
            r["rank"] = int(i)
        best_b = ranked_b[0] if ranked_b else None
        rows.extend(rows_b)
        failures += int(fail_b)
        stage_summaries["B"] = {
            "mode": "refine_accept",
            "step_fixed": float(best_a["params"]["step_len_m"]) if best_a is not None else float("nan"),
            "vxy_fixed": float(best_a["params"]["vxy_cap"]) if best_a is not None else float("nan"),
            "candidate_count": int(len(rows_b)),
            "ranked": ranked_b,
            "best_candidate": best_b,
        }

        ranked = ranked_b if ranked_b else ranked_a
        best = ranked[0] if ranked else None
    else:
        rows, failures = _run_candidates(
            args=args,
            out_dir=out_dir,
            scale=scale,
            toggles=toggles,
            combos=combos,
            candidate_start=1,
            stage_label="S",
        )
        ranked = sorted(rows, key=lambda r: _rank_tuple(r["metrics"]), reverse=True)
        for i, r in enumerate(ranked, start=1):
            r["rank"] = int(i)
        best = ranked[0] if ranked else None

    baseline_metrics = {
        "coverage_mean": _num_or_nan(pinned_metrics.get("coverage_mean")),
        "cov95_hit_rate": _num_or_nan(pinned_metrics.get("cov95_hit_rate")),
        "p95_dist_to_target": _num_or_nan(pinned_metrics.get("p95_dist_to_target")),
        "clamp_rate_per_min": _num_or_nan(pinned_metrics.get("clamp_rate_per_min")),
        "oob_events": _num_or_nan(pinned_metrics.get("oob_events")),
        "mean_abs_cross_track": _num_or_nan(pinned_metrics.get("mean_abs_cross_track")),
    }

    improved_vs_baseline = bool(best is not None and _beats_baseline(best["metrics"], baseline_metrics))
    blessed = False
    bless_info: dict[str, Any] = {}
    if int(args.auto_bless) == 1 and bool(improved_vs_baseline) and best is not None:
        pinned_prod = Path("runs") / "production_ardupilot_defaults" / f"{args.scale_bucket}_opt_summary.json"
        prev_prod = Path("runs") / "production_ardupilot_defaults" / f"{args.scale_bucket}_opt_summary_prev.json"
        if pinned_prod.exists():
            shutil.copy2(pinned_prod, prev_prod)
        best_cfg_for_pin = dict(toggles)
        best_cfg_for_pin.update(
            {
                "step_len_m": float(best["params"]["step_len_m"]),
                "accept_radius_m": float(best["params"]["accept_radius_m"]),
                "vxy_cap": float(best["params"]["vxy_cap"]),
            }
        )
        best_opt_path = _make_blessable_opt_summary(
            out_dir / "best_opt_summary.json",
            scale=scale,
            bounds_m=(float(args.bounds_m[0]), float(args.bounds_m[1])),
            duration_s=float(args.duration_s),
            runs=int(args.runs),
            winner_cfg=best_cfg_for_pin,
            winner_metrics=best["metrics"],
        )
        bless_cmd = [
            sys.executable,
            "-m",
            "scripts.bless_ardupilot_defaults",
            "--scale-bucket",
            str(args.scale_bucket),
            "--source",
            f"path:{best_opt_path}",
        ]
        proc = subprocess.run(bless_cmd, capture_output=True, text=True, check=False)
        blessed = int(proc.returncode) == 0
        bless_info = {
            "command": bless_cmd,
            "returncode": int(proc.returncode),
            "stdout_tail": proc.stdout.splitlines()[-20:],
            "stderr_tail": proc.stderr.splitlines()[-20:],
            "previous_backup": str(prev_prod),
            "best_opt_summary": str(best_opt_path),
        }

    sweep_summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "out_dir": str(out_dir),
        "config": {
            "scale_bucket": str(args.scale_bucket),
            "scan_path_len_scale": float(scale),
            "conn": str(args.conn),
            "model": str(args.model),
            "preset": str(args.preset),
            "bounds_m": [float(args.bounds_m[0]), float(args.bounds_m[1])],
            "duration_s": float(args.duration_s),
            "runs": int(args.runs),
            "grid": str(args.grid),
            "dry_run": int(args.dry_run),
            "two_stage": int(args.two_stage),
            "auto_bless": int(args.auto_bless),
        },
        "pinned_baseline_path": str(pinned_path),
        "pinned_toggles": toggles,
        "baseline_metrics": baseline_metrics,
        "stage_summaries": stage_summaries,
        "bless_threshold_primary_delta": float(PRIMARY_IMPROVE_THRESHOLD),
        "candidates": rows,
        "ranked": ranked,
        "best_candidate": best,
        "improved_vs_baseline": bool(improved_vs_baseline),
        "auto_bless_performed": int(args.auto_bless),
        "auto_bless_succeeded": int(blessed),
        "auto_bless_info": bless_info,
        "failures": int(failures),
    }
    out_path = out_dir / "sweep_summary.json"
    out_path.write_text(json.dumps(sweep_summary, indent=2), encoding="utf-8")

    print("rank,id,step,accept,vxy,cov95,coverage,p95_dist,clamp_rate,oob")
    for row in ranked:
        p = row["params"]
        m = row["metrics"]
        print(
            f"{row['rank']},{row['candidate_id']},{p['step_len_m']:.3f},{p['accept_radius_m']:.3f},{p['vxy_cap']:.3f},"
            f"{_fmt(m.get('cov95_hit_rate'))},{_fmt(m.get('coverage_mean'))},{_fmt(m.get('p95_dist_to_target'))},"
            f"{_fmt(m.get('clamp_rate_per_min'))},{_fmt(m.get('oob_events'))}"
        )
    print(f"[param_sweep] improved_vs_baseline={int(bool(improved_vs_baseline))}")
    if int(args.auto_bless) == 1:
        print(f"[param_sweep] auto_bless_succeeded={int(bool(blessed))}")
    if best is not None:
        bp = best["params"]
        print(
            f"[param_sweep] best_final_params: "
            f"step_len_m={float(bp['step_len_m']):.3f} "
            f"accept_radius_m={float(bp['accept_radius_m']):.3f} "
            f"vxy_cap={float(bp['vxy_cap']):.3f}"
        )
    print(f"[param_sweep] sweep_summary: {out_path}")

    raise SystemExit(1 if failures > 0 else 0)


if __name__ == "__main__":
    main()
