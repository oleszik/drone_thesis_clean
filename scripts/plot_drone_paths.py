from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SUCCESS_EXIT_REASONS = {
    "stop_cov",
    "strict_lawnmower_complete",
}


@dataclass
class TraceSeries:
    label: str
    flown_xy: np.ndarray
    planned_xy: np.ndarray
    summary: dict[str, Any]
    source_summary: Path | None
    source_telemetry: Path | None
    source_run_export: Path | None = None
    track_t_s: np.ndarray | None = None
    speed_m_s: np.ndarray | None = None
    rel_alt_m: np.ndarray | None = None
    coverage_xy_count: np.ndarray | None = None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                rows.append(obj)
        except Exception as exc:
            raise ValueError(f"failed to parse JSON at {path}:{idx}: {exc}") from exc
    if not rows:
        raise ValueError(f"telemetry file has no rows: {path}")
    return rows


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if math.isfinite(out):
        return out
    return None


def _ll_to_xy(origin_lng: float, origin_lat: float, lng: float, lat: float) -> tuple[float, float]:
    d_lat = float(lat) - float(origin_lat)
    d_lng = float(lng) - float(origin_lng)
    y = d_lat * 111320.0
    x = d_lng * 111320.0 * max(0.1, abs(math.cos(math.radians(float(origin_lat)))))
    return x, y


def _extract_flown_xy(row: dict[str, Any]) -> tuple[float, float] | None:
    x = _as_float(row.get("pos_x_abs"))
    y = _as_float(row.get("pos_y_abs"))
    if x is not None and y is not None:
        return x, y
    pos = row.get("pos")
    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        x = _as_float(pos[0])
        y = _as_float(pos[1])
        if x is not None and y is not None:
            return x, y
    return None


def _extract_target_xy(row: dict[str, Any]) -> tuple[float, float] | None:
    tgt = row.get("target")
    if isinstance(tgt, (list, tuple)) and len(tgt) >= 2:
        x = _as_float(tgt[0])
        y = _as_float(tgt[1])
        if x is not None and y is not None:
            return x, y
    x = _as_float(row.get("tgt_x_abs"))
    y = _as_float(row.get("tgt_y_abs"))
    if x is not None and y is not None:
        return x, y
    return None


def _resolve_telemetry_path(
    run_dir: Path | None,
    summary: dict[str, Any] | None,
    summary_path: Path | None,
    explicit_telemetry: Path | None,
    repo_root: Path,
) -> Path:
    if explicit_telemetry is not None:
        p = explicit_telemetry
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        return p

    if run_dir is not None:
        p = run_dir / "telemetry.jsonl"
        if p.exists():
            return p

    if summary:
        raw = summary.get("telemetry_jsonl") or summary.get("telemetry_path")
        if isinstance(raw, str) and raw.strip():
            s = raw.strip()
            p = Path(s)
            if not p.is_absolute():
                p = (repo_root / Path(s.replace("\\", "/"))).resolve()
            return p

    if summary_path is not None:
        p = summary_path.parent / "telemetry.jsonl"
        if p.exists():
            return p
    raise FileNotFoundError("could not resolve telemetry.jsonl path; pass --telemetry explicitly")


def _load_trace_from_run_export(*, label: str, run_export_json: Path) -> TraceSeries:
    export_path = run_export_json.resolve()
    data = _read_json(export_path)
    snapshot = data.get("snapshot") or {}
    summary_block = data.get("summary") or {}

    mission = snapshot.get("mission_path") or {}
    planned_lng_lat = []
    for wp in mission.get("waypoints_lng_lat") or []:
        if not isinstance(wp, (list, tuple)) or len(wp) < 2:
            continue
        lng = _as_float(wp[0])
        lat = _as_float(wp[1])
        if lng is None or lat is None:
            continue
        planned_lng_lat.append((lng, lat))

    track_items = list((snapshot.get("track") or {}).get("items") or [])
    flown_lng_lat: list[tuple[float, float]] = []
    track_t_s: list[float] = []
    speed_vals: list[float] = []
    alt_vals: list[float] = []
    t0: float | None = None
    for row in track_items:
        lat = _as_float(row.get("lat"))
        lng = _as_float(row.get("lon"))
        if lat is None or lng is None:
            continue
        flown_lng_lat.append((lng, lat))
        t_unix = _as_float(row.get("t_unix"))
        if t_unix is not None:
            if t0 is None:
                t0 = t_unix
            track_t_s.append(max(0.0, t_unix - float(t0)))
        speed_vals.append(float(_as_float(row.get("speed_m_s")) or float("nan")))
        alt_vals.append(float(_as_float(row.get("rel_alt_m")) or float("nan")))

    if not flown_lng_lat and not planned_lng_lat:
        raise ValueError(f"run export has no path points: {export_path}")

    if planned_lng_lat:
        origin_lng, origin_lat = planned_lng_lat[0]
    elif flown_lng_lat:
        origin_lng, origin_lat = flown_lng_lat[0]
    else:
        origin = (snapshot.get("coverage") or {}).get("origin") or {}
        origin_lng = _as_float(origin.get("lng"))
        origin_lat = _as_float(origin.get("lat"))
        if origin_lng is None or origin_lat is None:
            origin_lng, origin_lat = 0.0, 0.0

    planned_xy = np.asarray(
        [_ll_to_xy(origin_lng, origin_lat, lng, lat) for lng, lat in planned_lng_lat],
        dtype=np.float64,
    ) if planned_lng_lat else np.zeros((0, 2), dtype=np.float64)
    flown_xy = np.asarray(
        [_ll_to_xy(origin_lng, origin_lat, lng, lat) for lng, lat in flown_lng_lat],
        dtype=np.float64,
    ) if flown_lng_lat else np.zeros((0, 2), dtype=np.float64)

    coverage_cells = (snapshot.get("coverage") or {}).get("covered_cells") or []
    coverage_xy_count: list[tuple[float, float, float]] = []
    for c in coverage_cells:
        lat_min = _as_float(c.get("lat_min"))
        lat_max = _as_float(c.get("lat_max"))
        lng_min = _as_float(c.get("lng_min"))
        lng_max = _as_float(c.get("lng_max"))
        cnt = _as_float(c.get("count"))
        if None in (lat_min, lat_max, lng_min, lng_max, cnt):
            continue
        cx_lng = 0.5 * (lng_min + lng_max)
        cx_lat = 0.5 * (lat_min + lat_max)
        x, y = _ll_to_xy(origin_lng, origin_lat, cx_lng, cx_lat)
        coverage_xy_count.append((x, y, float(cnt)))

    # Normalize summary keys so status/annotation logic can work across sources.
    merged_summary = dict(summary_block)
    cov_pct = _as_float(summary_block.get("coverage_pct"))
    if cov_pct is not None and "final_coverage" not in merged_summary:
        merged_summary["final_coverage"] = float(cov_pct) / 100.0
    if "exit_reason" not in merged_summary:
        sitl_state = snapshot.get("sitl_state") or {}
        mission_sim = snapshot.get("mission_sim") or {}
        merged_summary["exit_reason"] = str(sitl_state.get("state") or mission_sim.get("sim_status") or "website_run")

    return TraceSeries(
        label=label,
        flown_xy=flown_xy,
        planned_xy=planned_xy,
        summary=merged_summary,
        source_summary=None,
        source_telemetry=None,
        source_run_export=export_path,
        track_t_s=np.asarray(track_t_s, dtype=np.float64) if track_t_s else None,
        speed_m_s=np.asarray(speed_vals, dtype=np.float64) if speed_vals else None,
        rel_alt_m=np.asarray(alt_vals, dtype=np.float64) if alt_vals else None,
        coverage_xy_count=np.asarray(coverage_xy_count, dtype=np.float64) if coverage_xy_count else None,
    )


def _load_trace(
    *,
    label: str,
    run_dir: Path | None,
    telemetry: Path | None,
    summary_path: Path | None,
    run_export_json: Path | None,
    repo_root: Path,
) -> TraceSeries:
    if run_export_json is not None:
        return _load_trace_from_run_export(label=label, run_export_json=run_export_json)

    rd = None if run_dir is None else run_dir.resolve()
    sp = None if summary_path is None else summary_path.resolve()
    summary: dict[str, Any] = {}
    if sp is None and rd is not None and (rd / "summary.json").exists():
        sp = (rd / "summary.json").resolve()
    if sp is not None and sp.exists():
        summary = _read_json(sp)

    tele_path = _resolve_telemetry_path(rd, summary, sp, telemetry, repo_root)
    if not tele_path.exists():
        raise FileNotFoundError(f"telemetry file not found: {tele_path}")
    rows = _parse_jsonl(tele_path)

    flown: list[tuple[float, float]] = []
    planned: list[tuple[float, float]] = []
    for row in rows:
        p = _extract_flown_xy(row)
        if p is not None:
            flown.append(p)
        t = _extract_target_xy(row)
        if t is not None:
            if not planned:
                planned.append(t)
            if int(row.get("target_update_flag", 0)) == 1:
                planned.append(t)

    if not flown:
        raise ValueError(f"no flown path points found in telemetry: {tele_path}")

    planned_clean: list[tuple[float, float]] = []
    for p in planned:
        if not planned_clean:
            planned_clean.append(p)
            continue
        if math.hypot(p[0] - planned_clean[-1][0], p[1] - planned_clean[-1][1]) > 1e-6:
            planned_clean.append(p)

    return TraceSeries(
        label=label,
        flown_xy=np.asarray(flown, dtype=np.float64),
        planned_xy=np.asarray(planned_clean, dtype=np.float64) if planned_clean else np.zeros((0, 2), dtype=np.float64),
        summary=summary,
        source_summary=sp,
        source_telemetry=tele_path.resolve(),
        source_run_export=None,
    )


def _status(summary: dict[str, Any], cov_success_threshold: float) -> tuple[bool, str]:
    if not summary:
        return False, "status=unknown"
    cov_f = _as_float(summary.get("final_coverage", summary.get("coverage_mean")))
    cov_pct = _as_float(summary.get("coverage_pct"))
    if cov_f is None and cov_pct is not None:
        cov_f = float(cov_pct) / 100.0
    exit_reason = str(summary.get("exit_reason") or "").strip()
    ok = bool((cov_f is not None and cov_f >= cov_success_threshold) or (exit_reason in SUCCESS_EXIT_REASONS))
    parts = [
        f"exit={exit_reason or 'unknown'}",
        f"coverage={cov_f:.3f}" if cov_f is not None else (f"coverage_pct={cov_pct:.1f}" if cov_pct is not None else "coverage=NA"),
    ]
    return ok, ", ".join(parts)


def _set_axes_limits(ax, traces: list[TraceSeries]) -> None:
    xs: list[float] = []
    ys: list[float] = []
    for t in traces:
        if t.flown_xy.size:
            xs.extend(t.flown_xy[:, 0].tolist())
            ys.extend(t.flown_xy[:, 1].tolist())
        if t.planned_xy.size:
            xs.extend(t.planned_xy[:, 0].tolist())
            ys.extend(t.planned_xy[:, 1].tolist())
    if not xs or not ys:
        return
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    span = max(x_max - x_min, y_max - y_min, 1.0)
    pad = 0.08 * span
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    half = 0.5 * span + pad
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)


def _draw_bounds_from_summary(ax, summary: dict[str, Any]) -> None:
    try:
        import matplotlib.patches as mpatches
    except Exception:
        return
    mm = summary.get("bounds_minmax_xy")
    if isinstance(mm, (list, tuple)) and len(mm) >= 4:
        x_min = _as_float(mm[0])
        x_max = _as_float(mm[1])
        y_min = _as_float(mm[2])
        y_max = _as_float(mm[3])
        if None not in (x_min, x_max, y_min, y_max):
            rect = mpatches.Rectangle(
                (float(x_min), float(y_min)),
                float(x_max) - float(x_min),
                float(y_max) - float(y_min),
                fill=False,
                linewidth=1.1,
                linestyle="--",
                edgecolor="#6c757d",
                alpha=0.85,
            )
            ax.add_patch(rect)
            return
    bounds = summary.get("bounds")
    origin = summary.get("scan_origin_xy")
    if not (isinstance(bounds, (list, tuple)) and len(bounds) >= 2):
        return
    if not (isinstance(origin, (list, tuple)) and len(origin) >= 2):
        return
    w = _as_float(bounds[0])
    h = _as_float(bounds[1])
    ox = _as_float(origin[0])
    oy = _as_float(origin[1])
    if None in (w, h, ox, oy):
        return
    x_min = ox - 0.5 * w
    y_min = oy - 0.5 * h
    rect = mpatches.Rectangle(
        (x_min, y_min),
        w,
        h,
        fill=False,
        linewidth=1.1,
        linestyle="--",
        edgecolor="#6c757d",
        alpha=0.85,
    )
    ax.add_patch(rect)


def _style_axis(ax, title: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, linestyle="--", alpha=0.28)
    ax.set_aspect("equal", adjustable="box")


def _plot_trace(ax, trace: TraceSeries, flown_color: str = "#1d4ed8", planned_color: str = "#111827") -> None:
    if trace.planned_xy.size:
        ax.plot(
            trace.planned_xy[:, 0],
            trace.planned_xy[:, 1],
            color=planned_color,
            linestyle="--",
            linewidth=1.9,
            marker="o",
            markersize=2.6,
            alpha=0.92,
            label="Planned target sequence",
        )
    ax.plot(
        trace.flown_xy[:, 0],
        trace.flown_xy[:, 1],
        color=flown_color,
        linewidth=2.4,
        alpha=0.95,
        label="Flown trajectory",
    )
    start = trace.flown_xy[0]
    end = trace.flown_xy[-1]
    ax.scatter([start[0]], [start[1]], c="#16a34a", s=42, marker="o", label="Start", zorder=5)
    ax.scatter([end[0]], [end[1]], c="#dc2626", s=58, marker="X", label="End", zorder=5)
    _draw_bounds_from_summary(ax, trace.summary)


def _save_figure(fig, out: Path, *, use_tight_layout: bool = True) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if use_tight_layout:
        fig.tight_layout()
    fig.savefig(out, dpi=220)
    print(f"[plot] wrote: {out}")


def _cmd_single(args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit("matplotlib is required. Install with: pip install matplotlib") from exc
    repo_root = Path(__file__).resolve().parents[1]
    trace = _load_trace(
        label=str(args.label).strip() or "Run",
        run_dir=args.run_dir,
        telemetry=args.telemetry,
        summary_path=args.summary,
        run_export_json=args.run_export_json,
        repo_root=repo_root,
    )
    ok, status = _status(trace.summary, float(args.coverage_success_threshold))
    fig, ax = plt.subplots(figsize=(8.8, 8.0))
    _plot_trace(ax, trace)
    _set_axes_limits(ax, [trace])
    _style_axis(ax, trace.label)
    status_color = "#166534" if ok else "#991b1b"
    ax.text(
        0.01,
        0.99,
        f"{'SUCCESS' if ok else 'FAIL'} | {status}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color=status_color,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1", "alpha": 0.95},
    )
    ax.legend(loc="best")
    _save_figure(fig, args.out)


def _cmd_compare(args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit("matplotlib is required. Install with: pip install matplotlib") from exc
    repo_root = Path(__file__).resolve().parents[1]
    sim = _load_trace(
        label=str(args.sim_label).strip() or "Simulation",
        run_dir=args.sim_run_dir,
        telemetry=args.sim_telemetry,
        summary_path=args.sim_summary,
        run_export_json=args.sim_run_export_json,
        repo_root=repo_root,
    )
    real = _load_trace(
        label=str(args.real_label).strip() or "Real",
        run_dir=args.real_run_dir,
        telemetry=args.real_telemetry,
        summary_path=args.real_summary,
        run_export_json=args.real_run_export_json,
        repo_root=repo_root,
    )

    fig, ax = plt.subplots(figsize=(9.0, 8.0))
    if sim.planned_xy.size:
        ax.plot(
            sim.planned_xy[:, 0],
            sim.planned_xy[:, 1],
            color="#111827",
            linestyle="--",
            linewidth=1.8,
            alpha=0.9,
            label="Planned target sequence",
        )
    ax.plot(sim.flown_xy[:, 0], sim.flown_xy[:, 1], color="#0f766e", linewidth=2.5, alpha=0.92, label=sim.label)
    ax.plot(real.flown_xy[:, 0], real.flown_xy[:, 1], color="#b91c1c", linewidth=2.3, alpha=0.9, label=real.label)
    ax.scatter([sim.flown_xy[0, 0]], [sim.flown_xy[0, 1]], c="#0f766e", marker="o", s=40, zorder=5)
    ax.scatter([real.flown_xy[0, 0]], [real.flown_xy[0, 1]], c="#b91c1c", marker="o", s=40, zorder=5)
    ax.scatter([sim.flown_xy[-1, 0]], [sim.flown_xy[-1, 1]], c="#0f766e", marker="X", s=54, zorder=5)
    ax.scatter([real.flown_xy[-1, 0]], [real.flown_xy[-1, 1]], c="#b91c1c", marker="X", s=54, zorder=5)
    _draw_bounds_from_summary(ax, sim.summary)
    _draw_bounds_from_summary(ax, real.summary)
    _set_axes_limits(ax, [sim, real])
    _style_axis(ax, str(args.title).strip() or "Sim vs Real Trajectory Comparison")
    ax.legend(loc="best")
    _save_figure(fig, args.out)


def _plot_panel(ax, trace: TraceSeries, cov_success_threshold: float, flown_color: str) -> None:
    ok, status = _status(trace.summary, cov_success_threshold)
    _plot_trace(ax, trace, flown_color=flown_color)
    _style_axis(ax, trace.label)
    status_color = "#166534" if ok else "#991b1b"
    ax.text(
        0.01,
        0.99,
        f"{'SUCCESS' if ok else 'FAIL'} | {status}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color=status_color,
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1", "alpha": 0.95},
    )


def _cmd_success_failure(args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit("matplotlib is required. Install with: pip install matplotlib") from exc
    repo_root = Path(__file__).resolve().parents[1]
    success = _load_trace(
        label=str(args.success_label).strip() or "Successful Trajectory",
        run_dir=args.success_run_dir,
        telemetry=args.success_telemetry,
        summary_path=args.success_summary,
        run_export_json=args.success_run_export_json,
        repo_root=repo_root,
    )
    failure = _load_trace(
        label=str(args.failure_label).strip() or "Failed Trajectory",
        run_dir=args.failure_run_dir,
        telemetry=args.failure_telemetry,
        summary_path=args.failure_summary,
        run_export_json=args.failure_run_export_json,
        repo_root=repo_root,
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.6), sharex=True, sharey=True)
    _plot_panel(ax1, success, float(args.coverage_success_threshold), flown_color="#166534")
    _plot_panel(ax2, failure, float(args.coverage_success_threshold), flown_color="#b91c1c")
    _set_axes_limits(ax1, [success, failure])
    _set_axes_limits(ax2, [success, failure])
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(str(args.title).strip() or "Successful vs Failed Drone Trajectories", fontsize=14, fontweight="bold")
    _save_figure(fig, args.out)


def _cmd_website_report(args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit("matplotlib is required. Install with: pip install matplotlib") from exc

    repo_root = Path(__file__).resolve().parents[1]
    trace = _load_trace(
        label=str(args.label).strip() or "Website Simulation Run",
        run_dir=None,
        telemetry=None,
        summary_path=None,
        run_export_json=args.run_export_json,
        repo_root=repo_root,
    )
    ok, status = _status(trace.summary, float(args.coverage_success_threshold))

    fig = plt.figure(figsize=(14.0, 8.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], hspace=0.28, wspace=0.2)
    ax_path = fig.add_subplot(gs[:, 0])
    ax_ts = fig.add_subplot(gs[0, 1])
    ax_cov = fig.add_subplot(gs[1, 1])

    _plot_trace(ax_path, trace, flown_color="#0f766e")
    _set_axes_limits(ax_path, [trace])
    _style_axis(ax_path, "Planned vs Flown Path")
    status_color = "#166534" if ok else "#991b1b"
    ax_path.text(
        0.01,
        0.99,
        f"{'SUCCESS' if ok else 'CHECK'} | {status}",
        transform=ax_path.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        color=status_color,
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1", "alpha": 0.95},
    )
    ax_path.legend(loc="best")

    n_track = int(trace.flown_xy.shape[0])
    t = trace.track_t_s
    if t is None or len(t) != n_track:
        t = np.arange(n_track, dtype=np.float64)
        x_label = "Sample Index"
    else:
        x_label = "Time (s)"
    speed = trace.speed_m_s
    alt = trace.rel_alt_m
    if speed is not None and len(speed) == n_track:
        ax_ts.plot(t, speed, color="#1d4ed8", linewidth=2.0, label="Speed (m/s)")
    if alt is not None and len(alt) == n_track:
        ax2 = ax_ts.twinx()
        ax2.plot(t, alt, color="#b45309", linewidth=1.9, alpha=0.9, label="Rel Alt (m)")
        ax2.set_ylabel("Rel Altitude (m)")
        h1, l1 = ax_ts.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax_ts.legend(h1 + h2, l1 + l2, loc="best")
    else:
        ax_ts.legend(loc="best")
    ax_ts.set_title("Flight Dynamics")
    ax_ts.set_xlabel(x_label)
    ax_ts.set_ylabel("Speed (m/s)")
    ax_ts.grid(True, linestyle="--", alpha=0.28)

    cov_pts = trace.coverage_xy_count
    if cov_pts is not None and cov_pts.size:
        sc = ax_cov.scatter(
            cov_pts[:, 0],
            cov_pts[:, 1],
            c=cov_pts[:, 2],
            cmap="YlOrRd",
            s=22,
            marker="s",
            alpha=0.9,
            linewidths=0.0,
        )
        _set_axes_limits(ax_cov, [trace])
        ax_cov.set_aspect("equal", adjustable="box")
        ax_cov.set_title("Coverage Hit Intensity")
        ax_cov.set_xlabel("X (m)")
        ax_cov.set_ylabel("Y (m)")
        ax_cov.grid(True, linestyle="--", alpha=0.22)
        cbar = fig.colorbar(sc, ax=ax_cov, fraction=0.046, pad=0.04)
        cbar.set_label("Hit Count")
    else:
        ax_cov.set_title("Coverage Summary")
        ax_cov.set_axis_off()

    cov_pct = _as_float(trace.summary.get("coverage_pct"))
    if cov_pct is None:
        cov_f = _as_float(trace.summary.get("final_coverage"))
        cov_pct = None if cov_f is None else 100.0 * cov_f
    overlap_pct = _as_float(trace.summary.get("overlap_pct"))
    track_len = _as_float(trace.summary.get("track_length_m"))
    stats_lines = [
        f"Coverage: {cov_pct:.1f}%" if cov_pct is not None else "Coverage: NA",
        f"Overlap: {overlap_pct:.1f}%" if overlap_pct is not None else "Overlap: NA",
        f"Track length: {track_len:.1f} m" if track_len is not None else "Track length: NA",
        f"Track points: {n_track}",
    ]
    ax_cov.text(
        0.02,
        0.98,
        "\n".join(stats_lines),
        transform=ax_cov.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1", "alpha": 0.96},
    )

    fig.suptitle(str(args.title).strip() or "Website Simulation Run Report", fontsize=14, fontweight="bold")
    fig.subplots_adjust(top=0.9)
    _save_figure(fig, args.out, use_tight_layout=False)


def _coverage_over_time_from_run_export(
    *,
    run_export_json: Path,
    footprint_radius_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    data = _read_json(run_export_json.resolve())
    snapshot = data.get("snapshot") or {}
    coverage = snapshot.get("coverage") or {}
    stats = (coverage.get("stats") or {}) if isinstance(coverage, dict) else {}
    track_items = list((snapshot.get("track") or {}).get("items") or [])
    cells = list((coverage.get("covered_cells") or []) if isinstance(coverage, dict) else [])
    if not track_items:
        raise ValueError(f"run export has no track items: {run_export_json}")
    if not cells:
        raise ValueError(f"run export has no covered_cells: {run_export_json}")

    t_vals: list[float] = []
    track_lng_lat: list[tuple[float, float]] = []
    t0: float | None = None
    for row in track_items:
        lat = _as_float(row.get("lat"))
        lng = _as_float(row.get("lon"))
        if lat is None or lng is None:
            continue
        t_unix = _as_float(row.get("t_unix"))
        if t_unix is None:
            t_vals.append(float(len(t_vals)))
        else:
            if t0 is None:
                t0 = t_unix
            t_vals.append(max(0.0, float(t_unix) - float(t0)))
        track_lng_lat.append((lng, lat))
    if not track_lng_lat:
        raise ValueError(f"run export track has no valid lat/lon points: {run_export_json}")

    origin = (coverage.get("origin") or {}) if isinstance(coverage, dict) else {}
    origin_lng = _as_float(origin.get("lng"))
    origin_lat = _as_float(origin.get("lat"))
    if origin_lng is None or origin_lat is None:
        origin_lng, origin_lat = track_lng_lat[0]

    cell_centers_xy: list[tuple[float, float]] = []
    for c in cells:
        lat_min = _as_float(c.get("lat_min"))
        lat_max = _as_float(c.get("lat_max"))
        lng_min = _as_float(c.get("lng_min"))
        lng_max = _as_float(c.get("lng_max"))
        if None in (lat_min, lat_max, lng_min, lng_max):
            continue
        cx_lng = 0.5 * (lng_min + lng_max)
        cx_lat = 0.5 * (lat_min + lat_max)
        x, y = _ll_to_xy(origin_lng, origin_lat, cx_lng, cx_lat)
        cell_centers_xy.append((x, y))
    if not cell_centers_xy:
        raise ValueError(f"run export coverage has no valid cell bounds: {run_export_json}")

    track_xy = np.asarray(
        [_ll_to_xy(origin_lng, origin_lat, lng, lat) for lng, lat in track_lng_lat],
        dtype=np.float64,
    )
    cells_xy = np.asarray(cell_centers_xy, dtype=np.float64)

    r = _as_float(footprint_radius_m)
    if r is None:
        r = _as_float(coverage.get("footprint_radius_m"))
    if r is None:
        r = 6.0
    r2 = float(r) * float(r)

    seen = np.zeros((cells_xy.shape[0],), dtype=bool)
    covered_counts = np.zeros((track_xy.shape[0],), dtype=np.float64)
    for i in range(track_xy.shape[0]):
        dx = cells_xy[:, 0] - track_xy[i, 0]
        dy = cells_xy[:, 1] - track_xy[i, 1]
        seen |= (dx * dx + dy * dy) <= r2
        covered_counts[i] = float(np.count_nonzero(seen))

    total_cells = _as_float(stats.get("total_cells"))
    if total_cells is None:
        rows = _as_float(coverage.get("rows"))
        cols = _as_float(coverage.get("cols"))
        if rows is not None and cols is not None:
            total_cells = float(rows * cols)
        else:
            total_cells = float(cells_xy.shape[0])
    total_cells = max(float(total_cells), 1.0)
    coverage_pct = 100.0 * covered_counts / total_cells
    return np.asarray(t_vals, dtype=np.float64), coverage_pct, float(total_cells), float(r)


def _cmd_coverage_over_time(args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit("matplotlib is required. Install with: pip install matplotlib") from exc

    t_s, coverage_pct, total_cells, footprint_radius_m = _coverage_over_time_from_run_export(
        run_export_json=args.run_export_json,
        footprint_radius_m=_as_float(args.footprint_radius_m),
    )
    if t_s.size != coverage_pct.size or t_s.size == 0:
        raise ValueError("coverage over time extraction produced empty/invalid arrays")

    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    ax.plot(t_s, coverage_pct, color="#0f766e", linewidth=2.4, label="Coverage (%)")
    ax.scatter([t_s[-1]], [coverage_pct[-1]], color="#b91c1c", s=30, zorder=5)
    ax.axhline(
        y=100.0 * float(args.coverage_success_threshold),
        color="#6b7280",
        linestyle="--",
        linewidth=1.4,
        alpha=0.85,
        label=f"Success threshold ({100.0 * float(args.coverage_success_threshold):.1f}%)",
    )
    ax.set_title(str(args.title).strip() or "Coverage Over Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Coverage (%)")
    ax.set_ylim(0.0, min(100.0, max(100.0 * float(args.coverage_success_threshold) + 5.0, coverage_pct.max() + 2.0)))
    ax.grid(True, linestyle="--", alpha=0.28)
    ax.legend(loc="lower right")
    ax.text(
        0.015,
        0.985,
        "\n".join(
            [
                f"Final coverage: {coverage_pct[-1]:.2f}%",
                f"Total cells: {int(round(total_cells))}",
                f"Footprint radius: {footprint_radius_m:.2f} m",
            ]
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1", "alpha": 0.95},
    )
    _save_figure(fig, args.out)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate thesis-quality drone path figures from bridge telemetry and summary artifacts."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    single = sub.add_parser("single", help="Single run: planned path vs flown trajectory.")
    single.add_argument("--run-dir", type=Path, default=None, help="Run directory containing summary.json/telemetry.jsonl.")
    single.add_argument("--telemetry", type=Path, default=None, help="Path to telemetry.jsonl.")
    single.add_argument("--summary", type=Path, default=None, help="Path to summary.json.")
    single.add_argument("--run-export-json", type=Path, default=None, help="Path to /api/runs/current/export/json output.")
    single.add_argument("--label", type=str, default="Trajectory")
    single.add_argument("--coverage-success-threshold", type=float, default=0.95)
    single.add_argument("--out", type=Path, required=True, help="Output image path (png/svg/pdf).")
    single.set_defaults(func=_cmd_single)

    compare = sub.add_parser("compare", help="Overlay two trajectories (e.g., simulation vs real).")
    compare.add_argument("--sim-run-dir", type=Path, default=None)
    compare.add_argument("--sim-telemetry", type=Path, default=None)
    compare.add_argument("--sim-summary", type=Path, default=None)
    compare.add_argument("--sim-run-export-json", type=Path, default=None)
    compare.add_argument("--sim-label", type=str, default="Simulation")
    compare.add_argument("--real-run-dir", type=Path, default=None)
    compare.add_argument("--real-telemetry", type=Path, default=None)
    compare.add_argument("--real-summary", type=Path, default=None)
    compare.add_argument("--real-run-export-json", type=Path, default=None)
    compare.add_argument("--real-label", type=str, default="Real")
    compare.add_argument("--title", type=str, default="Sim vs Real Trajectory Comparison")
    compare.add_argument("--out", type=Path, required=True, help="Output image path (png/svg/pdf).")
    compare.set_defaults(func=_cmd_compare)

    sf = sub.add_parser("success-failure", help="Side-by-side success/failure trajectory panel.")
    sf.add_argument("--success-run-dir", type=Path, default=None)
    sf.add_argument("--success-telemetry", type=Path, default=None)
    sf.add_argument("--success-summary", type=Path, default=None)
    sf.add_argument("--success-run-export-json", type=Path, default=None)
    sf.add_argument("--success-label", type=str, default="Successful Trajectory")
    sf.add_argument("--failure-run-dir", type=Path, default=None)
    sf.add_argument("--failure-telemetry", type=Path, default=None)
    sf.add_argument("--failure-summary", type=Path, default=None)
    sf.add_argument("--failure-run-export-json", type=Path, default=None)
    sf.add_argument("--failure-label", type=str, default="Failed Trajectory")
    sf.add_argument("--coverage-success-threshold", type=float, default=0.95)
    sf.add_argument("--title", type=str, default="Successful vs Failed Drone Trajectories")
    sf.add_argument("--out", type=Path, required=True, help="Output image path (png/svg/pdf).")
    sf.set_defaults(func=_cmd_success_failure)

    wr = sub.add_parser("website-report", help="Multi-panel report from website run export JSON.")
    wr.add_argument("--run-export-json", type=Path, required=True, help="Path to /api/runs/current/export/json output.")
    wr.add_argument("--label", type=str, default="Website Simulation Run")
    wr.add_argument("--title", type=str, default="Website Simulation Run Report")
    wr.add_argument("--coverage-success-threshold", type=float, default=0.95)
    wr.add_argument("--out", type=Path, required=True, help="Output image path (png/svg/pdf).")
    wr.set_defaults(func=_cmd_website_report)

    cov = sub.add_parser("coverage-over-time", help="Coverage percentage progression over time (from run export JSON).")
    cov.add_argument("--run-export-json", type=Path, required=True, help="Path to /api/runs/current/export/json output.")
    cov.add_argument("--title", type=str, default="Coverage Over Time")
    cov.add_argument("--footprint-radius-m", type=float, default=None, help="Override coverage footprint radius (meters).")
    cov.add_argument("--coverage-success-threshold", type=float, default=0.95)
    cov.add_argument("--out", type=Path, required=True, help="Output image path (png/svg/pdf).")
    cov.set_defaults(func=_cmd_coverage_over_time)

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
