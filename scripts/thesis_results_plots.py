from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ScanRunMetrics:
    name: str
    run_dir: Path
    source: str
    episodes: int | None
    success_rate: float | None
    coverage_mean: float | None
    steps_mean: float | None
    crash_rate: float | None
    version: int | None
    mtime: float


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if math.isfinite(out):
        return out
    return None


def _as_int(value: Any) -> int | None:
    try:
        out = int(value)
    except Exception:
        return None
    if out >= 0:
        return out
    return None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_scan_version(name: str) -> int | None:
    m = re.search(r"scan_v(\d+)", str(name).lower())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _metrics_from_gate_json(run_dir: Path, gate_path: Path) -> ScanRunMetrics:
    payload = _load_json(gate_path)
    episodes = _as_int(payload.get("episodes"))
    success_rate = _as_float(payload.get("success_rate_mean"))
    if success_rate is None:
        sc = _as_float(payload.get("success_count"))
        if episodes and sc is not None and episodes > 0:
            success_rate = sc / float(episodes)
    coverage_mean = _as_float(payload.get("coverage_mean"))
    steps_mean = _as_float(payload.get("steps_mean"))
    crash_rate = _as_float(payload.get("crash_rate_mean"))
    if crash_rate is None:
        cc = _as_float(payload.get("crash_count"))
        if episodes and cc is not None and episodes > 0:
            crash_rate = cc / float(episodes)

    return ScanRunMetrics(
        name=run_dir.name,
        run_dir=run_dir,
        source=str(gate_path.relative_to(run_dir)),
        episodes=episodes,
        success_rate=success_rate,
        coverage_mean=coverage_mean,
        steps_mean=steps_mean,
        crash_rate=crash_rate,
        version=_parse_scan_version(run_dir.name),
        mtime=gate_path.stat().st_mtime,
    )


def _best_metrics_from_gate_dir(run_dir: Path) -> ScanRunMetrics | None:
    gate_dir = run_dir / "gates"
    if not gate_dir.exists():
        return None
    gate_jsons = sorted(gate_dir.glob("*.json"))
    if not gate_jsons:
        return None
    candidates: list[ScanRunMetrics] = []
    for gp in gate_jsons:
        try:
            candidates.append(_metrics_from_gate_json(run_dir, gp))
        except Exception:
            continue
    if not candidates:
        return None
    # Prefer highest success rate, then highest coverage.
    def _score(m: ScanRunMetrics) -> tuple[float, float]:
        return (
            float(m.success_rate if m.success_rate is not None else -1.0),
            float(m.coverage_mean if m.coverage_mean is not None else -1.0),
        )

    return sorted(candidates, key=_score, reverse=True)[0]


def _metrics_from_best_by_gate_summary(run_dir: Path, summary_path: Path) -> ScanRunMetrics | None:
    payload = _load_json(summary_path)
    episodes = _as_int(payload.get("episodes"))
    success_rate: float | None = None
    coverage_mean: float | None = None
    steps_mean: float | None = None
    crash_rate: float | None = None

    best_row = payload.get("best_row")
    if isinstance(best_row, dict):
        coverage_mean = _as_float(best_row.get("coverage_mean"))
        steps_mean = _as_float(best_row.get("steps_mean"))
        sc = _as_float(best_row.get("success_count"))
        if episodes and sc is not None and episodes > 0:
            success_rate = sc / float(episodes)
        crash_rate = _as_float(best_row.get("crash_rate_mean"))
        if crash_rate is None:
            cc = _as_float(best_row.get("crash_count"))
            if episodes and cc is not None and episodes > 0:
                crash_rate = cc / float(episodes)
    else:
        coverage_mean = _as_float(payload.get("coverage_mean"))
        steps_mean = _as_float(payload.get("steps_mean"))
        sc = _as_float(payload.get("success_count"))
        if episodes and sc is not None and episodes > 0:
            success_rate = sc / float(episodes)
        crash_rate = _as_float(payload.get("crash_rate_mean"))
        if crash_rate is None:
            cc = _as_float(payload.get("crash_count"))
            if episodes and cc is not None and episodes > 0:
                crash_rate = cc / float(episodes)

    if all(v is None for v in (success_rate, coverage_mean, steps_mean, crash_rate)):
        return None

    return ScanRunMetrics(
        name=run_dir.name,
        run_dir=run_dir,
        source=summary_path.name,
        episodes=episodes,
        success_rate=success_rate,
        coverage_mean=coverage_mean,
        steps_mean=steps_mean,
        crash_rate=crash_rate,
        version=_parse_scan_version(run_dir.name),
        mtime=summary_path.stat().st_mtime,
    )


def collect_scan_metrics(runs_root: Path) -> list[ScanRunMetrics]:
    out: list[ScanRunMetrics] = []
    for run_dir in sorted(runs_root.glob("scan_*")):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "best_by_gate_summary.json"
        metrics: ScanRunMetrics | None = None
        if summary_path.exists():
            try:
                metrics = _metrics_from_best_by_gate_summary(run_dir, summary_path)
            except Exception:
                metrics = None
        if metrics is None:
            metrics = _best_metrics_from_gate_dir(run_dir)
        if metrics is not None:
            out.append(metrics)
    return out


def select_scan_runs(metrics: list[ScanRunMetrics], top_n: int) -> list[ScanRunMetrics]:
    if not metrics:
        return []
    # Latest by scan version when available; else file modification time.
    ordered = sorted(
        metrics,
        key=lambda m: (
            -1 if m.version is None else m.version,
            m.mtime,
        ),
        reverse=True,
    )
    chosen = ordered[: max(1, int(top_n))]
    # Plot older->newer left-to-right.
    return sorted(chosen, key=lambda m: (m.version if m.version is not None else -1, m.mtime))


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "NA"
    return f"{100.0 * float(v):.1f}%"


def _fmt_num(v: float | None, digits: int = 2) -> str:
    if v is None:
        return "NA"
    return f"{float(v):.{digits}f}"


def plot_scan_results_bar(metrics: list[ScanRunMetrics], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    if not metrics:
        raise ValueError("no scan metrics available for bar chart")
    labels = [m.name for m in metrics]
    success_pct = [100.0 * float(m.success_rate or 0.0) for m in metrics]
    coverage_vals = [float(m.coverage_mean) if m.coverage_mean is not None else float("nan") for m in metrics]

    fig, ax = plt.subplots(figsize=(max(10.0, 0.72 * len(labels) + 3.2), 5.6))
    cmap = plt.get_cmap("YlGn")
    colors = []
    for c in coverage_vals:
        if math.isfinite(c):
            colors.append(cmap(max(0.0, min(1.0, c))))
        else:
            colors.append((0.6, 0.6, 0.6, 1.0))

    bars = ax.bar(range(len(labels)), success_pct, color=colors, edgecolor="#1f2937", linewidth=0.6)
    ax.set_title("Scan Results Across Runs (Success Rate)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0.0, 100.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)

    for i, b in enumerate(bars):
        txt = _fmt_pct(metrics[i].success_rate)
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height() + 1.2,
            txt,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#111827",
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.015)
    cbar.set_label("Coverage Mean")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    print(f"[plot] wrote: {out_path}")


def write_main_results_table(metrics: list[ScanRunMetrics], out_png: Path, out_csv: Path) -> None:
    import matplotlib.pyplot as plt

    if not metrics:
        raise ValueError("no scan metrics available for summary table")

    rows: list[list[str]] = []
    for m in metrics:
        rows.append(
            [
                m.name,
                _fmt_pct(m.success_rate),
                _fmt_pct(m.coverage_mean),
                _fmt_num(m.steps_mean, 1),
                _fmt_pct(m.crash_rate),
                str(m.episodes) if m.episodes is not None else "NA",
            ]
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "success_rate", "coverage_mean", "steps_mean", "crash_rate", "episodes"])
        for r in rows:
            w.writerow(r)
    print(f"[table] wrote: {out_csv}")

    fig_h = max(3.4, 1.2 + 0.36 * len(rows))
    fig, ax = plt.subplots(figsize=(11.8, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Run", "Success", "Coverage", "Steps", "Crash", "Episodes"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(1.0, 1.15)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", color="#111827")
            cell.set_facecolor("#e5e7eb")
        else:
            cell.set_facecolor("#f9fafb" if r % 2 == 0 else "#ffffff")
    ax.set_title("Main Results Summary Table (Scan Runs)", fontsize=13, fontweight="bold", pad=8.0)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    print(f"[table] wrote: {out_png}")


def _load_ep_lengths(run_dir: Path) -> np.ndarray | None:
    npz_path = run_dir / "eval" / "evaluations.npz"
    if not npz_path.exists():
        return None
    try:
        arr = np.load(npz_path, allow_pickle=True)
    except Exception:
        return None
    if "ep_lengths" not in arr.files:
        return None
    vals = np.asarray(arr["ep_lengths"], dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0.0]
    if vals.size == 0:
        return None
    return vals


def plot_episode_length_distribution(metrics: list[ScanRunMetrics], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    if not metrics:
        raise ValueError("no scan metrics available for episode length distribution")
    labels: list[str] = []
    series: list[np.ndarray] = []
    for m in metrics:
        vals = _load_ep_lengths(m.run_dir)
        if vals is None:
            continue
        labels.append(m.name)
        series.append(vals)
    if not series:
        raise ValueError("no eval/evaluations.npz with ep_lengths found for selected runs")

    all_lengths = np.concatenate(series)
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(max(10.0, 0.72 * len(labels) + 3.2), 8.4),
        gridspec_kw={"height_ratios": [1.1, 1.0]},
    )

    ax1.boxplot(series, tick_labels=labels, showfliers=False)
    ax1.set_title("Episode Length Distribution Across Runs", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Episode Length (steps)")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax1.tick_params(axis="x", rotation=35)

    bins = min(40, max(12, int(round(math.sqrt(all_lengths.size)))))
    ax2.hist(all_lengths, bins=bins, color="#0f766e", alpha=0.9, edgecolor="#111827", linewidth=0.5)
    ax2.set_xlabel("Episode Length (steps)")
    ax2.set_ylabel("Count")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax2.text(
        0.99,
        0.95,
        f"n={int(all_lengths.size)}\nmean={float(np.mean(all_lengths)):.1f}\nmedian={float(np.median(all_lengths)):.1f}",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1", "alpha": 0.95},
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    print(f"[plot] wrote: {out_path}")


def _generate_sequence_waypoints(*, preset: str, seed: int, start_xyz: tuple[float, float, float]) -> tuple[np.ndarray, float]:
    from quad_rl.curriculum.presets import get_preset

    cfg = get_preset(preset)
    rng = np.random.default_rng(int(seed))
    start = np.asarray(start_xyz, dtype=np.float64)
    cursor = start.copy()
    pts: list[np.ndarray] = []

    for _ in range(int(cfg.seq_n_waypoints)):
        hop = float(rng.uniform(float(cfg.seq_hop_min), float(cfg.seq_hop_max)))
        theta = float(rng.uniform(-np.pi, np.pi))
        cursor = cursor + np.asarray([hop * np.cos(theta), hop * np.sin(theta), 0.0], dtype=np.float64)
        if bool(cfg.seq_lock_altitude):
            cursor[2] = float(start[2])
        else:
            z_next = float(cursor[2] + rng.uniform(-0.3, 0.3))
            cursor[2] = float(np.clip(z_next, float(cfg.world_z_min) + 0.4, float(cfg.world_z_max) - 0.4))
        pts.append(cursor.copy())
    return np.asarray(pts, dtype=np.float64), float(cfg.seq_success_radius)


def plot_sequence_waypoint_path(
    *,
    out_path: Path,
    preset: str = "A2",
    seed: int = 456,
    start_xyz: tuple[float, float, float] = (0.0, 0.0, 1.5),
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    waypoints, success_radius = _generate_sequence_waypoints(preset=preset, seed=seed, start_xyz=start_xyz)
    start = np.asarray(start_xyz, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.8, 8.0))
    ax.plot(
        np.r_[start[0], waypoints[:, 0]],
        np.r_[start[1], waypoints[:, 1]],
        color="#1d4ed8",
        linewidth=2.2,
        marker="o",
        markersize=3.5,
        alpha=0.95,
        label="Waypoint sequence path",
    )
    ax.scatter([start[0]], [start[1]], c="#16a34a", s=52, marker="o", zorder=5, label="Start")
    ax.scatter([waypoints[-1, 0]], [waypoints[-1, 1]], c="#dc2626", s=62, marker="X", zorder=5, label="Final waypoint")

    for i, (x, y, _z) in enumerate(waypoints, start=1):
        ax.text(x, y, str(i), fontsize=8, ha="left", va="bottom", color="#111827")
        ax.add_patch(
            mpatches.Circle((x, y), radius=success_radius, fill=False, linewidth=0.8, linestyle="--", edgecolor="#6b7280", alpha=0.45)
        )

    xs = np.r_[start[0], waypoints[:, 0]]
    ys = np.r_[start[1], waypoints[:, 1]]
    span = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 1.0)
    pad = 0.12 * span
    ax.set_xlim(float(xs.min()) - pad, float(xs.max()) + pad)
    ax.set_ylim(float(ys.min()) - pad, float(ys.max()) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.28)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Sequence Task Waypoint Path ({preset}, seed={seed})", fontsize=13, fontweight="bold")
    ax.legend(loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    print(f"[plot] wrote: {out_path}")


def _cmd_all(args: argparse.Namespace) -> None:
    runs_root = args.runs_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_all = collect_scan_metrics(runs_root)
    metrics = select_scan_runs(metrics_all, top_n=int(args.top_n))
    if not metrics:
        raise SystemExit("No scan metrics found under runs_root.")

    bar_path = out_dir / "scan_results_across_runs_bar.png"
    table_png = out_dir / "main_results_summary_table.png"
    table_csv = out_dir / "main_results_summary_table.csv"
    epdist_path = out_dir / "episode_length_distribution_scan_runs.png"
    seq_path = out_dir / f"sequence_task_waypoint_path_{args.sequence_preset}_seed{int(args.sequence_seed)}.png"

    plot_scan_results_bar(metrics, bar_path)
    write_main_results_table(metrics, table_png, table_csv)
    plot_episode_length_distribution(metrics, epdist_path)
    plot_sequence_waypoint_path(
        out_path=seq_path,
        preset=str(args.sequence_preset),
        seed=int(args.sequence_seed),
        start_xyz=(0.0, 0.0, float(args.sequence_start_alt_m)),
    )

    print("\n[done] Generated artifacts:")
    for p in [bar_path, table_png, table_csv, seq_path, epdist_path]:
        print(f"  - {p}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate thesis result visuals from local run artifacts.")
    sub = p.add_subparsers(dest="cmd", required=True)

    all_p = sub.add_parser("all", help="Generate all requested thesis visuals.")
    all_p.add_argument("--runs-root", type=Path, default=Path("runs"))
    all_p.add_argument("--out-dir", type=Path, default=Path("runs/figures"))
    all_p.add_argument("--top-n", type=int, default=12, help="Number of scan runs in cross-run bar/table/distribution.")
    all_p.add_argument("--sequence-preset", type=str, default="A2")
    all_p.add_argument("--sequence-seed", type=int, default=456)
    all_p.add_argument("--sequence-start-alt-m", type=float, default=1.5)
    all_p.set_defaults(func=_cmd_all)
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
