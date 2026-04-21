from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.patches import Circle, Rectangle

from quad_rl.curriculum.presets import get_preset, list_presets
from quad_rl.utils.config_overrides import apply_overrides, parse_override_pairs

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a metrically-correct scan geometry figure (Figure 4.3.1 style)."
    )
    parser.add_argument("--preset", type=str, default="A2", choices=list_presets())
    parser.add_argument("--seed", type=int, default=456)
    parser.add_argument(
        "--out",
        type=str,
        default="runs/figures/figure_4_3_1_scan_geometry_fixed.png",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--coverage-ds",
        type=float,
        default=0.20,
        help="Arc-length step (m) used to stamp coverage disks along the path.",
    )
    parser.add_argument(
        "--pad-m",
        type=float,
        default=0.9,
        help="Padding around the scan boundary in meters.",
    )
    parser.add_argument(
        "--show-world-bound",
        type=int,
        default=0,
        choices=(0, 1),
        help="Draw dashed +/-world_xy_bound square.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Coverage Scan Mission Geometry and Coverage Accumulation (Metric-Accurate)",
    )
    parser.add_argument(
        "--cfg-override",
        action="append",
        default=[],
        help="Override preset fields with key=value (repeatable).",
    )
    return parser.parse_args()


def _mark_disk_counts(
    counts: np.ndarray,
    x: float,
    y: float,
    radius: float,
    cell_size: float,
    x_min: float,
    y_min: float,
) -> None:
    nx, ny = counts.shape
    r = float(max(radius, 0.0))
    cs = float(max(cell_size, 1e-6))
    r2 = r * r

    ix0 = int(np.floor((x - r - x_min) / cs))
    ix1 = int(np.floor((x + r - x_min) / cs))
    iy0 = int(np.floor((y - r - y_min) / cs))
    iy1 = int(np.floor((y + r - y_min) / cs))
    ix0 = int(np.clip(ix0, 0, nx - 1))
    ix1 = int(np.clip(ix1, 0, nx - 1))
    iy0 = int(np.clip(iy0, 0, ny - 1))
    iy1 = int(np.clip(iy1, 0, ny - 1))

    for ix in range(ix0, ix1 + 1):
        cx = x_min + (float(ix) + 0.5) * cs
        dx = cx - x
        for iy in range(iy0, iy1 + 1):
            cy = y_min + (float(iy) + 0.5) * cs
            dy = cy - y
            if (dx * dx + dy * dy) <= r2:
                counts[ix, iy] += 1


def _sample_range(
    rng: np.random.Generator,
    low: float,
    high: float,
    default: float,
    *,
    min_value: float,
) -> float:
    lo = float(low)
    hi = float(high)
    if hi < lo:
        lo, hi = hi, lo
    if not np.isfinite(lo) or not np.isfinite(hi):
        return max(float(default), min_value)
    if abs(hi - lo) < 1e-9:
        return max(float(lo), min_value)
    return max(float(rng.uniform(lo, hi)), min_value)


def _build_lawnmower(center: np.ndarray, rows: int, cols: int, spacing: float) -> np.ndarray:
    points = []
    x0 = float(center[0]) - 0.5 * spacing * max(0, rows - 1)
    y0 = float(center[1]) - 0.5 * spacing * max(0, cols - 1)
    z = float(center[2])
    for r in range(rows):
        y_indices = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        x = x0 + r * spacing
        for c in y_indices:
            y = y0 + c * spacing
            points.append(np.array([x, y, z], dtype=np.float32))
    return np.asarray(points, dtype=np.float32)


def _compact_path(points: np.ndarray, min_spacing: float) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    compact = [points[0]]
    for p in points[1:]:
        if np.linalg.norm((p - compact[-1])[:2]) >= max(1e-6, float(min_spacing)):
            compact.append(p)
    path = np.asarray(compact, dtype=np.float32)
    if len(path) < 2:
        dx = max(0.5, float(min_spacing))
        p1 = path[0] + np.array([dx, 0.0, 0.0], dtype=np.float32)
        path = np.vstack([path, p1]).astype(np.float32)
    return path


def _clamp_path_to_world(points: np.ndarray, world_xy_bound: float, coverage_radius: float) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    bound = float(world_xy_bound)
    if bound <= 0.0:
        return np.asarray(points, dtype=np.float32).copy()
    margin = max(float(coverage_radius), 0.5, 1e-3)
    limit = max(0.0, bound - margin)
    out = np.asarray(points, dtype=np.float32).copy()
    out[:, 0] = np.clip(out[:, 0], -limit, limit)
    out[:, 1] = np.clip(out[:, 1], -limit, limit)
    return out


def _rebuild_path_cache(path_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    seg_lengths = np.linalg.norm(np.diff(path_points[:, :2], axis=0), axis=1).astype(np.float32)
    cum_lengths = np.concatenate(
        [np.zeros((1,), dtype=np.float32), np.cumsum(seg_lengths, dtype=np.float32)]
    )
    total_length = float(cum_lengths[-1])
    return seg_lengths, cum_lengths, total_length


def _turn_vertices(path_points: np.ndarray) -> np.ndarray:
    turns = np.zeros((len(path_points),), dtype=bool)
    for i in range(1, len(path_points) - 1):
        v0 = path_points[i] - path_points[i - 1]
        v1 = path_points[i + 1] - path_points[i]
        n0 = float(np.linalg.norm(v0[:2]))
        n1 = float(np.linalg.norm(v1[:2]))
        if n0 <= 1e-6 or n1 <= 1e-6:
            continue
        cosang = float(np.clip(np.dot(v0[:2], v1[:2]) / (n0 * n1), -1.0, 1.0))
        angle = float(np.arccos(cosang))
        if angle > 0.2:
            turns[i] = True
    return turns


def _point_and_tangent_at_progress(
    path_points: np.ndarray,
    seg_lengths: np.ndarray,
    cum_lengths: np.ndarray,
    total_length: float,
    progress: float,
) -> tuple[np.ndarray, float]:
    g = float(np.clip(progress, 0.0, total_length))
    last_seg = len(seg_lengths) - 1
    if last_seg < 0:
        return path_points[0].copy(), 0.0
    seg_idx = int(np.searchsorted(cum_lengths, g, side="right") - 1)
    seg_idx = int(np.clip(seg_idx, 0, last_seg))
    seg_len = float(seg_lengths[seg_idx])
    s_local = float(g - float(cum_lengths[seg_idx]))
    s_local = float(np.clip(s_local, 0.0, seg_len))
    a = path_points[seg_idx]
    b = path_points[seg_idx + 1]
    if seg_len <= 1e-6:
        return a.copy(), 0.0
    seg_dir = (b[:2] - a[:2]) / seg_len
    point = a.copy()
    point[:2] = a[:2] + seg_dir * s_local
    point[2] = float(a[2] + (s_local / seg_len) * (b[2] - a[2]))
    yaw = float(np.arctan2(seg_dir[1], seg_dir[0]))
    return point, yaw


def _init_coverage_grid(path_points: np.ndarray, cfg) -> tuple[float, float, float, int, int, float]:
    cell_size = max(float(getattr(cfg, "scan_cov_cell_size", 0.25)), 1e-3)
    spacing = max(float(getattr(cfg, "scan_spacing", 1.0)), cell_size)
    cov_radius = max(float(getattr(cfg, "scan_coverage_radius", 0.5 * spacing)), 0.0)
    path_xy = path_points[:, :2]
    mins = np.min(path_xy, axis=0)
    maxs = np.max(path_xy, axis=0)
    pad = 0.5 * cell_size
    mins = mins - pad
    maxs = maxs + pad
    span = np.maximum(maxs - mins, cell_size)
    nx = max(1, int(np.ceil(float(span[0]) / cell_size)))
    ny = max(1, int(np.ceil(float(span[1]) / cell_size)))
    return float(cell_size), float(mins[0]), float(mins[1]), int(nx), int(ny), float(cov_radius)


def _accumulate_coverage_counts_from_geometry(
    path_points: np.ndarray,
    seg_lengths: np.ndarray,
    cum_lengths: np.ndarray,
    total_length: float,
    ds: float,
    nx: int,
    ny: int,
    x_min: float,
    y_min: float,
    cell_size: float,
    cov_radius: float,
) -> np.ndarray:
    counts = np.zeros((int(nx), int(ny)), dtype=np.int32)
    step = float(max(ds, 1e-6))
    progress = 0.0
    while True:
        p, _ = _point_and_tangent_at_progress(path_points, seg_lengths, cum_lengths, total_length, progress)
        _mark_disk_counts(
            counts=counts,
            x=float(p[0]),
            y=float(p[1]),
            radius=float(cov_radius),
            cell_size=float(cell_size),
            x_min=float(x_min),
            y_min=float(y_min),
        )
        if progress >= total_length:
            break
        progress = min(total_length, progress + step)
    return counts


def _effective_rows_cols(path_scale: float, cfg) -> tuple[int, int]:
    rows = int(cfg.scan_rows)
    cols = int(cfg.scan_cols)
    scale = max(float(path_scale), 0.1)
    if abs(scale - 1.0) > 1e-6:
        rc_scale = float(np.sqrt(scale))
        rows = max(2, int(round(rows * rc_scale)))
        cols = max(2, int(round(cols * rc_scale)))
    return rows, cols


def main() -> None:
    args = parse_args()
    cfg = get_preset(args.preset)
    overrides = parse_override_pairs(args.cfg_override)
    if overrides:
        apply_overrides(cfg, overrides)

    rng = np.random.default_rng(int(args.seed))
    ep_path_len_scale = _sample_range(
        rng,
        getattr(cfg, "scan_path_len_scale_min", getattr(cfg, "scan_path_len_scale", 1.0)),
        getattr(cfg, "scan_path_len_scale_max", getattr(cfg, "scan_path_len_scale", 1.0)),
        getattr(cfg, "scan_path_len_scale", 1.0),
        min_value=0.1,
    )
    rows_eff, cols_eff = _effective_rows_cols(ep_path_len_scale, cfg)
    spacing_base = float(getattr(cfg, "scan_spacing", 0.8))
    spacing_jitter = max(0.0, float(getattr(cfg, "scan_spacing_jitter", 0.0)))
    ep_spacing = spacing_base + float(rng.uniform(-spacing_jitter, spacing_jitter)) if spacing_jitter > 0.0 else spacing_base
    ep_spacing = max(ep_spacing, float(getattr(cfg, "scan_min_wp_spacing", 0.0)), 1e-3)
    z0 = float(getattr(cfg, "scan_z_target", 0.0) or max(float(cfg.world_z_min) + 1.0, 1.0))
    center = np.array([0.0, 0.0, z0], dtype=np.float32)

    raw = _build_lawnmower(center=center, rows=rows_eff, cols=cols_eff, spacing=float(ep_spacing))
    raw = _clamp_path_to_world(
        raw,
        world_xy_bound=float(getattr(cfg, "world_xy_bound", 0.0)),
        coverage_radius=float(getattr(cfg, "scan_coverage_radius", 0.5)),
    )
    path = _compact_path(raw, min_spacing=float(getattr(cfg, "scan_min_wp_spacing", 0.0)))
    path = _clamp_path_to_world(
        path,
        world_xy_bound=float(getattr(cfg, "world_xy_bound", 0.0)),
        coverage_radius=float(getattr(cfg, "scan_coverage_radius", 0.5)),
    )
    path = _compact_path(path, min_spacing=float(getattr(cfg, "scan_min_wp_spacing", 0.0)))
    seg_lengths, cum_lengths, total_length = _rebuild_path_cache(path)
    turns = _turn_vertices(path)
    turn_idx = np.flatnonzero(turns)

    cs, x_min, y_min, nx, ny, cov_radius = _init_coverage_grid(path, cfg)
    counts = _accumulate_coverage_counts_from_geometry(
        path_points=path,
        seg_lengths=seg_lengths,
        cum_lengths=cum_lengths,
        total_length=total_length,
        ds=float(args.coverage_ds),
        nx=int(nx),
        ny=int(ny),
        x_min=float(x_min),
        y_min=float(y_min),
        cell_size=float(cs),
        cov_radius=float(cov_radius),
    )

    path = np.asarray(path, dtype=np.float64)
    x_max = x_min + float(nx) * cs
    y_max = y_min + float(ny) * cs
    width = x_max - x_min
    height = y_max - y_min

    fig, ax = plt.subplots(figsize=(8.4, 8.4), dpi=int(args.dpi))

    # Coverage grid visualization: first-pass vs revisit.
    for ix in range(int(nx)):
        x0 = x_min + float(ix) * cs
        for iy in range(int(ny)):
            y0 = y_min + float(iy) * cs
            n = int(counts[ix, iy])
            if n <= 0:
                continue
            if n == 1:
                face = "#9fdcff"
                alpha = 0.50
            else:
                face = "#f7c98a"
                alpha = 0.55
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    cs,
                    cs,
                    facecolor=face,
                    edgecolor="none",
                    alpha=alpha,
                    zorder=1,
                )
            )

    # Footprint disks at waypoints (concept layer).
    for p in path:
        ax.add_patch(
            Circle(
                (float(p[0]), float(p[1])),
                radius=float(cov_radius),
                facecolor="#9fdcff",
                edgecolor="none",
                alpha=0.12,
                zorder=2,
            )
        )

    # Scan boundary from actual coverage grid extents.
    ax.add_patch(
        Rectangle(
            (x_min, y_min),
            width,
            height,
            fill=False,
            edgecolor="black",
            linewidth=2.2,
            zorder=4,
        )
    )

    # Optional world bound.
    if int(args.show_world_bound) == 1:
        wb = float(cfg.world_xy_bound)
        ax.add_patch(
            Rectangle(
                (-wb, -wb),
                2.0 * wb,
                2.0 * wb,
                fill=False,
                edgecolor="#666666",
                linewidth=1.4,
                linestyle="--",
                zorder=0,
            )
        )

    # Lawnmower path and directional arrows.
    ax.plot(path[:, 0], path[:, 1], color="#1f77b4", linewidth=2.4, zorder=5)
    for i in range(len(path) - 1):
        a = path[i, :2]
        b = path[i + 1, :2]
        ax.annotate(
            "",
            xy=(float(b[0]), float(b[1])),
            xytext=(float(a[0]), float(a[1])),
            arrowprops={
                "arrowstyle": "->",
                "color": "#1f77b4",
                "lw": 1.5,
                "alpha": 0.85,
                "shrinkA": 9,
                "shrinkB": 9,
            },
            zorder=6,
        )

    if len(turn_idx) > 0:
        turns = path[turn_idx, :2]
        ax.scatter(turns[:, 0], turns[:, 1], s=80, c="red", edgecolors="black", linewidths=0.7, zorder=7)

    # Start/end.
    ax.scatter(path[0, 0], path[0, 1], s=120, c="#1ca71c", edgecolors="black", linewidths=1.0, zorder=8)
    ax.scatter(path[-1, 0], path[-1, 1], s=120, c="#a020b0", edgecolors="black", linewidths=1.0, zorder=8)
    ax.text(float(path[0, 0]) - 0.18, float(path[0, 1]) - 0.36, "Start", fontsize=12, weight="bold")
    ax.text(float(path[-1, 0]) - 0.16, float(path[-1, 1]) + 0.18, "End", fontsize=12, weight="bold")

    info = (
        f"Preset {args.preset}\n"
        f"rows={rows_eff}, cols={cols_eff}\n"
        f"spacing={float(ep_spacing):.2f} m\n"
        f"cell={cs:.2f} m, radius={float(cov_radius):.2f} m\n"
        f"boundary={width:.2f} x {height:.2f} m"
    )
    ax.text(
        x_max + 0.35,
        y_max + 0.15,
        info,
        fontsize=11.5,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.95, "edgecolor": "#333333"},
    )

    legend_text = (
        "Black rectangle: scan boundary\n"
        "Blue line/arrows: lawnmower path\n"
        "Red circles: turn vertices\n"
        "Light blue cells: first-pass coverage\n"
        "Orange cells: revisits"
    )
    ax.text(
        x_min - 0.05,
        y_min - 1.12,
        legend_text,
        fontsize=10.8,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.93, "edgecolor": "#333333"},
    )

    # Metric correctness: enforce 1:1 in x/y and square displayed range.
    pad = float(max(args.pad_m, 0.0))
    x0 = min(float(np.min(path[:, 0])), x_min) - pad
    x1 = max(float(np.max(path[:, 0])), x_max) + pad
    y0 = min(float(np.min(path[:, 1])), y_min) - pad
    y1 = max(float(np.max(path[:, 1])), y_max) + pad
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    half = 0.5 * max(x1 - x0, y1 - y0)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("x (m)", fontsize=13)
    ax.set_ylabel("y (m)", fontsize=13)
    ax.set_title(args.title, fontsize=18, pad=12)
    ax.grid(True, linestyle=":", alpha=0.35)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    print(f"[scan_geometry_figure] wrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()
