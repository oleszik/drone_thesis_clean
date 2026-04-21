from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib.patches import Circle, Ellipse, Rectangle
from matplotlib import pyplot as plt

from scripts.bridge.coverage import CoverageGridTracker
from scripts.bridge.pathing import bounds_minmax_xy, build_lawnmower_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot bridge-style scan geometry with FOV-based footprint coverage."
    )
    parser.add_argument("--bounds-m", type=float, nargs=2, default=(100.0, 100.0), metavar=("W", "H"))
    parser.add_argument("--margin-m", type=float, default=2.0)
    parser.add_argument("--step-len-m", type=float, default=12.0, help="Lane spacing in bridge mode.")
    parser.add_argument("--alt-m", type=float, default=10.0)
    parser.add_argument("--turn-style", type=str, default="sharp", choices=("sharp", "arc"))
    parser.add_argument("--turn-radius-m", type=float, default=2.5)
    parser.add_argument("--cov-cell-m", type=float, default=0.35)
    parser.add_argument("--camera-hfov-deg", type=float, default=151.5)
    parser.add_argument("--camera-vfov-deg", type=float, default=131.3)
    parser.add_argument("--footprint-model", type=str, default="circle_min", choices=("ellipse", "circle_min", "circle_area"))
    parser.add_argument("--footprint-fov-scale", type=float, default=0.35)
    parser.add_argument("--sample-ds-m", type=float, default=2.0, help="Resampling distance along path for coverage stamps.")
    parser.add_argument("--pad-m", type=float, default=3.0)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--title",
        type=str,
        default="Coverage Scan Geometry (Bridge-Accurate: 100x100 m Example)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="runs/figures/figure_4_3_1_scan_geometry_bridge_100x100.png",
    )
    return parser.parse_args()


def _resample_polyline(path: np.ndarray, ds: float) -> np.ndarray:
    if len(path) <= 1:
        return path.copy()
    out = [path[0].copy()]
    step = max(1e-3, float(ds))
    for i in range(len(path) - 1):
        a = path[i]
        b = path[i + 1]
        seg = b[:2] - a[:2]
        seg_len = float(np.linalg.norm(seg))
        if seg_len <= 1e-9:
            continue
        n = max(1, int(np.ceil(seg_len / step)))
        for k in range(1, n + 1):
            t = float(k / n)
            p = (1.0 - t) * a + t * b
            out.append(p.astype(np.float32))
    return np.asarray(out, dtype=np.float32)


def _turn_vertices(path: np.ndarray, angle_thresh_rad: float = 0.2) -> np.ndarray:
    flags = np.zeros((len(path),), dtype=bool)
    for i in range(1, len(path) - 1):
        v0 = path[i] - path[i - 1]
        v1 = path[i + 1] - path[i]
        n0 = float(np.linalg.norm(v0[:2]))
        n1 = float(np.linalg.norm(v1[:2]))
        if n0 <= 1e-9 or n1 <= 1e-9:
            continue
        c = float(np.clip(np.dot(v0[:2], v1[:2]) / (n0 * n1), -1.0, 1.0))
        angle = float(np.arccos(c))
        if angle > float(angle_thresh_rad):
            flags[i] = True
    return flags


def _footprint_radii(
    alt_m: float,
    camera_hfov_deg: float,
    camera_vfov_deg: float,
    footprint_fov_scale: float,
) -> tuple[float, float]:
    hfov_half_rad = float(np.radians(max(1e-3, float(camera_hfov_deg)) * 0.5))
    vfov_half_rad = float(np.radians(max(1e-3, float(camera_vfov_deg)) * 0.5))
    rx = float(max(0.0, float(alt_m) * math.tan(hfov_half_rad) * float(footprint_fov_scale)))
    ry = float(max(0.0, float(alt_m) * math.tan(vfov_half_rad) * float(footprint_fov_scale)))
    return rx, ry


def _effective_radius(model: str, rx: float, ry: float) -> float:
    m = str(model).strip().lower()
    if m == "circle_min":
        return float(min(rx, ry))
    if m == "circle_area":
        return float(math.sqrt(max(0.0, rx * ry)))
    return float(math.sqrt(max(0.0, rx * ry)))


def main() -> None:
    args = parse_args()

    bounds_w = float(args.bounds_m[0])
    bounds_h = float(args.bounds_m[1])
    margin_m = float(args.margin_m)
    step_len = float(args.step_len_m)
    alt_m = float(args.alt_m)

    path_list = build_lawnmower_targets(
        bounds_w=bounds_w,
        bounds_h=bounds_h,
        margin=margin_m,
        step_len=step_len,
        alt_m=alt_m,
        turn_style=str(args.turn_style),
        turn_radius_m=float(args.turn_radius_m),
    )
    path = np.asarray(path_list, dtype=np.float32)
    if len(path) < 2:
        raise RuntimeError("Generated path has fewer than 2 points.")
    sampled = _resample_polyline(path, ds=float(args.sample_ds_m))
    turn_flags = _turn_vertices(path, angle_thresh_rad=0.2)
    turn_pts = path[np.flatnonzero(turn_flags), :2] if np.any(turn_flags) else np.zeros((0, 2), dtype=np.float32)

    rx, ry = _footprint_radii(
        alt_m=alt_m,
        camera_hfov_deg=float(args.camera_hfov_deg),
        camera_vfov_deg=float(args.camera_vfov_deg),
        footprint_fov_scale=float(args.footprint_fov_scale),
    )
    r_eff = _effective_radius(str(args.footprint_model), rx, ry)

    cov = CoverageGridTracker(
        width_m=bounds_w,
        height_m=bounds_h,
        cell_size_m=float(args.cov_cell_m),
        radius_m=max(0.0, r_eff),
        boundary_cells=2,
        track_pass_counts=True,
    )
    for p in sampled:
        x = float(p[0])
        y = float(p[1])
        model = str(args.footprint_model).strip().lower()
        if model == "ellipse":
            cov.update_ellipse(x, y, rx_m=rx, ry_m=ry, count_hits=True)
        elif model == "circle_area":
            cov.update_disk_radius(x, y, radius_m=float(math.sqrt(max(0.0, rx * ry))), count_hits=True)
        else:
            cov.update_disk_radius(x, y, radius_m=float(min(rx, ry)), count_hits=True)

    pass_counts = np.asarray(cov.pass_counts, dtype=np.int32)
    cs = float(cov.cell_size)
    x_min_cov = float(cov.x_min)
    y_min_cov = float(cov.y_min)

    x_min_scan, x_max_scan, y_min_scan, y_max_scan = bounds_minmax_xy(
        origin_x=0.0,
        origin_y=0.0,
        bounds_w=bounds_w,
        bounds_h=bounds_h,
        margin=margin_m,
    )

    fig, ax = plt.subplots(figsize=(10.2, 10.2), dpi=int(args.dpi))

    for ix in range(int(cov.nx)):
        x0 = x_min_cov + float(ix) * cs
        for iy in range(int(cov.ny)):
            n = int(pass_counts[ix, iy])
            if n <= 0:
                continue
            y0 = y_min_cov + float(iy) * cs
            if n == 1:
                face, alpha = "#9fdcff", 0.42
            else:
                face, alpha = "#f6c787", 0.54
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

    step_vis = max(1, int(len(sampled) // 90))
    for i, p in enumerate(sampled):
        if i % step_vis != 0:
            continue
        x, y = float(p[0]), float(p[1])
        model = str(args.footprint_model).strip().lower()
        if model == "ellipse":
            ax.add_patch(
                Ellipse(
                    (x, y),
                    width=2.0 * rx,
                    height=2.0 * ry,
                    facecolor="#9fdcff",
                    edgecolor="none",
                    alpha=0.08,
                    zorder=2,
                )
            )
        else:
            rr = float(r_eff if model == "circle_area" else min(rx, ry))
            ax.add_patch(
                Circle(
                    (x, y),
                    radius=rr,
                    facecolor="#9fdcff",
                    edgecolor="none",
                    alpha=0.08,
                    zorder=2,
                )
            )

    ax.add_patch(
        Rectangle(
            (-0.5 * bounds_w, -0.5 * bounds_h),
            bounds_w,
            bounds_h,
            fill=False,
            edgecolor="#666666",
            linewidth=1.6,
            linestyle="--",
            zorder=3,
        )
    )
    ax.add_patch(
        Rectangle(
            (x_min_scan, y_min_scan),
            float(x_max_scan - x_min_scan),
            float(y_max_scan - y_min_scan),
            fill=False,
            edgecolor="black",
            linewidth=2.4,
            zorder=4,
        )
    )

    xy = np.asarray(path[:, :2], dtype=np.float64)
    ax.plot(xy[:, 0], xy[:, 1], color="#1f77b4", linewidth=2.2, zorder=5)
    # Place arrows using local segment tangents to avoid fake diagonals from coarse jumps.
    if len(xy) >= 2:
        dxy = np.diff(xy, axis=0)
        seg_len = np.linalg.norm(dxy, axis=1)
        cum_s = np.concatenate(([0.0], np.cumsum(seg_len)))
        total_s = float(cum_s[-1])
        arrow_every_m = max(6.0, 0.75 * float(step_len))
        arrow_half_m = max(1.2, 0.20 * float(step_len))
        for s in np.arange(arrow_every_m, total_s, arrow_every_m):
            i = int(np.searchsorted(cum_s, s, side="right") - 1)
            i = int(np.clip(i, 0, len(seg_len) - 1))
            L = float(seg_len[i])
            if L <= 1e-9:
                continue
            t = float((s - float(cum_s[i])) / L)
            t = float(np.clip(t, 0.0, 1.0))
            p = xy[i] + t * (xy[i + 1] - xy[i])
            u = (xy[i + 1] - xy[i]) / L
            p0 = p - u * arrow_half_m
            p1 = p + u * arrow_half_m
            ax.annotate(
                "",
                xy=(float(p1[0]), float(p1[1])),
                xytext=(float(p0[0]), float(p0[1])),
                arrowprops={"arrowstyle": "->", "color": "#1f77b4", "lw": 1.3, "alpha": 0.85},
                zorder=6,
            )

    if len(turn_pts) > 0:
        ax.scatter(turn_pts[:, 0], turn_pts[:, 1], s=72, c="red", edgecolors="black", linewidths=0.6, zorder=7)

    ax.scatter(float(xy[0, 0]), float(xy[0, 1]), s=120, c="#1ca71c", edgecolors="black", linewidths=1.0, zorder=8)
    ax.scatter(float(xy[-1, 0]), float(xy[-1, 1]), s=120, c="#a020b0", edgecolors="black", linewidths=1.0, zorder=8)
    ax.text(float(xy[0, 0]) - 2.6, float(xy[0, 1]) - 4.5, "Start", fontsize=12, weight="bold")
    ax.text(float(xy[-1, 0]) - 2.0, float(xy[-1, 1]) + 3.0, "End", fontsize=12, weight="bold")

    model = str(args.footprint_model).strip().lower()
    info = (
        f"Example mission settings\n"
        f"bounds = {bounds_w:.0f}x{bounds_h:.0f} m, margin = {margin_m:.1f} m\n"
        f"lane spacing (step_len) = {step_len:.1f} m\n"
        f"altitude = {alt_m:.1f} m\n"
        f"HFOV/VFOV = {float(args.camera_hfov_deg):.1f}/{float(args.camera_vfov_deg):.1f} deg\n"
        f"footprint_fov_scale = {float(args.footprint_fov_scale):.2f}\n"
        f"footprint model = {model}\n"
        f"rx = {rx:.2f} m, ry = {ry:.2f} m, effective radius = {r_eff:.2f} m\n"
        f"cell size = {cs:.2f} m, sampled points = {len(sampled)}"
    )
    ax.text(
        x_min_scan,
        y_min_scan - 8.8,
        info,
        fontsize=11.0,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.95, "edgecolor": "#333333"},
    )

    x0 = -0.5 * bounds_w - float(args.pad_m)
    x1 = +0.5 * bounds_w + float(args.pad_m)
    y0 = -0.5 * bounds_h - float(args.pad_m)
    y1 = +0.5 * bounds_h + float(args.pad_m)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)", fontsize=13)
    ax.set_ylabel("y (m)", fontsize=13)
    ax.set_title(str(args.title), fontsize=18, pad=12)
    ax.grid(True, linestyle=":", alpha=0.25)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    print(f"[scan_geometry_bridge_figure] wrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()
