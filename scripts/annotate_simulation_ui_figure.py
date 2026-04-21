#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


@dataclass(frozen=True)
class Callout:
    label: str
    color: str
    box: tuple[float, float, float, float]  # x, y, w, h in normalized coordinates


def _parse_box(raw: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("box must contain 4 comma-separated values: x,y,w,h")
    try:
        values = tuple(float(v) for v in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("box values must be numeric") from exc
    if any(v < 0 or v > 1 for v in values):
        raise argparse.ArgumentTypeError("all box values must be in [0, 1]")
    x, y, w, h = values
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("box width/height must be > 0")
    if x + w > 1 or y + h > 1:
        raise argparse.ArgumentTypeError("box must remain within [0, 1] image bounds")
    return values


def _to_pixels(
    box: tuple[float, float, float, float],
    width_px: int,
    height_px: int,
) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return x * width_px, y * height_px, w * width_px, h * height_px


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Create a thesis-ready Figure 11 from a simulation UI screenshot by adding "
            "callouts for map, mission planning tools, and telemetry/status panels."
        )
    )
    p.add_argument("--input", type=Path, required=True, help="Input UI screenshot image path.")
    p.add_argument(
        "--allow-report-image",
        action="store_true",
        help="Allow report/plot images as input (disabled by default to avoid wrong Figure 11 backgrounds).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("runs/figures/figure_6_6_simulation_workflow_ui.png"),
        help="Output PNG path.",
    )
    p.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("runs/figures/figure_6_6_simulation_workflow_ui.pdf"),
        help="Output PDF path.",
    )
    p.add_argument("--dpi", type=int, default=300, help="Export DPI.")
    p.add_argument(
        "--title",
        type=str,
        default="Simulation Workflow UI: Map, Planning, and Telemetry/Status",
        help="Figure title.",
    )
    p.add_argument(
        "--planning-box",
        type=_parse_box,
        default=(0.01, 0.17, 0.23, 0.78),
        help="Normalized box for mission planning tools: x,y,w,h",
    )
    p.add_argument(
        "--map-box",
        type=_parse_box,
        default=(0.25, 0.17, 0.49, 0.78),
        help="Normalized box for map area: x,y,w,h",
    )
    p.add_argument(
        "--telemetry-box",
        type=_parse_box,
        default=(0.75, 0.17, 0.24, 0.78),
        help="Normalized box for telemetry/status panels: x,y,w,h",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    in_path = args.input.resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"input screenshot not found: {in_path}")

    lower_name = in_path.name.lower()
    banned_tokens = (
        "website_sim_report",
        "website_single_path",
        "website_coverage_over_time",
        "sequence_result_summary",
        "scan_geometry",
    )
    if (not args.allow_report_image) and any(t in lower_name for t in banned_tokens):
        raise ValueError(
            "input appears to be a generated report/plot image, not a raw UI screenshot. "
            "Capture the real /sim UI page first (for example runs/figures/sim_ui_raw.png), "
            "or pass --allow-report-image to bypass this safeguard."
        )

    image = mpimg.imread(in_path)
    height_px, width_px = image.shape[0], image.shape[1]

    # Keep A4-friendly width while preserving aspect ratio.
    fig_w = 12.0
    fig_h = fig_w * (height_px / width_px)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=args.dpi)
    ax.imshow(image)
    ax.set_xlim(0, width_px)
    ax.set_ylim(height_px, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(args.title, fontsize=15, pad=10, fontweight="semibold")

    callouts = [
        Callout("Mission Planning Tools", "#1f4e79", args.planning_box),
        Callout("Mission Map Workspace", "#2e7d32", args.map_box),
        Callout("Telemetry and Status Panels", "#7f5539", args.telemetry_box),
    ]

    for item in callouts:
        x, y, w, h = _to_pixels(item.box, width_px, height_px)
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                linewidth=2.4,
                edgecolor=item.color,
                facecolor=item.color,
                alpha=0.08,
            )
        )
        ax.text(
            x + 10,
            max(22, y - 14),
            item.label,
            fontsize=11.5,
            color="black",
            ha="left",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": item.color,
                "linewidth": 1.4,
                "alpha": 0.92,
            },
        )

    ax.text(
        0.01,
        0.02,
        "Section 6.6 | Operator workflow view for simulation mission planning, execution, and monitoring.",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#333333",
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.85},
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output_pdf, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[ok] saved PNG: {args.output.resolve()}")
    print(f"[ok] saved PDF: {args.output_pdf.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
