from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report scan gate coverage threshold hit rates from gate JSON files.")
    parser.add_argument(
        "--json",
        action="append",
        default=[],
        help="Gate summary JSON path (repeatable).",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="",
        help="Optional glob pattern for gate JSON files (e.g. runs/scan_*/gates/*.json).",
    )
    return parser.parse_args()


def _load_paths(args: argparse.Namespace) -> list[Path]:
    out: list[Path] = []
    for p in args.json:
        path = Path(p)
        if path.exists():
            out.append(path)
    if args.glob.strip():
        out.extend(sorted(Path().glob(args.glob.strip())))
    uniq: dict[str, Path] = {}
    for p in out:
        uniq[str(p.resolve())] = p
    return list(uniq.values())


def _rate_from_summary(summary: dict, threshold: float) -> float:
    coverage_episode = summary.get("coverage_episode")
    if isinstance(coverage_episode, list) and len(coverage_episode) > 0:
        arr = np.asarray(coverage_episode, dtype=np.float32)
        return float(np.mean(arr >= threshold))
    key = f"coverage_ge_{int(round(threshold * 100)):03d}_rate"
    if key in summary:
        return float(summary[key])
    return float("nan")


def main() -> None:
    args = parse_args()
    paths = _load_paths(args)
    if not paths:
        raise SystemExit("No gate JSON files found. Provide --json and/or --glob.")

    print("file,success_count,crash_count,coverage_mean,pct_ge_090,pct_ge_093,pct_ge_095")
    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        success = int(summary.get("success_count", 0))
        crash = int(summary.get("crash_count", 0))
        cov_mean = float(summary.get("coverage_mean", float("nan")))
        r90 = _rate_from_summary(summary, 0.90)
        r93 = _rate_from_summary(summary, 0.93)
        r95 = _rate_from_summary(summary, 0.95)
        print(
            f"{path},{success},{crash},{cov_mean:.6f},"
            f"{100.0*r90:.2f},{100.0*r93:.2f},{100.0*r95:.2f}"
        )


if __name__ == "__main__":
    main()
