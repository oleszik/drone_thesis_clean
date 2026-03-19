from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


def safe_slug(text: str) -> str:
    raw = (text or "").strip()
    slug = "".join(ch if (ch.isalnum() or ch in "-_") else "-" for ch in raw)
    slug = slug.strip("-_")
    return slug or "run"


def next_free_dir(path: Path) -> Path:
    if not path.exists():
        return path
    i = 1
    while True:
        cand = path.with_name(f"{path.name}_{i}")
        if not cand.exists():
            return cand
        i += 1


@dataclass(frozen=True)
class BridgeRunPaths:
    run_dir: Path
    telemetry_path: Path
    summary_path: Path


def build_run_paths(selected_profile: str, root: Path | None = None) -> BridgeRunPaths:
    run_root = (root or (Path("runs") / "ardupilot_scan"))
    run_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = next_free_dir(run_root / f"{stamp}_{safe_slug(selected_profile)}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return BridgeRunPaths(
        run_dir=run_dir,
        telemetry_path=run_dir / "telemetry.jsonl",
        summary_path=run_dir / "summary.json",
    )
