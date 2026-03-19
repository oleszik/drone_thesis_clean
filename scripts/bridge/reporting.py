from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_summary(summary_path: Path, summary: dict[str, Any]) -> None:
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def print_summary_paths(telemetry_path: Path, summary_path: Path) -> None:
    print(f"[bridge] telemetry: {telemetry_path}")
    print(f"[bridge] summary:   {summary_path}")


def print_summary_payload(summary: dict[str, Any]) -> None:
    print(f"[bridge] result: {json.dumps(summary, indent=2)}")
