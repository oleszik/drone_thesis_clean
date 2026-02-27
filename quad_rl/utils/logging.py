from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def ensure_run_dir(path: str | Path) -> Path:
    run_dir = Path(path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "tb").mkdir(parents=True, exist_ok=True)
    return run_dir


def dump_json(path: str | Path, payload: Any) -> None:
    obj = asdict(payload) if is_dataclass(payload) else payload
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
