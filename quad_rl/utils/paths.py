from __future__ import annotations

from pathlib import Path


def normalize_model_path(path: str) -> str:
    """Normalize model paths and avoid `.zip.zip` mistakes."""
    raw = str(path).strip()
    if raw.endswith(".zip.zip"):
        raw = raw[:-4]

    p = Path(raw)
    if p.suffix != ".zip":
        candidate = Path(f"{raw}.zip")
        if candidate.exists() or not p.exists():
            p = candidate
    return str(p)
