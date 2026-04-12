from __future__ import annotations

from typing import Literal

RuntimeMode = Literal["simulation", "real_mission"]


def normalize_runtime_mode(raw: str | None) -> RuntimeMode:
    v = str(raw or "").strip().lower()
    if v == "real_mission":
        return "real_mission"
    return "simulation"


def resolve_transport_url(*, runtime_mode: RuntimeMode, configured_url: str) -> str:
    # Keep behavior compatible: use configured URL directly.
    # This helper centralizes runtime-mode-aware transport resolution.
    _ = runtime_mode
    return str(configured_url)
