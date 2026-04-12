from __future__ import annotations

from typing import Any, Callable

from fastapi import HTTPException


class AutonomyGuard:
    def __init__(self, snapshot_provider: Callable[[], dict[str, Any]]) -> None:
        self._snapshot_provider = snapshot_provider

    def snapshot(self) -> dict[str, Any]:
        return self._snapshot_provider()

    def require(self, action: str) -> dict[str, Any]:
        snap = self.snapshot()
        if bool(snap.get("can_autonomous")):
            return snap
        raise HTTPException(
            status_code=409,
            detail={
                "error": f"autonomy_blocked:{action}",
                "blocking_reasons": list(snap.get("blocking_reasons") or []),
                "readiness_snapshot": snap,
            },
        )
