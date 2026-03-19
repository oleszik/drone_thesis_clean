from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import Request


def encode_sse(*, data: Any, event: str | None = None, event_id: str | None = None) -> bytes:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    lines: list[str] = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    if event is not None:
        lines.append(f"event: {event}")
    for line in payload.splitlines() or ["{}"]:
        lines.append(f"data: {line}")
    lines.append("")
    return ("\n".join(lines) + "\n").encode("utf-8")


def encode_sse_comment(text: str = "ping") -> bytes:
    return f": {text}\n\n".encode("utf-8")


def stable_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


async def sleep_or_disconnect(request: Request, seconds: float) -> bool:
    await asyncio.sleep(max(0.0, float(seconds)))
    return await request.is_disconnected()
