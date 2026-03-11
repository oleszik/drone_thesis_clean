from __future__ import annotations

from typing import Any


def _parse_bool(text: str) -> bool:
    v = text.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid bool value: {text!r}")


def _coerce_from_value(raw: str, current: Any) -> Any:
    if isinstance(current, tuple):
        token = raw.strip()
        parts = [p.strip() for p in token.split(",") if p.strip()]
        if len(parts) == len(current):
            out = []
            for i, part in enumerate(parts):
                out.append(_coerce_from_value(part, current[i]))
            return tuple(out)
        return token
    if isinstance(current, bool):
        return _parse_bool(raw)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if current is None:
        token = raw.strip()
        low = token.lower()
        if low in {"none", "null"}:
            return None
        try:
            return _parse_bool(token)
        except ValueError:
            pass
        try:
            return int(token)
        except ValueError:
            pass
        try:
            return float(token)
        except ValueError:
            return token
    return raw


def parse_override_pairs(items: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        text = (item or "").strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        k, v = text.split("=", 1)
        key = k.strip()
        if not key:
            raise ValueError(f"Override key is empty in: {item!r}")
        out[key] = v.strip()
    return out


def apply_overrides(cfg: Any, overrides: dict[str, str]) -> dict[str, Any]:
    applied: dict[str, Any] = {}
    for key, raw in overrides.items():
        if not hasattr(cfg, key):
            raise KeyError(f"Unknown preset field override: {key!r}")
        current = getattr(cfg, key)
        coerced = _coerce_from_value(raw, current)
        setattr(cfg, key, coerced)
        applied[key] = coerced
    return applied
