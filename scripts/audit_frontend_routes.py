#!/usr/bin/env python3
"""Lightweight frontend route audit for sim/real endpoint mixups."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_SRC = REPO_ROOT / "frontend" / "src"

REAL_DANGEROUS = ["/api/sim/", "/api/mission/", "/api/control/"]
SIM_DANGEROUS = ["/api/real/"]


def classify_file(path: Path, text: str) -> tuple[bool, bool]:
    rel = str(path.relative_to(FRONTEND_SRC)).lower()
    is_real = "real" in rel or "variant=\"real\"" in text or "variant='real'" in text
    is_sim = "sim" in rel or "variant=\"sim\"" in text or "variant='sim'" in text
    return is_real, is_sim


def find_hits(text: str, needle: str) -> list[int]:
    return [m.start() for m in re.finditer(re.escape(needle), text)]


def line_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def audit_file(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="ignore")
    is_real, is_sim = classify_file(path, text)
    warnings: list[str] = []

    # Shared files that intentionally contain both real/sim branches are skipped
    # to reduce false positives.
    if is_real and is_sim:
        return warnings

    if is_real:
        for needle in REAL_DANGEROUS:
            for hit in find_hits(text, needle):
                warnings.append(f"REAL_WARN {path.relative_to(REPO_ROOT)}:{line_for_offset(text, hit)} uses '{needle}'")

    if is_sim:
        for needle in SIM_DANGEROUS:
            for hit in find_hits(text, needle):
                warnings.append(f"SIM_WARN {path.relative_to(REPO_ROOT)}:{line_for_offset(text, hit)} uses '{needle}'")

    return warnings


def main() -> int:
    if not FRONTEND_SRC.exists():
        print(f"error: missing {FRONTEND_SRC}")
        return 2

    files = sorted([p for p in FRONTEND_SRC.rglob("*") if p.suffix in {".js", ".jsx", ".ts", ".tsx"}])
    all_warnings: list[str] = []
    for path in files:
        all_warnings.extend(audit_file(path))

    if not all_warnings:
        print("route audit: OK (no dangerous route mixups found)")
        return 0

    print("route audit: warnings found")
    for line in all_warnings:
        print(line)
    print(f"total warnings: {len(all_warnings)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
