from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pin a scale-bucket SITL recommended opt summary.")
    parser.add_argument("--scale-bucket", type=str, required=True, choices=("scale1", "scale2"))
    parser.add_argument("--source", type=str, default="latest", help="latest or path:<opt_summary.json>")
    return parser.parse_args(argv)


def _expected_profile(scale_bucket: str) -> str:
    return "patch7_scale2" if str(scale_bucket) == "scale2" else "patch5_default"


def _selected_profile_from_summary(data: dict) -> str | None:
    meta = data.get("metadata", {})
    if isinstance(meta, dict):
        sel = str(meta.get("selected_profile", "")).strip()
        if sel:
            return sel
        if "scan_path_len_scale" in meta:
            try:
                return "patch7_scale2" if float(meta["scan_path_len_scale"]) >= 2.0 else "patch5_default"
            except Exception:
                pass
    cfg = data.get("config", {})
    if isinstance(cfg, dict) and "scan_path_len_scale" in cfg:
        try:
            return "patch7_scale2" if float(cfg["scan_path_len_scale"]) >= 2.0 else "patch5_default"
        except Exception:
            pass
    return None


def _find_latest_matching(expected_profile: str) -> Path | None:
    root = Path("runs") / "ardupilot_scan_opt_suite"
    if not root.exists():
        return None
    cands = sorted(root.glob("*/opt_summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in cands:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if _selected_profile_from_summary(data) == expected_profile:
                return p
        except Exception:
            continue
    return None


def main() -> None:
    args = parse_args()
    expected_profile = _expected_profile(args.scale_bucket)
    source_raw = str(args.source or "latest").strip()
    source_lower = source_raw.lower()

    if source_lower == "latest":
        src = _find_latest_matching(expected_profile)
        if src is None:
            raise SystemExit(f"No matching opt_summary.json found for profile={expected_profile}")
    elif source_lower.startswith("path:"):
        raw = source_raw[5:].strip()
        src = Path(raw).expanduser()
        if not src.exists():
            raise SystemExit(f"Source file not found: {src}")
    else:
        raise SystemExit("Invalid --source. Use latest or path:<opt_summary.json>")

    data = json.loads(src.read_text(encoding="utf-8"))
    selected_profile = _selected_profile_from_summary(data)
    if selected_profile != expected_profile:
        raise SystemExit(
            f"Source profile mismatch: expected={expected_profile} got={selected_profile or 'unknown'} source={src}"
        )

    out_root = Path("runs") / "production_ardupilot_defaults"
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{args.scale_bucket}_opt_summary.json"
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print(f"[bless] pinned: {out_path}")
    print(f"[bless] source: {src}")
    print(f"[bless] profile: {expected_profile}")
    print(f"[bless] usage: --sitl-recommended-source path:{out_path}")
    print(f"[bless] done at {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
