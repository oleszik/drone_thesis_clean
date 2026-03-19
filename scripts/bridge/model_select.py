from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quad_rl.utils.paths import normalize_model_path
from quad_rl.utils.scan_scale_profile import (
    apply_scan_obs_profile,
    assert_scan_obs_profile,
    get_scan_path_scale_upper,
    resolve_scan_production_model_path,
)


def selected_profile_for_scale(scale: float) -> str:
    return "patch7_scale2" if float(scale) >= 2.0 else "patch5_default"


def iter_opt_summary_paths() -> list[Path]:
    root = Path("runs") / "ardupilot_scan_opt_suite"
    if not root.exists():
        return []
    return sorted(root.glob("*/opt_summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def pinned_opt_summary_path_for_profile(profile_name: str) -> Path | None:
    root = Path("runs") / "production_ardupilot_defaults"
    if not root.exists():
        return None
    bucket = "scale2" if str(profile_name) == "patch7_scale2" else "scale1"
    path = root / f"{bucket}_opt_summary.json"
    return path if path.exists() else None


def winner_from_opt_summary(data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    defaults: dict[str, Any] = {
        "lookahead_enable": 1,
        "step_sched_enable": 1,
        "lane_keep_enable": 0,
        "adaptive_tracking": 0,
        "lane_kp": 0.4,
        "corner_vxy_cap": 0.8,
    }
    winner_arm = "A_baseline"
    ranked = data.get("ranked", [])
    if not isinstance(ranked, list) or not ranked:
        return winner_arm, defaults
    top = ranked[0]
    if not isinstance(top, dict):
        return winner_arm, defaults
    winner_arm = str(top.get("arm", winner_arm))
    cfg = top.get("config", {})
    if isinstance(cfg, dict):
        for k in ("lookahead_enable", "step_sched_enable", "lane_keep_enable", "adaptive_tracking"):
            if k in cfg:
                defaults[k] = int(cfg[k])
        if "lane_kp" in cfg:
            defaults["lane_kp"] = float(cfg["lane_kp"])
        if "corner_vxy_cap" in cfg:
            defaults["corner_vxy_cap"] = float(cfg["corner_vxy_cap"])
    return winner_arm, defaults


def summary_selected_profile(data: dict[str, Any]) -> str | None:
    meta = data.get("metadata", {})
    if isinstance(meta, dict):
        sel = str(meta.get("selected_profile", "")).strip()
        if sel:
            return sel
        scale_meta = meta.get("scan_path_len_scale", None)
        if scale_meta is not None:
            try:
                return selected_profile_for_scale(float(scale_meta))
            except Exception:
                pass
    cfg = data.get("config", {})
    if isinstance(cfg, dict):
        scale_cfg = cfg.get("scan_path_len_scale", None)
        if scale_cfg is not None:
            try:
                return selected_profile_for_scale(float(scale_cfg))
            except Exception:
                return None
    return None


def resolve_sitl_recommended_profile(source: str, scale_upper: float) -> tuple[str, dict[str, Any], str, bool]:
    expected_profile = selected_profile_for_scale(float(scale_upper))
    src_raw = str(source or "latest").strip()
    src = src_raw.lower()

    if src.startswith("path:"):
        raw_path = src_raw[5:].strip()
        p = Path(raw_path).expanduser()
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                arm, cfg = winner_from_opt_summary(data)
                prof = summary_selected_profile(data)
                return arm, cfg, f"path:{p}", bool(prof == expected_profile)
            except Exception:
                pass
        arm, cfg = winner_from_opt_summary({})
        return arm, cfg, f"path:{p}", False

    if src not in {"latest", ""}:
        src = "latest"

    pinned = pinned_opt_summary_path_for_profile(expected_profile)
    if pinned is not None:
        try:
            data = json.loads(pinned.read_text(encoding="utf-8"))
            prof = summary_selected_profile(data)
            if prof == expected_profile:
                arm, cfg = winner_from_opt_summary(data)
                return arm, cfg, f"latest:pinned:{pinned}", True
        except Exception:
            pass

    for p in iter_opt_summary_paths():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            prof = summary_selected_profile(data)
            if prof != expected_profile:
                continue
            arm, cfg = winner_from_opt_summary(data)
            return arm, cfg, f"latest:{p}", True
        except Exception:
            continue

    arm, cfg = winner_from_opt_summary({})
    return arm, cfg, "latest:fallback", False


def resolve_runtime_model(task_key: str, model_arg: str, cfg, ctx: str) -> tuple[str, str]:
    model_path = ""
    selected_profile = "manual"
    auto_tokens = {"", "auto", "production", "production_scan"}
    raw = (model_arg or "").strip()
    if task_key == "scan" and raw.lower() in auto_tokens:
        model_path_obj, profile = resolve_scan_production_model_path(cfg)
        apply_scan_obs_profile(cfg, profile)
        assert_scan_obs_profile(cfg, profile, ctx=ctx)
        model_path = str(model_path_obj)
        selected_profile = str(profile.name)
    elif raw:
        model_path = normalize_model_path(raw)
    return selected_profile, model_path


def scan_path_scale_upper(cfg) -> float:
    return float(get_scan_path_scale_upper(cfg))
