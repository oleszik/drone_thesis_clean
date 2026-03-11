from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScanScaleProfile:
    name: str
    min_path_scale: float
    obs_patch_size: int
    obs_boundary_feat: bool
    obs_global_coverage_enable: bool
    obs_global_size: int
    model_candidates: tuple[str, ...]


SCAN_PROFILE_PATCH5 = ScanScaleProfile(
    name="patch5_default",
    min_path_scale=0.0,
    obs_patch_size=5,
    obs_boundary_feat=True,
    obs_global_coverage_enable=False,
    obs_global_size=8,
    model_candidates=(
        "runs/production_scan_v4/best_model.zip",
        "runs/production_scan/best_model.zip",
    ),
)

SCAN_PROFILE_PATCH7_SCALE2 = ScanScaleProfile(
    name="patch7_scale2",
    min_path_scale=2.0,
    obs_patch_size=7,
    obs_boundary_feat=True,
    obs_global_coverage_enable=False,
    obs_global_size=8,
    model_candidates=(
        "runs/production_scan_v4_scale2/best_model.zip",
        "runs/scan_scale2_C_patch7_ft300k/best_model.zip",
    ),
)


def get_scan_path_scale_upper(cfg) -> float:
    vals = [
        float(getattr(cfg, "scan_path_len_scale", 1.0)),
        float(getattr(cfg, "scan_path_len_scale_min", 1.0)),
        float(getattr(cfg, "scan_path_len_scale_max", 1.0)),
    ]
    return max(vals)


def select_scan_scale_profile(cfg) -> ScanScaleProfile:
    path_scale = get_scan_path_scale_upper(cfg)
    if path_scale >= SCAN_PROFILE_PATCH7_SCALE2.min_path_scale:
        return SCAN_PROFILE_PATCH7_SCALE2
    return SCAN_PROFILE_PATCH5


def apply_scan_obs_profile(cfg, profile: ScanScaleProfile) -> None:
    setattr(cfg, "scan_obs_aug_enable", True)
    setattr(cfg, "scan_obs_patch_size", int(profile.obs_patch_size))
    setattr(cfg, "scan_obs_boundary_feat", bool(profile.obs_boundary_feat))
    setattr(cfg, "scan_obs_global_coverage_enable", bool(profile.obs_global_coverage_enable))
    setattr(cfg, "scan_obs_global_size", int(profile.obs_global_size))


def assert_scan_obs_profile(cfg, profile: ScanScaleProfile, ctx: str) -> None:
    got_enable = bool(getattr(cfg, "scan_obs_aug_enable", False))
    got_patch = int(getattr(cfg, "scan_obs_patch_size", 5))
    got_boundary = bool(getattr(cfg, "scan_obs_boundary_feat", True))
    got_global = bool(getattr(cfg, "scan_obs_global_coverage_enable", False))
    got_global_size = int(getattr(cfg, "scan_obs_global_size", 8))
    exp_enable = True
    exp_patch = int(profile.obs_patch_size)
    exp_boundary = bool(profile.obs_boundary_feat)
    exp_global = bool(profile.obs_global_coverage_enable)
    exp_global_size = int(profile.obs_global_size)
    if (
        got_enable == exp_enable
        and got_patch == exp_patch
        and got_boundary == exp_boundary
        and got_global == exp_global
        and got_global_size == exp_global_size
    ):
        return
    raise ValueError(
        f"[{ctx}] Scan obs profile mismatch for profile={profile.name}: "
        f"expected(enable={exp_enable}, patch={exp_patch}, boundary_feat={exp_boundary}, "
        f"global_cov={exp_global}, global_size={exp_global_size}) "
        f"got(enable={got_enable}, patch={got_patch}, boundary_feat={got_boundary}, "
        f"global_cov={got_global}, global_size={got_global_size})"
    )


def resolve_scan_production_model_path(cfg) -> tuple[Path, ScanScaleProfile]:
    profile = select_scan_scale_profile(cfg)
    for raw in profile.model_candidates:
        p = Path(raw)
        if p.exists():
            return p, profile
    return Path(profile.model_candidates[0]), profile


def scan_step_scaling_details(cfg) -> tuple[int, float, int, bool]:
    base_steps = int(getattr(cfg, "scan_max_steps", getattr(cfg, "max_steps", 300)))
    path_scale = get_scan_path_scale_upper(cfg)
    if not bool(getattr(cfg, "scan_scale_max_steps_with_path", False)):
        return base_steps, path_scale, base_steps, False
    ref = float(getattr(cfg, "scan_path_len_scale_ref", 1.0))
    if ref <= 1e-6:
        ref = 1.0
    ratio = max(1.0, path_scale / ref)
    cap_hit = False
    ratio_cap = float(getattr(cfg, "scan_max_steps_scale_cap", 0.0))
    if ratio_cap > 0.0:
        cap_hit = ratio > ratio_cap
        ratio = min(ratio, ratio_cap)
    eff = int(max(1, round(base_steps * ratio)))
    return base_steps, path_scale, eff, bool(cap_hit)


def effective_scan_max_steps(cfg) -> int:
    _, _, eff, _ = scan_step_scaling_details(cfg)
    return eff
