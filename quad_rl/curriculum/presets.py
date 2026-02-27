from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Dict


@dataclass
class PresetConfig:
    name: str
    dt: float = 0.1
    max_steps: int = 300
    seq_max_steps: int = 800

    dynamics_k_vel: float = 2.0
    dynamics_tau_yaw: float = 0.4
    dynamics_g: float = 9.81
    gravity_compensation: float = 1.0

    v_xy_max: float = 0.8
    v_z_max: float = 0.6
    yaw_rate_max: float = 1.0

    world_xy_bound: float = 20.0
    world_z_min: float = 0.0
    world_z_max: float = 8.0
    allow_oob_touch: bool = False
    allow_oob_touch_scan: bool | None = None
    allow_oob_touch_sequence: bool | None = None
    k_oob_touch: float = 0.0
    k_oob_touch_step: float = 0.0
    crash_tilt_limit: float = 1.25
    crash_penalty: float = 25.0
    step_penalty: float = 0.01

    hover_success_radius: float = 0.35
    hover_hold_steps: int = 12

    yaw_success_radius: float = 0.4
    yaw_hold_steps: int = 12
    yaw_tolerance: float = 0.25
    yaw_target_range: float = 1.57

    landing_target_z: float = 0.05
    landing_xy_radius: float = 0.45
    landing_vz_thresh: float = 0.25
    landing_hold_steps: int = 3

    waypoint_hop_min: float = 0.5
    waypoint_hop_max: float = 2.5
    waypoint_hold_steps: int = 10
    waypoint_success_radius: float = 0.8
    waypoint_k_prog: float = 4.0
    waypoint_hold_decay: float = 1.0
    waypoint_hold_reset_margin: float = 0.1
    waypoint_k_stay: float = 0.2
    waypoint_k_v_in: float = 0.05
    waypoint_lock_altitude: bool = True

    seq_n_waypoints: int = 6
    seq_hop_min: float = 0.5
    seq_hop_max: float = 2.5
    seq_hold_steps: int = 10
    seq_success_radius: float = 0.8
    seq_k_prog: float = 2.0
    seq_hold_decay: float = 1.0
    seq_hold_reset_margin: float = 0.1
    seq_k_stay: float = 0.2
    seq_k_v_in: float = 0.05
    seq_wall_margin: float = 0.0
    seq_k_wall: float = 0.0
    seq_k_wall_vel: float = 0.0
    seq_wall_vmin_scale: float = 1.0
    seq_wall_power: float = 1.0
    seq_k_oob_touch: float = 0.0
    seq_lock_altitude: bool = True

    scan_rows: int = 4
    scan_cols: int = 4
    scan_spacing: float = 0.8
    scan_hold_steps: int = 3
    scan_max_steps: int = 300
    scan_lookahead: float = 1.0
    scan_k_prog: float = 2.0
    scan_k_ct: float = 0.5
    scan_k_yaw: float = 0.2
    scan_k_la: float = 0.0
    scan_la_clip: float = 3.0
    scan_min_wp_spacing: float = 0.8
    scan_v_xy_max: float | None = None
    scan_yaw_rate_max: float | None = None
    scan_disable_wall_clamp: bool = False
    scan_wall_margin: float | None = None
    scan_wall_vmin_scale: float | None = None
    scan_wall_power: float | None = None
    scan_path_len_scale: float = 1.0
    scan_path_len_scale_min: float = 1.0
    scan_path_len_scale_max: float = 1.0
    scan_hold_turns_only: bool = False
    scan_turn_hold_steps: int = 3
    scan_turn_hold_radius: float = 0.8
    scan_turn_hold_decay: int = 1
    scan_turn_v_in_max: float = 0.2
    scan_turn_v_penalty: float = 0.0
    scan_turn_v_target: float = 0.2
    scan_k_act: float = 0.0
    scan_prog_eps: float = 0.01
    scan_k_stuck: float = 0.0
    scan_start_on_path: bool = False
    scan_start_xy_noise: float = 0.0
    scan_start_yaw_noise: float = 0.0
    scan_spacing_jitter: float = 0.0
    scan_lookahead_min: float = 1.0
    scan_lookahead_max: float = 1.0
    scan_cov_cell_size: float = 0.25
    scan_coverage_radius: float = 0.5
    scan_k_cov_gain: float = 0.0
    scan_k_cov_revisit: float = 0.0
    scan_k_yawrate: float = 0.0
    scan_k_dvxy: float = 0.0
    scan_z_min: float | None = None
    scan_z_max: float | None = None
    scan_z_cmd_min: float | None = None
    scan_z_cmd_max: float | None = None
    scan_z_target: float | None = None
    scan_k_z_p: float = 0.0
    scan_oob_grace_steps: int = 0
    scan_cov_late_thresh: float = 1.0
    scan_k_cov_gain_late: float = 0.0
    scan_k_cov_stall: float = 0.0
    scan_debug_oob: bool = False

    mission_takeoff_height: float = 1.0
    mission_transit_hop: float = 3.0


PRESETS: Dict[str, PresetConfig] = {
    "A0": PresetConfig(
        name="A0",
        v_xy_max=0.8,
        v_z_max=0.5,
        waypoint_hop_min=0.5,
        waypoint_hop_max=2.0,
        seq_n_waypoints=4,
        seq_hop_min=0.5,
        seq_hop_max=2.0,
        seq_hold_steps=8,
        seq_success_radius=0.9,
        seq_k_prog=3.0,
    ),
    "A0_S2": PresetConfig(
        name="A0_S2",
        v_xy_max=0.8,
        v_z_max=0.5,
        seq_n_waypoints=6,
        seq_hop_min=0.5,
        seq_hop_max=2.5,
        seq_hold_steps=10,
        seq_success_radius=0.8,
        seq_k_prog=3.0,
    ),
    "A1": PresetConfig(
        name="A1",
        v_xy_max=0.8,
        v_z_max=0.6,
        waypoint_hop_min=1.0,
        waypoint_hop_max=3.5,
        waypoint_success_radius=0.75,
        seq_n_waypoints=8,
        seq_hop_min=1.0,
        seq_hop_max=3.5,
        seq_hold_steps=10,
        seq_success_radius=0.8,
        seq_k_prog=2.0,
    ),
    "A1_S3": PresetConfig(
        name="A1_S3",
        v_xy_max=0.8,
        v_z_max=0.6,
        seq_n_waypoints=10,
        seq_hop_min=1.0,
        seq_hop_max=4.0,
        seq_hold_steps=10,
        seq_success_radius=0.9,
        seq_k_prog=2.0,
        seq_max_steps=800,
    ),
    "A1_S3b": PresetConfig(
        name="A1_S3b",
        v_xy_max=1.0,
        v_z_max=0.6,
        seq_n_waypoints=10,
        seq_hop_min=1.0,
        seq_hop_max=4.0,
        seq_hold_steps=1,
        seq_success_radius=1.0,
        seq_k_prog=10.0,
        seq_max_steps=800,
    ),
    "A2": PresetConfig(
        name="A2",
        v_xy_max=0.8,
        v_z_max=0.7,
        allow_oob_touch=True,
        allow_oob_touch_scan=False,
        allow_oob_touch_sequence=True,
        k_oob_touch=0.2,
        k_oob_touch_step=0.01,
        waypoint_hop_min=2.0,
        waypoint_hop_max=5.0,
        waypoint_success_radius=0.7,
        seq_n_waypoints=12,
        seq_hop_min=2.0,
        seq_hop_max=5.0,
        seq_hold_steps=15,
        seq_success_radius=0.84525,
        seq_hold_reset_margin=0.15,
        seq_k_v_in=0.0625,
        seq_wall_margin=2.0,
        seq_k_wall=0.5,
        seq_k_wall_vel=0.1,
        seq_wall_vmin_scale=0.15,
        seq_wall_power=3.0,
        seq_k_oob_touch=1.0,
        seq_k_prog=2.0,
        scan_max_steps=2400,
        scan_lookahead=1.5,
        scan_k_prog=8.0,
        scan_k_ct=0.1,
        scan_k_yaw=0.0,
        scan_k_la=0.5,
        scan_la_clip=3.0,
        scan_min_wp_spacing=1.0,
        scan_v_xy_max=1.0,
        scan_yaw_rate_max=0.6,
        scan_disable_wall_clamp=False,
        scan_wall_margin=3.0,
        scan_wall_vmin_scale=0.10,
        scan_wall_power=4.0,
        scan_path_len_scale=1.0,
        scan_path_len_scale_min=0.7,
        scan_path_len_scale_max=1.0,
        scan_hold_turns_only=True,
        scan_turn_hold_steps=5,
        scan_turn_hold_radius=1.0,
        scan_turn_hold_decay=1,
        scan_turn_v_in_max=0.2,
        scan_turn_v_penalty=0.05,
        scan_turn_v_target=0.2,
        scan_k_act=0.0,
        scan_prog_eps=0.01,
        scan_k_stuck=0.2,
        scan_start_on_path=True,
        scan_start_xy_noise=0.05,
        scan_start_yaw_noise=0.1,
        scan_spacing_jitter=0.05,
        scan_lookahead_min=1.2,
        scan_lookahead_max=1.8,
        scan_cov_cell_size=0.35,
        scan_coverage_radius=0.85,
        scan_k_cov_gain=0.012,
        scan_k_cov_revisit=0.01,
        scan_k_yawrate=0.0,
        scan_k_dvxy=0.01,
        scan_z_min=1.0,
        scan_z_max=2.2,
        scan_z_cmd_min=-0.2,
        scan_z_cmd_max=0.2,
        scan_z_target=1.8,
        scan_k_z_p=0.6,
        scan_oob_grace_steps=3,
        scan_cov_late_thresh=0.85,
        scan_k_cov_gain_late=0.01,
        scan_k_cov_stall=0.0,
        scan_debug_oob=False,
        seq_max_steps=1000,
    ),
    "A2_S4": PresetConfig(
        name="A2_S4",
        v_xy_max=0.8,
        v_z_max=0.7,
        seq_n_waypoints=15,
        seq_hop_min=2.0,
        seq_hop_max=6.0,
        seq_hold_steps=20,
        seq_success_radius=0.6,
        seq_k_prog=2.0,
        seq_max_steps=1200,
    ),
}

# Scan ablations for reward/safety tweaks (based on A2).
PRESETS["A2_ABL_A"] = replace(
    PRESETS["A2"],
    name="A2_ABL_A",
    scan_oob_grace_steps=3,
    scan_k_cov_gain_late=0.0,
    scan_k_cov_stall=0.0,
)
PRESETS["A2_ABL_B"] = replace(
    PRESETS["A2"],
    name="A2_ABL_B",
    scan_oob_grace_steps=3,
    scan_k_cov_gain=0.01,
    scan_cov_late_thresh=0.90,
    scan_k_cov_gain_late=0.01,
    scan_k_cov_stall=0.0,
)
PRESETS["A2_ABL_C"] = replace(
    PRESETS["A2"],
    name="A2_ABL_C",
    scan_oob_grace_steps=0,
    scan_cov_late_thresh=0.90,
    scan_k_cov_gain_late=0.0,
    scan_k_cov_stall=0.005,
)
PRESETS["A2_ABL_B125"] = replace(
    PRESETS["A2_ABL_B"],
    name="A2_ABL_B125",
    scan_k_cov_gain=0.0125,
)
PRESETS["A2_ABL_B15"] = replace(
    PRESETS["A2_ABL_B"],
    name="A2_ABL_B15",
    scan_k_cov_gain=0.015,
)
PRESETS["A2_ABL_B125_T85"] = replace(
    PRESETS["A2_ABL_B125"],
    name="A2_ABL_B125_T85",
    scan_cov_late_thresh=0.85,
)


def list_presets():
    return sorted(PRESETS.keys())


def get_preset(name: str) -> PresetConfig:
    key = (name or "A0").strip()
    if key not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Available: {', '.join(list_presets())}")
    return deepcopy(PRESETS[key])
