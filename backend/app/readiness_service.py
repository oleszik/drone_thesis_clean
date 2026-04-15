from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from .config import BackendConfig
from .fence_service import get_operating_fence, validate_mission_payload_inside_fence


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _meters_to_deg_lat(m: float) -> float:
    return float(m) / 111320.0


def _meters_to_deg_lng(m: float, lat_deg: float) -> float:
    c = math.cos(math.radians(float(lat_deg)))
    return float(m) / (111320.0 * max(0.1, abs(c)))


def _safe_float(v: Any) -> float | None:
    try:
        out = float(v)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


class ReadinessService:
    def __init__(self, cfg: BackendConfig) -> None:
        self._cfg = cfg

    def evaluate(
        self,
        status: dict[str, Any],
        telemetry: dict[str, Any],
        mission_path: dict[str, Any],
    ) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []

        connected = bool(status.get("connected"))
        checks.append(
            {
                "key": "mavlink_connected",
                "ok": connected,
                "severity": "critical",
                "message": "MAVLink link is connected" if connected else "MAVLink link is disconnected",
                "value": connected,
            }
        )

        hb_age = _safe_float(status.get("last_heartbeat_age_s"))
        hb_limit = max(1.0, float(self._cfg.mavlink_heartbeat_timeout_sec) * 2.0)
        heartbeat_ok = connected and hb_age is not None and hb_age <= hb_limit
        checks.append(
            {
                "key": "heartbeat_age_sec",
                "ok": heartbeat_ok,
                "severity": "critical",
                "message": (
                    f"heartbeat age is {hb_age:.2f}s"
                    if heartbeat_ok
                    else f"heartbeat is stale or unavailable (age={hb_age})"
                ),
                "value": hb_age,
            }
        )

        failsafes = status.get("failsafes") if isinstance(status.get("failsafes"), dict) else {}

        gps_ok = bool(failsafes.get("gps_ok"))
        checks.append(
            {
                "key": "gps_ok",
                "ok": gps_ok,
                "severity": "critical",
                "message": "GPS fix is valid" if gps_ok else "GPS fix is not valid",
                "value": {
                    "gps_fix": telemetry.get("gps_fix"),
                    "satellites": telemetry.get("satellites"),
                },
            }
        )

        ekf_ok = bool(failsafes.get("ekf_ok"))
        checks.append(
            {
                "key": "ekf_ok",
                "ok": ekf_ok,
                "severity": "critical",
                "message": "EKF is healthy" if ekf_ok else "EKF is not healthy",
                "value": telemetry.get("ekf_ok"),
            }
        )

        battery_pct = _safe_float(telemetry.get("battery_percent"))
        battery_low_flag = failsafes.get("battery_low") if "battery_low" in failsafes else None
        battery_low = bool(battery_low_flag) if battery_low_flag is not None else None
        # Prefer FC failsafe battery state when available; percent threshold is a secondary guard.
        if battery_low is not None:
            battery_ok = (not battery_low) and (battery_pct is None or battery_pct >= 25.0)
        else:
            battery_ok = battery_pct is not None and battery_pct >= 25.0
        checks.append(
            {
                "key": "battery_ok",
                "ok": battery_ok,
                "severity": "critical",
                "message": (
                    f"battery healthy (pct={battery_pct}, fc_low={battery_low})"
                    if battery_ok
                    else f"battery not ready (pct={battery_pct}, fc_low={battery_low})"
                ),
                "value": {
                    "battery_percent": battery_pct,
                    "battery_low": battery_low,
                },
            }
        )

        rc_link_ok = connected and heartbeat_ok
        checks.append(
            {
                "key": "rc_link_ok",
                "ok": rc_link_ok,
                "severity": "warning",
                "message": (
                    "RC link assumed present from healthy live link"
                    if rc_link_ok
                    else "RC link unavailable or cannot be inferred"
                ),
                "value": rc_link_ok,
            }
        )

        lat_ok = _safe_float(telemetry.get("lat")) is not None
        lon_ok = _safe_float(telemetry.get("lon")) is not None
        home_set = gps_ok and lat_ok and lon_ok
        checks.append(
            {
                "key": "home_position_set",
                "ok": home_set,
                "severity": "critical",
                "message": (
                    "home position inferred from live GPS position"
                    if home_set
                    else "home position unavailable"
                ),
                "value": {
                    "lat": telemetry.get("lat"),
                    "lon": telemetry.get("lon"),
                },
            }
        )

        current_mode = str(status.get("mode") or "UNKNOWN")
        checks.append(
            {
                "key": "current_mode",
                "ok": current_mode != "UNKNOWN",
                "severity": "info",
                "message": f"current mode is {current_mode}",
                "value": current_mode,
            }
        )

        armed = bool(status.get("armed"))
        checks.append(
            {
                "key": "armed",
                "ok": True,
                "severity": "info",
                "message": "vehicle is armed" if armed else "vehicle is disarmed",
                "value": armed,
            }
        )

        fence = get_operating_fence(self._cfg)
        fence_configured = bool(fence.get("configured"))
        checks.append(
            {
                "key": "fence_configured",
                "ok": fence_configured,
                "severity": "critical",
                "message": "operation fence is configured" if fence_configured else "operation fence is not configured",
                "value": {
                    "source": fence.get("source"),
                    "point_count": fence.get("point_count"),
                    "polygon_lng_lat": fence.get("polygon_lng_lat"),
                },
            }
        )

        mission_within_fence, mission_msg, mission_value = validate_mission_payload_inside_fence(mission_path, fence)
        checks.append(
            {
                "key": "mission_within_fence",
                "ok": mission_within_fence,
                "severity": "critical",
                "message": mission_msg,
                "value": mission_value,
            }
        )

        provider = str(self._cfg.map_provider or "").strip().lower()
        frontend_ok = bool(str(self._cfg.frontend_origin or "").strip())
        map_key_ok = True if provider != "tencent" else bool(str(self._cfg.tencent_key or "").strip())
        backend_config_ok = frontend_ok and map_key_ok
        checks.append(
            {
                "key": "backend_config_ok",
                "ok": backend_config_ok,
                "severity": "critical",
                "message": "backend config is valid" if backend_config_ok else "backend config is incomplete",
                "value": {
                    "frontend_origin_set": frontend_ok,
                    "map_provider": provider,
                    "map_key_ok": map_key_ok,
                },
            }
        )

        index = {item["key"]: item for item in checks}
        manual_required = ["mavlink_connected", "heartbeat_age_sec", "backend_config_ok"]
        autonomous_required = [
            "mavlink_connected",
            "heartbeat_age_sec",
            "gps_ok",
            "ekf_ok",
            "battery_ok",
            "home_position_set",
            "fence_configured",
            "mission_within_fence",
            "backend_config_ok",
        ]
        can_manual = all(bool(index[k]["ok"]) for k in manual_required)
        can_autonomous = all(bool(index[k]["ok"]) for k in autonomous_required)
        overall_ready = bool(can_autonomous)

        blocking_reasons = [
            str(item["message"])
            for item in checks
            if item.get("severity") == "critical" and not bool(item.get("ok"))
        ]

        return {
            "overall_ready": overall_ready,
            "can_manual": can_manual,
            "can_autonomous": can_autonomous,
            "checks": checks,
            "blocking_reasons": blocking_reasons,
            "timestamp": _utc_now_iso(),
        }
