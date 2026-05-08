const BACKEND_BASE = import.meta.env.VITE_BACKEND_BASE || "http://127.0.0.1:8000";

export class ApiError extends Error {
  constructor(message, detail, status, payload = null) {
    super(message);
    this.name = "ApiError";
    this.detail = detail;
    this.status = status;
    this.payload = payload;
  }
}

function buildErrorMessage(detail, status) {
  if (!detail) return `HTTP ${status}`;
  if (typeof detail === "string") return detail;
  if (typeof detail === "object") {
    const reasons = Array.isArray(detail.blocking_reasons) ? detail.blocking_reasons : [];
    const base = detail.message || detail.error || `HTTP ${status}`;
    return reasons.length ? `${base} (${reasons.join("; ")})` : String(base);
  }
  return String(detail);
}

async function fetchReal(path, init = {}) {
  const resp = await fetch(`${BACKEND_BASE}${path}`, init);
  const payload = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const detail = payload?.detail ?? payload;
    throw new ApiError(buildErrorMessage(detail, resp.status), detail, resp.status, payload);
  }
  return payload;
}

function postReal(path, body) {
  const init = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  };
  if (body !== undefined) {
    init.body = JSON.stringify(body);
  }
  return fetchReal(path, init);
}

function deleteReal(path) {
  return fetchReal(path, { method: "DELETE" });
}

export function getRealReadiness() {
  return fetchReal("/api/real/readiness");
}

export function getRealTelemetry() {
  return fetchReal("/api/real/telemetry");
}

export function getRealMissionState() {
  return fetchReal("/api/real/mission/state");
}

export function getRealRadioStatus() {
  return fetchReal("/api/real/connection/status");
}

export function getRealRadioPorts() {
  return fetchReal("/api/real/connection/ports");
}

export function connectRealRadio(serialPort, serialBaud) {
  return postReal("/api/real/connection/connect", {
    serial_port: String(serialPort || "").trim(),
    serial_baud: Number(serialBaud) || 57600,
  });
}

export function disconnectRealRadio() {
  return postReal("/api/real/connection/disconnect");
}

export function testRealHeartbeat() {
  return postReal("/api/real/connection/heartbeat_test");
}

export function armRealDrone() {
  return postReal("/api/real/control/arm");
}

export function disarmRealDrone() {
  return postReal("/api/real/control/disarm");
}

export function takeoffRealDrone(altM) {
  return postReal("/api/real/control/takeoff", { alt_m: Number(altM) });
}

export function landRealDrone() {
  return postReal("/api/real/control/land");
}

export function landHereRealDrone() {
  return postReal("/api/real/control/land_here");
}

export function rtlRealDrone() {
  return postReal("/api/real/control/rtl");
}

export function holdRealDrone() {
  return postReal("/api/real/control/hold");
}

export function setRealMode(mode) {
  return postReal("/api/real/control/mode", { mode: String(mode || "").trim() });
}

export function generateRealTinyMission(payload = {}) {
  return postReal("/api/real/mission/generate_tiny", payload);
}

export function startRealMission({ alt_m, accept_radius_m }) {
  return postReal("/api/real/mission/start", {
    alt_m: Number(alt_m),
    accept_radius_m: Number(accept_radius_m),
  });
}

export function validateRealMissionStart({ alt_m, accept_radius_m }) {
  return postReal("/api/real/mission/validate_start", {
    alt_m: Number(alt_m),
    accept_radius_m: Number(accept_radius_m),
  });
}

export function stopRealMission() {
  return postReal("/api/real/mission/stop");
}

export function startCompassCalibration() {
  return postReal("/api/real/control/compass_calibrate/start");
}

export function cancelCompassCalibration() {
  return postReal("/api/real/control/compass_calibrate/cancel");
}

export function saveNorthReference(northHeadingDeg = 0.0) {
  return postReal("/api/real/control/compass_calibrate/north_reference", {
    north_heading_deg: Number(northHeadingDeg),
  });
}

export function levelCalibrate() {
  return postReal("/api/real/control/level_calibrate");
}

export function setRealMissionStartPosition(lng, lat) {
  return postReal("/api/real/mission/start_position", { lng: Number(lng), lat: Number(lat) });
}

export function clearRealMissionLandingPosition() {
  return deleteReal("/api/real/mission/landing_position");
}

export function getRealLastCommandDebug() {
  return fetchReal("/api/real/debug/last_command");
}

export function getRealCommandAckLog(limit = 40) {
  return fetchReal(`/api/real/debug/command_ack_log?limit=${Math.max(1, Number(limit) || 40)}`);
}

export function getRealBatteryDebug() {
  return fetchReal("/api/real/debug/battery");
}
