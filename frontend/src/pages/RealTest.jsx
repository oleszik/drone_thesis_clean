import { useCallback, useEffect, useMemo, useState } from "react";
import { useLiveStream } from "../hooks/useLiveStream";
import { MapPanel } from "../MapPanel";

const BACKEND_BASE = import.meta.env.VITE_BACKEND_BASE || "http://127.0.0.1:8000";

class ApiError extends Error {
  constructor(message, detail, status) {
    super(message);
    this.name = "ApiError";
    this.detail = detail;
    this.status = status;
  }
}

async function fetchJson(path, init) {
  const resp = await fetch(`${BACKEND_BASE}${path}`, init);
  const payload = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const rawDetail = payload?.detail;
    let detail = rawDetail ? String(rawDetail) : `HTTP ${resp.status}`;
    if (rawDetail && typeof rawDetail === "object") {
      const reasons = Array.isArray(rawDetail.blocking_reasons) ? rawDetail.blocking_reasons : [];
      const msg = rawDetail.error ? String(rawDetail.error) : `HTTP ${resp.status}`;
      detail = reasons.length ? `${msg} (${reasons.join("; ")})` : msg;
    }
    throw new ApiError(detail, rawDetail, resp.status);
  }
  return payload;
}

function getErrorMessage(err) {
  if (err instanceof Error && err.message) return err.message;
  return String(err);
}

function getErrorLastStatusText(err) {
  const detail = err && typeof err === "object" ? err.detail : null;
  if (!detail || typeof detail !== "object") return "";
  return String(detail.last_status_text || "");
}

function toneFromReady(readiness) {
  if (readiness?.overall_ready) return "good";
  if (readiness?.can_manual) return "warn";
  return "bad";
}

function bannerLabel(readiness) {
  if (readiness?.overall_ready) return "Ready for tiny mission";
  if (readiness?.can_manual) return "Manual only";
  return "Autonomy blocked";
}

function toneFromMissionState(state) {
  const value = String(state || "IDLE").toUpperCase();
  if (value === "COMPLETE") return "good";
  if (value === "ERROR") return "bad";
  if (value === "STOPPED") return "warn";
  if (value === "RUN_PATH" || value === "ARMING" || value === "TAKEOFF") return "warn";
  return "neutral";
}

function fmt(v, suffix = "") {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "--";
  return `${Number(v).toFixed(1)}${suffix}`;
}

export function RealTest() {
  const [readiness, setReadiness] = useState({
    overall_ready: false,
    can_manual: false,
    can_autonomous: false,
    checks: [],
    blocking_reasons: [],
    timestamp: null,
  });
  const [loadError, setLoadError] = useState("");
  const [status, setStatus] = useState({
    mode: "UNKNOWN",
    armed: false,
    connected: false,
    failsafes: { gps_ok: false, ekf_ok: false },
    last_status_text: "",
    recent_status_text: [],
  });
  const [telemetry, setTelemetry] = useState({ battery_percent: null, updated_at_unix: null });
  const [missionState, setMissionState] = useState({
    state: "IDLE",
    scan_active: false,
    waypoint_index: 0,
    waypoint_count: 0,
    last_error: "",
    last_status_text: "",
  });
  const [lastFcMessage, setLastFcMessage] = useState("");
  const [actionMsg, setActionMsg] = useState("");
  const [actionTone, setActionTone] = useState("neutral");
  const [actionBusy, setActionBusy] = useState(false);
  const [radioPorts, setRadioPorts] = useState([]);
  const [portsError, setPortsError] = useState("");
  const [radioStatus, setRadioStatus] = useState({
    serial_port: "",
    serial_baud: 57600,
    connected: false,
    last_heartbeat_age_s: null,
    last_telemetry_age_s: null,
    stale: false,
    lost: true,
    state: "disconnected",
    error_message: "",
    last_status_text: "",
    recent_status_text: [],
  });
  const [serialPort, setSerialPort] = useState("");
  const [serialBaud, setSerialBaud] = useState(57600);
  const [showHealthyChecks, setShowHealthyChecks] = useState(false);
  const [showAllStatus, setShowAllStatus] = useState(false);
  const [planningState, setPlanningState] = useState({
    geometryValid: false,
    hasPath: false,
    missionStartConfigured: false,
    missionStartLngLat: null,
    startAltitudeM: 10,
    areaM2: 0,
    perimeterM: 0,
    routeLengthM: 0,
    fenceConfigured: false,
    missionInsideFence: false,
    validForMissionAction: false,
  });

  const onTelemetryStream = useCallback((payload) => {
    if (payload?.status) {
      setStatus((prev) => ({ ...prev, ...payload.status }));
      const text = String(payload.status.last_status_text || "");
      if (text) setLastFcMessage(text);
    }
    if (payload?.telemetry) {
      setTelemetry((prev) => ({ ...prev, ...payload.telemetry }));
    }
  }, []);

  const telemetryStream = useLiveStream("/api/real/stream/telemetry", {
    event: "telemetry",
    onMessage: onTelemetryStream,
  });

  useEffect(() => {
    let cancelled = false;
    let timer = null;

    async function poll() {
      try {
        const payload = await fetchJson("/api/real/readiness");
        if (!cancelled) {
          setReadiness(payload);
          if (payload?.last_status_text) {
            setLastFcMessage(String(payload.last_status_text));
          }
          setLoadError("");
        }
      } catch (err) {
        if (!cancelled) {
          setLoadError(getErrorMessage(err));
        }
      }
      if (!cancelled) timer = window.setTimeout(poll, 1500);
    }

    poll();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    let timer = null;

    async function pollMissionState() {
      try {
        const payload = await fetchJson("/api/real/mission/state");
        if (!cancelled) {
          setMissionState((prev) => ({ ...prev, ...(payload || {}) }));
          if (payload?.last_status_text) {
            setLastFcMessage(String(payload.last_status_text));
          }
        }
      } catch {
        // Keep last known state if polling fails temporarily.
      }
      if (!cancelled) timer = window.setTimeout(pollMissionState, 1000);
    }

    pollMissionState();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    let timer = null;

    async function pollRadio() {
      try {
        const [portsResp, statusResp] = await Promise.all([
          fetchJson("/api/real/connection/ports"),
          fetchJson("/api/real/connection/status"),
        ]);
        if (cancelled) return;
        const ports = Array.isArray(portsResp?.ports) ? portsResp.ports : [];
        setPortsError(String(portsResp?.error_message || ""));
        setRadioPorts(ports);
        setRadioStatus((prev) => ({ ...prev, ...(statusResp || {}) }));
        if (statusResp?.last_status_text) {
          setLastFcMessage(String(statusResp.last_status_text));
        }
        if (!serialPort) {
          const current = statusResp?.serial_port;
          const firstPort = ports[0]?.port;
          if (current) setSerialPort(String(current));
          else if (firstPort) setSerialPort(String(firstPort));
        }
      } catch {
        if (!cancelled) {
          setPortsError("failed to query serial ports");
          setRadioStatus((prev) => ({ ...prev, connected: false, state: "disconnected" }));
        }
      }
      if (!cancelled) timer = window.setTimeout(pollRadio, 2000);
    }

    pollRadio();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [serialPort]);

  const missionTone = toneFromMissionState(missionState.state);

  const recentFcMessages = useMemo(() => {
    const fromStatus = Array.isArray(status?.recent_status_text) ? status.recent_status_text : [];
    const fromRadio = Array.isArray(radioStatus?.recent_status_text) ? radioStatus.recent_status_text : [];
    const merged = [...fromStatus, ...fromRadio].map((item) => String(item || "").trim()).filter(Boolean);
    const deduped = [];
    for (const line of merged) {
      if (!deduped.includes(line)) deduped.push(line);
    }
    return deduped.slice(-5).reverse();
  }, [status?.recent_status_text, radioStatus?.recent_status_text]);

  const checksByKey = useMemo(() => {
    const out = {};
    for (const c of readiness.checks || []) {
      if (c?.key) out[c.key] = c;
    }
    return out;
  }, [readiness.checks]);

  const bannerTone = toneFromReady(readiness);

  const batteryCheckValue = useMemo(() => {
    const raw = checksByKey.battery_ok?.value;
    return raw && typeof raw === "object" ? raw : {};
  }, [checksByKey]);

  const batteryPercent = useMemo(() => {
    const fromCheck = Number(batteryCheckValue?.battery_percent);
    if (Number.isFinite(fromCheck)) return fromCheck;
    const fromTelemetry = Number(telemetry.battery_percent);
    return Number.isFinite(fromTelemetry) ? fromTelemetry : null;
  }, [batteryCheckValue, telemetry.battery_percent]);

  const liveCards = useMemo(() => {
    const hb = checksByKey.heartbeat_age_sec?.value;
    return [
      { label: "Mode", value: status.mode || "UNKNOWN", tone: "neutral" },
      { label: "Armed", value: status.armed ? "YES" : "NO", tone: status.armed ? "warn" : "good" },
      { label: "GPS", value: checksByKey.gps_ok?.ok ? "OK" : "BAD", tone: checksByKey.gps_ok?.ok ? "good" : "bad" },
      { label: "EKF", value: checksByKey.ekf_ok?.ok ? "OK" : "BAD", tone: checksByKey.ekf_ok?.ok ? "good" : "bad" },
      {
        label: "Battery",
        value: fmt(batteryPercent, "%"),
        tone: checksByKey.battery_ok?.ok ? "good" : "bad",
      },
      { label: "RC Link", value: checksByKey.rc_link_ok?.ok ? "OK" : "UNKNOWN", tone: checksByKey.rc_link_ok?.ok ? "good" : "warn" },
      { label: "Home", value: checksByKey.home_position_set?.ok ? "SET" : "NOT SET", tone: checksByKey.home_position_set?.ok ? "good" : "bad" },
      { label: "Fence", value: checksByKey.fence_configured?.ok ? "CONFIGURED" : "MISSING", tone: checksByKey.fence_configured?.ok ? "good" : "bad" },
      {
        label: "Heartbeat",
        value: hb === null || hb === undefined ? "--" : `${Number(hb).toFixed(2)}s`,
        tone: checksByKey.heartbeat_age_sec?.ok ? "good" : "bad",
      },
    ];
  }, [batteryPercent, checksByKey, status.armed, status.mode]);

  const primaryLiveCards = useMemo(() => {
    const priority = new Set(["Mode", "Armed", "Battery", "GPS", "EKF", "Heartbeat"]);
    return liveCards.filter((c) => priority.has(c.label));
  }, [liveCards]);

  const groupedChecks = useMemo(() => {
    const checks = Array.isArray(readiness.checks) ? readiness.checks : [];
    const critical = checks.filter((c) => !c?.ok && (c?.severity === "critical" || c?.severity === "warning"));
    const warnings = checks.filter((c) => c?.ok && c?.severity === "warning");
    const healthy = checks.filter((c) => c?.ok && c?.severity !== "warning");
    return { critical, warnings, healthy };
  }, [readiness.checks]);

  const compassCal = useMemo(() => {
    const raw = status?.compass_calibration;
    return raw && typeof raw === "object" ? raw : {};
  }, [status?.compass_calibration]);

  const compassCalState = String(compassCal?.state || "idle").toLowerCase();
  const compassCalMessage = String(compassCal?.message || "idle");
  const compassCalProgress = Number(compassCal?.completion_pct);
  const compassCalProgressLabel = Number.isFinite(compassCalProgress)
    ? `${Math.max(0, Math.min(100, compassCalProgress)).toFixed(0)}%`
    : "--";
  const compassCalActive = ["starting", "running", "waiting_to_start", "cancel_requested"].includes(compassCalState);
  const compassCalTone = compassCalState === "succeeded"
    ? "good"
    : (compassCalState === "failed" || compassCalState === "cancel_failed" || compassCalState === "link_lost")
      ? "bad"
      : compassCalActive
        ? "warn"
        : "neutral";

  const markActionMessage = useCallback((tone, text) => {
    setActionTone(tone);
    setActionMsg(text);
  }, []);

  const captureAutopilotMessage = useCallback((payload, err) => {
    const fromPayload = String(payload?.last_status_text || "");
    const fromError = getErrorLastStatusText(err);
    const chosen = fromPayload || fromError;
    if (chosen) setLastFcMessage(chosen);
  }, []);

  const runControlAction = useCallback(
    async (path, label, confirmText = "") => {
      if (confirmText && !window.confirm(confirmText)) return;
      setActionBusy(true);
      markActionMessage("warn", `Sending ${label}...`);
      try {
        const payload = await fetchJson(path, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
        });
        captureAutopilotMessage(payload, null);
        const modeHint = payload?.resulting_mode ? ` (mode: ${payload.resulting_mode})` : "";
        const armedHint = typeof payload?.armed === "boolean" ? ` (armed: ${payload.armed ? "YES" : "NO"})` : "";
        markActionMessage("good", `OK: ${label}${modeHint || armedHint}`);
      } catch (err) {
        captureAutopilotMessage(null, err);
        markActionMessage("bad", `Error: ${getErrorMessage(err)}`);
      } finally {
        setActionBusy(false);
      }
    },
    [captureAutopilotMessage, markActionMessage],
  );

  const runTinyMissionPreset = useCallback(async () => {
    if (!window.confirm("Generate Tiny Mission preset now? (takeoff/hold/forward/hold/RTL)")) return;
    setActionBusy(true);
    markActionMessage("warn", "Generating Tiny Mission...");
    try {
      const missionStart = Array.isArray(planningState.missionStartLngLat) && planningState.missionStartLngLat.length >= 2
        ? [Number(planningState.missionStartLngLat[0]), Number(planningState.missionStartLngLat[1])]
        : null;
      const telemetryStart = Number.isFinite(Number(telemetry.lon)) && Number.isFinite(Number(telemetry.lat))
        ? [Number(telemetry.lon), Number(telemetry.lat)]
        : null;
      const request = {};
      const selectedStart = missionStart || telemetryStart;
      const startSource = missionStart ? "map start point" : (telemetryStart ? "live GPS" : "autopilot reference");
      if (selectedStart) {
        request.start_lng = selectedStart[0];
        request.start_lat = selectedStart[1];
      }
      const headingDeg = Number(telemetry.yaw_deg);
      if (Number.isFinite(headingDeg)) {
        request.heading_deg = headingDeg;
      }

      const payload = await fetchJson("/api/real/mission/generate_tiny", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
      });
      captureAutopilotMessage(payload, null);
      const waypointCount = (payload?.waypoints_lng_lat || []).length;
      markActionMessage("good", `OK: Tiny Mission ready (${waypointCount} waypoints, start: ${startSource})`);
    } catch (err) {
      captureAutopilotMessage(null, err);
      markActionMessage("bad", `Error: ${getErrorMessage(err)}`);
    } finally {
      setActionBusy(false);
    }
  }, [captureAutopilotMessage, markActionMessage, planningState.missionStartLngLat, telemetry.lat, telemetry.lon, telemetry.yaw_deg]);

  const runSetStartFromLive = useCallback(async () => {
    const lng = Number(telemetry.lon);
    const lat = Number(telemetry.lat);
    if (!Number.isFinite(lng) || !Number.isFinite(lat)) {
      markActionMessage("bad", "Error: live GPS position unavailable, wait for telemetry");
      return;
    }
    setActionBusy(true);
    markActionMessage("warn", "Saving mission start point from live GPS...");
    try {
      const payload = await fetchJson("/api/mission/start_position", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lng, lat }),
      });
      captureAutopilotMessage(payload, null);
      setPlanningState((prev) => ({
        ...prev,
        missionStartConfigured: true,
        missionStartLngLat: [lng, lat],
      }));
      markActionMessage("good", `OK: mission start point saved (${lng.toFixed(6)}, ${lat.toFixed(6)})`);
    } catch (err) {
      captureAutopilotMessage(null, err);
      markActionMessage("bad", `Error: ${getErrorMessage(err)}`);
    } finally {
      setActionBusy(false);
    }
  }, [captureAutopilotMessage, markActionMessage, telemetry.lat, telemetry.lon]);

  const runStartMission = useCallback(async () => {
    if (!window.confirm("Start the generated mission on the real drone now?")) return;
    setActionBusy(true);
    markActionMessage("warn", "Starting real mission...");
    try {
      const payload = await fetchJson("/api/real/mission/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          alt_m: Number(planningState.startAltitudeM || 10),
          accept_radius_m: 3.0,
        }),
      });
      captureAutopilotMessage(payload, null);
      setMissionState((prev) => ({ ...prev, ...(payload || {}) }));

      const latestState = await fetchJson("/api/real/mission/state");
      setMissionState((prev) => ({ ...prev, ...(latestState || {}) }));
      captureAutopilotMessage(latestState, null);

      const stateName = String(latestState?.state || payload?.state || "UNKNOWN").toUpperCase();
      if (stateName === "ERROR") {
        const reason = String(latestState?.last_error || "mission failed during startup");
        markActionMessage("bad", `Mission failed: ${reason}`);
      } else {
        const count = Number(latestState?.waypoint_count ?? payload?.waypoint_count ?? 0);
        markActionMessage("good", `Mission start accepted (state=${stateName}${count ? `, waypoints=${count}` : ""})`);
      }
    } catch (err) {
      captureAutopilotMessage(null, err);
      markActionMessage("bad", `Error: ${getErrorMessage(err)}`);
    } finally {
      setActionBusy(false);
    }
  }, [captureAutopilotMessage, markActionMessage, planningState.startAltitudeM]);

  const runStopMission = useCallback(async () => {
    if (!window.confirm("Stop the active real mission and command RTL?")) return;
    setActionBusy(true);
    markActionMessage("warn", "Stopping real mission...");
    try {
      const payload = await fetchJson("/api/real/mission/stop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      captureAutopilotMessage(payload, null);
      setMissionState((prev) => ({ ...prev, ...(payload || {}) }));
      markActionMessage("good", `Mission stop requested (state=${String(payload?.state || "STOPPED").toUpperCase()})`);
    } catch (err) {
      captureAutopilotMessage(null, err);
      markActionMessage("bad", `Error: ${getErrorMessage(err)}`);
    } finally {
      setActionBusy(false);
    }
  }, [captureAutopilotMessage, markActionMessage]);

  const connectRadio = useCallback(async () => {
    setActionBusy(true);
    markActionMessage("warn", "Connecting radio...");
    try {
      const payload = await fetchJson("/api/real/connection/connect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ serial_port: serialPort, serial_baud: Number(serialBaud) }),
      });
      captureAutopilotMessage(payload, null);
      markActionMessage("good", `OK: radio connect (${payload?.serial_port || serialPort})`);
      const statusPayload = await fetchJson("/api/real/connection/status");
      setRadioStatus((prev) => ({ ...prev, ...(statusPayload || {}) }));
      captureAutopilotMessage(statusPayload, null);
    } catch (err) {
      captureAutopilotMessage(null, err);
      markActionMessage("bad", `Error: ${getErrorMessage(err)}`);
    } finally {
      setActionBusy(false);
    }
  }, [captureAutopilotMessage, markActionMessage, serialBaud, serialPort]);

  const disconnectRadio = useCallback(async () => {
    setActionBusy(true);
    markActionMessage("warn", "Disconnecting radio...");
    try {
      const payload = await fetchJson("/api/real/connection/disconnect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      captureAutopilotMessage(payload, null);
      markActionMessage("good", "OK: radio disconnected");
      const statusPayload = await fetchJson("/api/real/connection/status");
      setRadioStatus((prev) => ({ ...prev, ...(statusPayload || {}) }));
      captureAutopilotMessage(statusPayload, null);
    } catch (err) {
      captureAutopilotMessage(null, err);
      markActionMessage("bad", `Error: ${getErrorMessage(err)}`);
    } finally {
      setActionBusy(false);
    }
  }, [captureAutopilotMessage, markActionMessage]);

  const testHeartbeat = useCallback(async () => {
    setActionBusy(true);
    markActionMessage("warn", "Heartbeat test...");
    try {
      const payload = await fetchJson("/api/real/connection/heartbeat_test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      captureAutopilotMessage(payload, null);
      markActionMessage("good", `OK: heartbeat ${fmt(payload?.heartbeat_age_s, "s")}`);
      const statusPayload = await fetchJson("/api/real/connection/status");
      setRadioStatus((prev) => ({ ...prev, ...(statusPayload || {}) }));
      captureAutopilotMessage(statusPayload, null);
    } catch (err) {
      captureAutopilotMessage(null, err);
      markActionMessage("bad", `Error: ${getErrorMessage(err)}`);
    } finally {
      setActionBusy(false);
    }
  }, [captureAutopilotMessage, markActionMessage]);

  return (
    <main className="app app-shell real-test-app real-console">
      <header className="top console-header">
        <div>
          <span className="mode-tag">Flight Ops</span>
          <h1>Live Mission Flight Console</h1>
          <p className="hint">Preflight gate, field link, and intervention controls.</p>
        </div>
        <div className="chips">
          <a className="chip nav-chip" href="/sim">Simulation</a>
          <span className="chip">Telemetry stream: {telemetryStream.connected ? "connected" : "reconnecting"}</span>
        </div>
      </header>

      <section className={`real-banner tone-${bannerTone}`}>
        <strong>{bannerLabel(readiness)}</strong>
        <span>manual={String(Boolean(readiness.can_manual))} | autonomous={String(Boolean(readiness.can_autonomous))}</span>
      </section>

      {loadError ? <div className="real-error">Readiness load error: {loadError}</div> : null}

      <div className="real-widget-grid">
        <section className="panel real-panel real-widget">
          <div className="panel-header">
            <div>
              <h2>Field Link - Radio Connection (3DR X6-433)</h2>
              <p className="hint">Connect, then confirm heartbeat freshness.</p>
            </div>
          </div>
          <div className="real-radio-grid">
            <label className="real-field">
              <span>Serial Port</span>
              <input
                list="real-radio-port-list"
                value={serialPort}
                onChange={(e) => setSerialPort(e.target.value)}
                placeholder="e.g. COM5"
              />
              <datalist id="real-radio-port-list">
                {radioPorts.map((p) => (
                  <option key={p.port} value={p.port}>{p.description ? `${p.port} - ${p.description}` : p.port}</option>
                ))}
              </datalist>
            </label>
            <label className="real-field">
              <span>Baud</span>
              <select value={String(serialBaud)} onChange={(e) => setSerialBaud(Number(e.target.value) || 57600)}>
                <option value="57600">57600</option>
                <option value="115200">115200</option>
                <option value="38400">38400</option>
              </select>
            </label>
          </div>
          {portsError ? <p className="hint bad">Port scan: {portsError} (you can still type COM port manually)</p> : null}
          <div className="real-radio-actions real-action-row cols-3">
            <button className="real-connect-btn" disabled={actionBusy || !serialPort} onClick={connectRadio}>Connect</button>
            <button className="real-disconnect-btn" disabled={actionBusy || !radioStatus.connected} onClick={disconnectRadio}>Disconnect</button>
            <button className="real-test-btn" disabled={actionBusy || !radioStatus.connected} onClick={testHeartbeat}>Heartbeat Test</button>
          </div>
          <div className="status-pill-grid real-pill-grid compact-pill-grid">
            <div className={`status-pill tone-${radioStatus.connected ? "good" : "bad"}`}><span>Connected</span><strong>{radioStatus.connected ? "YES" : "NO"}</strong></div>
            <div className={`status-pill tone-${radioStatus.stale ? "warn" : "good"}`}><span>State</span><strong>{String(radioStatus.state || "unknown").toUpperCase()}</strong></div>
            <div className={`status-pill tone-${radioStatus.lost ? "bad" : "good"}`}><span>Heartbeat Age</span><strong>{fmt(radioStatus.last_heartbeat_age_s, "s")}</strong></div>
            <div className={`status-pill tone-${radioStatus.lost ? "bad" : "good"}`}><span>Telemetry Age</span><strong>{fmt(radioStatus.last_telemetry_age_s, "s")}</strong></div>
            {radioStatus.connected ? <div className="status-pill tone-neutral"><span>Port</span><strong>{radioStatus.serial_port || "--"}</strong></div> : null}
            {radioStatus.connected ? <div className="status-pill tone-neutral"><span>Baud</span><strong>{radioStatus.serial_baud || "--"}</strong></div> : null}
          </div>
          {radioStatus.error_message ? <p className="hint bad">Radio error: {String(radioStatus.error_message)}</p> : null}
        </section>

        <section className="panel real-panel real-widget">
          <div className="panel-header">
            <div>
              <h2>Safety Intervention Controls</h2>
              <p className="hint">Primary emergency actions.</p>
            </div>
            {actionMsg ? <span className={`chip ${actionTone === "bad" ? "bad" : actionTone === "good" ? "ok" : ""}`}>{actionMsg}</span> : null}
          </div>
          <div className="real-control-row real-action-row cols-4">
            <button
              className="real-arm-btn"
              disabled={!status.connected || actionBusy || status.armed}
              onClick={() => runControlAction("/api/real/control/arm", "ARM", "Arm the vehicle now?")}
            >
              Arm
            </button>
            <button
              className="real-disconnect-btn"
              disabled={!status.connected || actionBusy || !status.armed}
              onClick={() => runControlAction("/api/real/control/disarm", "DISARM", "Disarm the vehicle now?")}
            >
              Disarm
            </button>
            <button
              className="real-hold-btn"
              disabled={!status.connected || actionBusy}
              onClick={() => runControlAction("/api/real/control/hold", "HOLD", "Set vehicle to HOLD (LOITER) now?")}
            >
              Hold
            </button>
            <button
              className="real-rtl-btn"
              disabled={!status.connected || actionBusy}
              onClick={() => runControlAction("/api/real/control/rtl", "RTL", "Return to launch now?")}
            >
              RTL
            </button>
            <button
              className="real-land-btn"
              disabled={!status.connected || actionBusy}
              onClick={() => runControlAction("/api/real/control/land", "LAND", "Send LAND now?")}
            >
              Land
            </button>
            <button
              className="real-landhere-btn"
              disabled={!status.connected || actionBusy}
              onClick={() => runControlAction("/api/real/control/land_here", "LAND HERE", "Command immediate LAND at current position?")}
            >
              Land Here
            </button>
            <button
              className="real-test-btn"
              disabled={!status.connected || actionBusy || status.armed || compassCalActive}
              onClick={() => runControlAction("/api/real/control/compass_calibrate/start", "COMPASS CAL START", "Start compass calibration now? Keep vehicle disarmed and rotate slowly across all axes.")}
            >
              Compass Cal Start
            </button>
            <button
              className="real-test-btn"
              disabled={!status.connected || actionBusy || status.armed || compassCalActive}
              onClick={() => runControlAction("/api/real/control/level_calibrate", "LEVEL CAL", "Run level calibration now? Place the vehicle on a stable level surface and keep it disarmed.")}
            >
              Level Cal
            </button>
            <button
              className="real-disconnect-btn"
              disabled={!status.connected || actionBusy || !compassCalActive}
              onClick={() => runControlAction("/api/real/control/compass_calibrate/cancel", "COMPASS CAL CANCEL", "Cancel compass calibration now?")}
            >
              Compass Cal Cancel
            </button>
          </div>
          <div className="status-pill-grid real-pill-grid compact-pill-grid">
            <div className={`status-pill tone-${compassCalTone}`}>
              <span>Compass Cal</span>
              <strong>{compassCalState.toUpperCase()}</strong>
            </div>
            <div className={`status-pill tone-${compassCalActive ? "warn" : "neutral"}`}>
              <span>Progress</span>
              <strong>{compassCalProgressLabel}</strong>
            </div>
            <div className="status-pill tone-neutral">
              <span>Compass ID</span>
              <strong>{compassCal?.compass_id ?? "--"}</strong>
            </div>
            <div className="status-pill tone-neutral">
              <span>Cal Status</span>
              <strong>{String(compassCal?.cal_status_label || "--").toUpperCase()}</strong>
            </div>
          </div>
          <p className={`hint ${compassCalTone === "bad" ? "bad" : ""}`}>Compass calibration: {compassCalMessage}</p>
          <p className="hint">Field workflow: keep the vehicle disarmed, move away from metal/rebar, rotate through all axes slowly until status reaches SUCCESS, then arm.</p>
          <div className="status-pill-grid real-pill-grid compact-pill-grid">
            <div className={`status-pill tone-${lastFcMessage ? "warn" : "neutral"}`}>
              <span>Last FC Message</span>
              <strong>{lastFcMessage || "--"}</strong>
            </div>
          </div>
          {recentFcMessages.length ? (
            <details className="inline-collapsible">
              <summary>Recent autopilot messages</summary>
              <div className="collapsible-body compact-lines">
                {recentFcMessages.map((line, idx) => (
                  <p key={`${idx}-${line}`}>{line}</p>
                ))}
              </div>
            </details>
          ) : null}
        </section>

        <section className="panel real-panel real-widget real-widget-map real-widget-wide">
          <div className="panel-header">
            <div>
              <h2>Mission Planning Map</h2>
              <p className="hint">Draw mission area, generate path from live GPS reference, and verify validity.</p>
            </div>
          </div>
          <div className="chips compact-strip">
            <span className="chip"><strong>Path:</strong> {planningState.hasPath ? "ready" : "not generated"}</span>
            {!planningState.geometryValid ? <span className="chip"><strong>Geometry:</strong> invalid</span> : null}
            <span className="chip"><strong>Start:</strong> {planningState.missionStartConfigured ? "set" : "not set"}</span>
            <span className="chip">
              <strong>Fence:</strong> {planningState.fenceConfigured ? (planningState.hasPath ? (planningState.missionInsideFence ? "mission inside" : "check pending") : "configured") : "missing"}
            </span>
            {planningState.areaM2 > 0 ? <span className="chip"><strong>Area:</strong> {fmt(planningState.areaM2, " m^2")}</span> : null}
            {planningState.routeLengthM > 0 ? <span className="chip"><strong>Route:</strong> {fmt(planningState.routeLengthM, " m")}</span> : null}
          </div>
          <MapPanel
            telemetry={telemetry}
            mavConnected={Boolean(status.connected)}
            variant="real"
            onPlanningStateChange={setPlanningState}
          />
        </section>

        <section className="panel real-panel real-widget">
          <div className="panel-header">
            <div>
              <h2>Approved Autonomy Actions</h2>
              <p className="hint">Start a generated path or prepare the tiny preset.</p>
            </div>
          </div>
          <div className="real-preset-row">
            <button
              className="real-connect-btn"
              disabled={actionBusy || !readiness.can_autonomous || !planningState.validForMissionAction || !status.connected}
              onClick={runStartMission}
            >
              Start Mission
            </button>
            <button
              className="real-rtl-btn"
              disabled={actionBusy || !status.connected}
              onClick={runStopMission}
            >
              Stop Mission / RTL
            </button>
            <button
              className="real-test-btn"
              disabled={actionBusy || !status.connected}
              onClick={runSetStartFromLive}
            >
              Set Start = Live GPS
            </button>
            <button
              className="real-tiny-btn"
              disabled={actionBusy || !status.connected}
              onClick={runTinyMissionPreset}
            >
              Generate Tiny Mission
            </button>
          </div>
          <div className="status-pill-grid real-pill-grid compact-pill-grid">
            <div className={`status-pill tone-${missionTone}`}>
              <span>Executor</span>
              <strong>{String(missionState.state || "IDLE").toUpperCase()}</strong>
            </div>
            <div className="status-pill tone-neutral">
              <span>Scan Active</span>
              <strong>{missionState.scan_active ? "YES" : "NO"}</strong>
            </div>
            <div className="status-pill tone-neutral">
              <span>Waypoint</span>
              <strong>{Number(missionState.waypoint_index || 0)} / {Number(missionState.waypoint_count || 0)}</strong>
            </div>
          </div>
          {missionState.last_error ? <p className="hint bad">Mission executor error: {String(missionState.last_error)}</p> : null}
          {!readiness.can_autonomous || !planningState.validForMissionAction || !status.connected ? (
            <p className="hint">Start Mission requires a live link, green readiness, and a generated path inside the fence.</p>
          ) : null}
        </section>

        <section className="panel real-panel real-widget">
          <div className="panel-header">
            <div>
              <h2>Live Vehicle Status</h2>
              <p className="hint">At-a-glance status.</p>
            </div>
          </div>
          <div className="status-pill-grid real-pill-grid compact-pill-grid">
            {primaryLiveCards.map((item) => (
              <div key={item.label} className={`status-pill tone-${item.tone}`}>
                <span>{item.label}</span>
                <strong>{item.value}</strong>
              </div>
            ))}
          </div>
          <details className="inline-collapsible" open={showAllStatus} onToggle={(e) => setShowAllStatus(e.currentTarget.open)}>
            <summary>More status</summary>
            <div className="collapsible-body status-pill-grid real-pill-grid compact-pill-grid">
              {liveCards.filter((item) => !primaryLiveCards.some((primary) => primary.label === item.label)).map((item) => (
                <div key={item.label} className={`status-pill tone-${item.tone}`}>
                  <span>{item.label}</span>
                  <strong>{item.value}</strong>
                </div>
              ))}
            </div>
          </details>
        </section>

        <section className="panel real-panel real-widget real-widget-wide">
          <div className="panel-header">
            <div>
              <h2>Preflight Readiness Checklist</h2>
              <p className="hint">Blockers first, healthy checks minimized.</p>
            </div>
          </div>
          <div className="real-check-group critical">
            <h3>Critical blockers</h3>
            {groupedChecks.critical.length ? (
              groupedChecks.critical.map((check) => (
                <article key={check.key} className="real-check bad">
                  <div className="real-check-main">
                    <strong>{check.key}</strong>
                    <span className="chip">{check.severity}</span>
                  </div>
                  <p>{check.message}</p>
                </article>
              ))
            ) : (
              <div className="empty-card compact">No critical blockers.</div>
            )}
          </div>

          <div className="real-check-group warn">
            <h3>Warnings</h3>
            {groupedChecks.warnings.length ? (
              groupedChecks.warnings.map((check) => (
                <article key={check.key} className="real-check warn">
                  <div className="real-check-main">
                    <strong>{check.key}</strong>
                    <span className="chip">warning</span>
                  </div>
                  <p>{check.message}</p>
                </article>
              ))
            ) : (
              <div className="empty-card compact">No warnings.</div>
            )}
          </div>

          <details className="inline-collapsible" open={showHealthyChecks} onToggle={(e) => setShowHealthyChecks(e.currentTarget.open)}>
            <summary>Healthy checks ({groupedChecks.healthy.length})</summary>
            <div className="collapsible-body real-checklist compact-list">
              {groupedChecks.healthy.map((check) => (
                <article key={check.key} className="real-check ok compact">
                  <div className="real-check-main">
                    <strong>{check.key}</strong>
                  </div>
                  <p>{check.message}</p>
                </article>
              ))}
            </div>
          </details>
        </section>
      </div>
    </main>
  );
}
