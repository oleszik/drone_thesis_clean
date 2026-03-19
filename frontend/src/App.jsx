import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { MapPanel } from "./MapPanel";
import { ControlSection } from "./components/ControlSection";
import { EventLog } from "./components/EventLog";
import { StatusBar } from "./components/StatusBar";
import { TelemetryCards } from "./components/TelemetryCards";
import { useLiveStream } from "./hooks/useLiveStream";

const BACKEND_BASE = import.meta.env.VITE_BACKEND_BASE || "http://127.0.0.1:8000";

function missionModeLabel(mode) {
  return mode === "orbit_scan" ? "Orbit" : "Area";
}

async function fetchJson(path, init) {
  const resp = await fetch(`${BACKEND_BASE}${path}`, init);
  const payload = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const detail = payload?.detail ? String(payload.detail) : `HTTP ${resp.status}`;
    throw new Error(detail);
  }
  return payload;
}

function fmt(value, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return `${Number(value).toFixed(1)}${suffix}`;
}

function fmtPct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return `${Number(value).toFixed(1)}%`;
}

export function App() {
  const [health, setHealth] = useState({ ok: false, message: "Connecting...", mapProvider: "-" });
  const [status, setStatus] = useState({
    connected: false,
    armed: false,
    mode: "UNKNOWN",
    failsafes: { gps_ok: false, ekf_ok: false, battery_low: false },
    connection_url: "",
    last_error: "",
  });
  const [telemetry, setTelemetry] = useState({
    rel_alt_m: null,
    speed_m_s: null,
    yaw_deg: null,
    battery_percent: null,
    gps_fix: null,
    ekf_ok: null,
    updated_at_unix: null,
  });
  const [actionMsg, setActionMsg] = useState("");
  const [modeInput, setModeInput] = useState("GUIDED");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [coverageStats, setCoverageStats] = useState(null);
  const [coverageVersion, setCoverageVersion] = useState(0);
  const [missionState, setMissionState] = useState({
    mode: "idle",
    hasMission: false,
    targetIndex: null,
    targetCount: 0,
    coverageActive: false,
    debugAvailable: false,
    vehicleVisible: false,
    simStatus: "STOPPED",
    executorState: "IDLE",
    missionMessage: "",
  });
  const [uiEvents, setUiEvents] = useState([]);
  const eventSeqRef = useRef(0);
  const prevStatusRef = useRef(null);
  const prevMissionRef = useRef(null);
  const prevStreamRef = useRef(null);

  const appendEvent = useCallback((title, detail = "", tone = "neutral") => {
    setUiEvents((prev) => {
      const last = prev[0];
      if (last && last.title === title && last.detail === detail) {
        return prev;
      }
      eventSeqRef.current += 1;
      const next = [{ id: `ui-${eventSeqRef.current}`, ts: Date.now(), title, detail, tone }, ...prev];
      return next.slice(0, 24);
    });
  }, []);

  const onTelemetryStream = useCallback((payload) => {
    if (payload?.status) {
      setStatus((prev) => ({ ...prev, ...payload.status }));
    }
    if (payload?.telemetry) {
      setTelemetry((prev) => ({ ...prev, ...payload.telemetry }));
    }
  }, []);

  const telemetryStream = useLiveStream("/api/stream/telemetry", {
    event: "telemetry",
    onMessage: onTelemetryStream,
  });

  useEffect(() => {
    let cancelled = false;
    let healthTimer = null;

    async function runHealth() {
      try {
        const payload = await fetchJson("/api/health");
        if (!cancelled) {
          setHealth({
            ok: Boolean(payload?.ok),
            message: payload?.ok ? "backend connected" : "backend not ready",
            mapProvider: String(payload?.map_provider || "-"),
          });
        }
      } catch (err) {
        if (!cancelled) {
          setHealth({
            ok: false,
            message: `backend unavailable (${String(err)})`,
            mapProvider: "-",
          });
        }
      }
      if (!cancelled) {
        healthTimer = window.setTimeout(runHealth, 2000);
      }
    }
    runHealth();
    return () => {
      cancelled = true;
      if (healthTimer) window.clearTimeout(healthTimer);
    };
  }, []);

  useEffect(() => {
    const prev = prevStatusRef.current;
    if (prev) {
      if (prev.connected !== status.connected) {
        appendEvent(status.connected ? "Vehicle connected" : "Vehicle disconnected", status.connection_url || "", status.connected ? "good" : "warn");
      }
      if (prev.mode !== status.mode && status.mode) {
        appendEvent("Mode changed", String(status.mode), "neutral");
      }
      if (prev.armed !== status.armed) {
        appendEvent(status.armed ? "Vehicle armed" : "Vehicle disarmed", "", status.armed ? "warn" : "neutral");
      }
      if (prev.last_error !== status.last_error && status.last_error) {
        appendEvent("Vehicle error", String(status.last_error), "bad");
      }
    }
    prevStatusRef.current = status;
  }, [appendEvent, status]);

  useEffect(() => {
    const prev = prevMissionRef.current;
    if (prev) {
      if (prev.executorState !== missionState.executorState) {
        appendEvent("Mission state", missionState.executorState, missionState.executorState === "RUN_PATH" ? "good" : "neutral");
      }
      if (prev.targetIndex !== missionState.targetIndex && missionState.targetIndex !== null) {
        appendEvent("Target updated", `${missionState.targetIndex}/${missionState.targetCount || 0}`, "neutral");
      }
      if (prev.missionMessage !== missionState.missionMessage && missionState.missionMessage) {
        appendEvent("Mission update", missionState.missionMessage, "neutral");
      }
    }
    prevMissionRef.current = missionState;
  }, [appendEvent, missionState]);

  useEffect(() => {
    const next = telemetryStream.connected ? "connected" : "reconnecting";
    const prev = prevStreamRef.current;
    if (prev && prev !== next) {
      appendEvent("Telemetry stream", next, next === "connected" ? "good" : "warn");
    }
    prevStreamRef.current = next;
  }, [appendEvent, telemetryStream.connected]);

  async function runAction(path, body = null, opts = {}) {
    if (opts.confirmText && !window.confirm(opts.confirmText)) return;
    setActionMsg("Sending...");
    try {
      const payload = await fetchJson(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: body ? JSON.stringify(body) : undefined,
      });
      const msg = `OK: ${payload.action || "done"}`;
      setActionMsg(msg);
      appendEvent("Action complete", msg, "good");
    } catch (err) {
      const msg = `Error: ${String(err)}`;
      setActionMsg(msg);
      appendEvent("Action failed", msg, "bad");
    }
  }

  async function resetCoverage() {
    if (!window.confirm("Reset mission coverage history?")) return;
    setActionMsg("Resetting coverage...");
    try {
      await fetchJson("/api/coverage/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      setCoverageVersion((v) => v + 1);
      setActionMsg("OK: coverage reset");
      appendEvent("Coverage reset", "", "warn");
    } catch (err) {
      const msg = `Error: ${String(err)}`;
      setActionMsg(msg);
      appendEvent("Coverage reset failed", msg, "bad");
    }
  }

  const onCoverageUpdate = useCallback((stats) => {
    setCoverageStats(stats || null);
  }, []);

  const handleMissionStateChange = useCallback((nextState) => {
    setMissionState(nextState || {});
  }, []);

  const telemetryAgeSec = useMemo(() => {
    const updated = Number(telemetry.updated_at_unix);
    if (!Number.isFinite(updated) || updated <= 0) return null;
    return Math.max(0, Date.now() / 1000 - updated);
  }, [telemetry.updated_at_unix]);

  const telemetryStale = telemetryAgeSec !== null && telemetryAgeSec > 3;
  const telemetryMissing = telemetryAgeSec === null;

  const missionRunLabel = useMemo(() => {
    if (missionState.executorState && missionState.executorState !== "IDLE") return `Mission: ${missionState.executorState}`;
    if (missionState.simStatus && missionState.simStatus !== "STOPPED") return `Sim: ${missionState.simStatus}`;
    return missionState.hasMission ? "Mission loaded" : "No mission";
  }, [missionState.executorState, missionState.hasMission, missionState.simStatus]);

  const statusItems = useMemo(
    () => [
      { label: "Connection", value: status.connected ? "Online" : "Offline", tone: status.connected ? "good" : "bad" },
      { label: "Mode", value: status.mode || "UNKNOWN", tone: "neutral" },
      { label: "Arm", value: status.armed ? "Armed" : "Safe", tone: status.armed ? "warn" : "good" },
      { label: "Mission", value: missionRunLabel, tone: missionState.hasMission ? "neutral" : "subdued" },
      {
        label: "Target",
        value: missionState.targetCount ? `${missionState.targetIndex || 0}/${missionState.targetCount}` : "--",
        tone: missionState.targetCount ? "neutral" : "subdued",
      },
      {
        label: "Coverage",
        value: missionState.mode === "orbit_scan" ? "N/A" : fmtPct(coverageStats?.coverage_pct),
        tone: missionState.mode === "orbit_scan" ? "subdued" : (coverageStats?.coverage_pct > 0 ? "good" : "subdued"),
      },
      {
        label: "Telemetry",
        value: telemetryMissing ? "No data" : telemetryStale ? `Stale ${telemetryAgeSec.toFixed(1)}s` : "Live",
        tone: telemetryMissing ? "subdued" : telemetryStale ? "warn" : "good",
      },
    ],
    [coverageStats?.coverage_pct, missionRunLabel, missionState.hasMission, missionState.mode, missionState.targetCount, missionState.targetIndex, status.armed, status.connected, status.mode, telemetryAgeSec, telemetryMissing, telemetryStale],
  );

  const statusDetail = useMemo(() => {
    const pieces = [
      health.ok ? health.message : "Backend unavailable",
      `Map: ${health.mapProvider || "-"}`,
      `Telemetry stream: ${telemetryStream.connected ? "connected" : "reconnecting"}`,
    ];
    if (actionMsg) pieces.push(actionMsg);
    if (status.last_error) pieces.push(`Vehicle error: ${status.last_error}`);
    return pieces.join(" • ");
  }, [actionMsg, health.mapProvider, health.message, health.ok, status.last_error, telemetryStream.connected]);

  const telemetryItems = useMemo(
    () => [
      { label: "Altitude", value: fmt(telemetry.rel_alt_m, " m"), tone: "default" },
      { label: "Speed", value: fmt(telemetry.speed_m_s, " m/s"), tone: "default" },
      { label: "Yaw", value: fmt(telemetry.yaw_deg, " deg"), tone: "default" },
      {
        label: "Battery",
        value: fmt(telemetry.battery_percent, " %"),
        tone: telemetry.battery_percent !== null && Number(telemetry.battery_percent) < 25 ? "warn" : "default",
      },
      { label: "GPS Fix", value: telemetry.gps_fix ?? "--", tone: telemetry.gps_fix >= 3 ? "good" : "warn" },
      { label: "EKF", value: telemetry.ekf_ok === null ? "--" : telemetry.ekf_ok ? "OK" : "BAD", tone: telemetry.ekf_ok ? "good" : "warn" },
    ],
    [telemetry],
  );

  const eventItems = useMemo(() => uiEvents, [uiEvents]);

  return (
    <main className="app">
      <header className="top">
        <div>
          <h1>Drone Thesis Dashboard</h1>
          <p className="hint">Operator view for connection, mission execution, coverage, and live map supervision.</p>
        </div>
        <button className="small-btn" onClick={() => setSidebarCollapsed((v) => !v)}>
          {sidebarCollapsed ? "Show Controls" : "Hide Controls"}
        </button>
      </header>

      <StatusBar items={statusItems} detail={statusDetail} />

      <section className={`layout ${sidebarCollapsed ? "collapsed" : ""}`}>
        <aside className="panel">
          <div className="panel-header">
            <div>
              <h2>Controls</h2>
              <p className="hint">Actions are grouped by operator workflow and disabled when the current state does not allow them.</p>
            </div>
          </div>

          <ControlSection title="Connection" hint="Use this first when attaching to MAVLink or resetting the link.">
            <button disabled={status.connected} onClick={() => runAction("/api/connection/connect")}>Connect</button>
            <button disabled={!status.connected} onClick={() => runAction("/api/connection/disconnect")}>Disconnect</button>
            <p className="hint">Link: {status.connection_url || "--"}</p>
          </ControlSection>

          <ControlSection title="Vehicle" hint="Direct vehicle actions remain separate from mission path execution.">
            <button disabled={!status.connected} onClick={() => runAction("/api/control/arm")}>Arm</button>
            <button disabled={!status.connected} onClick={() => runAction("/api/control/disarm")}>Disarm</button>
            <button disabled={!status.connected} onClick={() => runAction("/api/control/takeoff", { alt_m: 10 })}>Takeoff</button>
            <button
              disabled={!status.connected}
              onClick={() => runAction("/api/control/land", null, { confirmText: "Send LAND to the vehicle?" })}
            >
              Land
            </button>
            <button disabled={!status.connected} onClick={() => runAction("/api/control/rtl")}>RTL</button>
            <label className="hint">Set Mode</label>
            <div className="mode-control">
              <select className="mode-select" value={modeInput} onChange={(e) => setModeInput(e.target.value)}>
                <option value="GUIDED">GUIDED</option>
                <option value="LOITER">LOITER</option>
                <option value="AUTO">AUTO</option>
                <option value="RTL">RTL</option>
                <option value="LAND">LAND</option>
                <option value="STABILIZE">STABILIZE</option>
              </select>
              <button className="mode-apply-btn" disabled={!status.connected} onClick={() => runAction("/api/control/set_mode", { mode: modeInput })}>Apply Mode</button>
            </div>
            <button className="danger" disabled={!status.connected} onClick={() => runAction("/api/control/rtl", null, { confirmText: "Abort the current vehicle action and return to launch?" })}>Abort / RTL</button>
          </ControlSection>

          <ControlSection title="Mission" hint="Mission drawing and scan execution controls stay next to the map.">
            <p className="hint">{missionState.hasMission ? `Loaded • ${missionState.targetCount || 0} waypoints` : "No mission loaded yet."}</p>
            <p className="hint">
              {missionState.mode === "orbit_scan"
                ? "Orbit missions do not use coverage tracking."
                : (missionState.coverageActive ? "Coverage tracking active." : "Coverage inactive until scan motion starts.")}
            </p>
          </ControlSection>

          <ControlSection title="Map / View" hint="Quick operator tools that do not change vehicle behavior.">
            <button className="small-btn" onClick={() => setSidebarCollapsed((v) => !v)}>
              {sidebarCollapsed ? "Show Control Column" : "Collapse Control Column"}
            </button>
            <button className="small-btn" onClick={resetCoverage}>Reset Coverage</button>
          </ControlSection>

          <ControlSection title="Debug" hint="Use these cues during demos when something looks wrong.">
            <p className="hint">Backend: {BACKEND_BASE}/api/health</p>
            <p className="hint">Telemetry stream: {telemetryStream.connected ? "connected" : "reconnecting"}</p>
            {status.last_error ? <p className="bad hint">{status.last_error}</p> : <p className="hint">No vehicle error reported.</p>}
          </ControlSection>
        </aside>

        <section className="panel map-panel">
          <div className="panel-header">
            <div>
              <h2>Mission Map</h2>
              <p className="hint">Planned path and live vehicle position stay together for faster operator decisions.</p>
            </div>
          </div>
          <MapPanel
            telemetry={telemetry}
            mavConnected={Boolean(status.connected)}
            sidebarVersion={sidebarCollapsed ? 1 : 0}
            coverageVersion={coverageVersion}
            onCoverageUpdate={onCoverageUpdate}
            onMissionStateChange={handleMissionStateChange}
          />
        </section>

        <aside className="panel right-rail">
          <div className="panel-header">
            <div>
              <h2>Telemetry</h2>
              <p className="hint">Cards highlight key flight values and scan progress without opening another panel.</p>
            </div>
          </div>
          <TelemetryCards
            items={telemetryItems}
            emptyText={health.ok ? "Waiting for telemetry from the vehicle..." : "Backend is not available yet."}
          />
          {missionState.mode !== "orbit_scan" ? (
            <div className="coverage-box">
              <div className="coverage-head">
                <strong>Coverage</strong>
                <span className="chip">{missionState.coverageActive ? "Active" : "Idle"}</span>
              </div>
              {coverageStats ? (
                <div className="coverage-grid">
                  <div><span>Coverage %</span><strong>{fmtPct(coverageStats?.coverage_pct)}</strong></div>
                  <div><span>Overlap %</span><strong>{fmtPct(coverageStats?.overlap_pct)}</strong></div>
                  <div><span>Covered Cells</span><strong>{coverageStats?.covered_cells ?? "--"}</strong></div>
                  <div><span>Total Cells</span><strong>{coverageStats?.total_cells ?? "--"}</strong></div>
                  <div><span>Total Hits</span><strong>{coverageStats?.total_hits ?? "--"}</strong></div>
                  <div><span>Elapsed</span><strong>{fmt(coverageStats?.time_elapsed_s, " s")}</strong></div>
                </div>
              ) : (
                <div className="empty-card">Coverage will appear once an area mission starts producing hits.</div>
              )}
            </div>
          ) : null}

          <div className="panel-header compact">
            <div>
              <h2>Event Log</h2>
              <p className="hint">Connection changes, mission transitions, and operator actions appear here.</p>
            </div>
          </div>
          <EventLog items={eventItems} emptyText="No activity yet. Connect a vehicle or load a mission to begin." />
        </aside>
      </section>
    </main>
  );
}
