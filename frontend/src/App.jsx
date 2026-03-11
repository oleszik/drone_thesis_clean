import { useCallback, useEffect, useMemo, useState } from "react";
import { MapPanel } from "./MapPanel";

const BACKEND_BASE = import.meta.env.VITE_BACKEND_BASE || "http://127.0.0.1:8000";

async function fetchJson(path, init) {
  const resp = await fetch(`${BACKEND_BASE}${path}`, init);
  const payload = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const detail = payload?.detail ? String(payload.detail) : `HTTP ${resp.status}`;
    throw new Error(detail);
  }
  return payload;
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
  });
  const [actionMsg, setActionMsg] = useState("");
  const [modeInput, setModeInput] = useState("GUIDED");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [coverageStats, setCoverageStats] = useState(null);
  const [coverageVersion, setCoverageVersion] = useState(0);
  const [runInfo, setRunInfo] = useState(null);
  const [runScenario, setRunScenario] = useState("A2_40x40_alt10");
  const [runController, setRunController] = useState("SITL_GUIDED_lawnmower");
  const [runNotes, setRunNotes] = useState("");

  useEffect(() => {
    let cancelled = false;
    let healthTimer = null;
    let statusTimer = null;
    let runTimer = null;

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

    async function runStatus() {
      try {
        const [s, t] = await Promise.all([fetchJson("/api/status"), fetchJson("/api/telemetry")]);
        if (!cancelled) {
          setStatus((prev) => ({ ...prev, ...s }));
          setTelemetry((prev) => ({ ...prev, ...t }));
        }
      } catch (_) {
        if (!cancelled) {
          setStatus((prev) => ({ ...prev, connected: false }));
        }
      }
      if (!cancelled) {
        statusTimer = window.setTimeout(runStatus, 500);
      }
    }

    async function runMeta() {
      try {
        const payload = await fetchJson("/api/runs/current");
        const run = payload?.run || null;
        if (!cancelled) {
          setRunInfo(run);
          if (run && typeof run.notes === "string") setRunNotes(run.notes);
        }
      } catch (_) {
        if (!cancelled) setRunInfo(null);
      }
      if (!cancelled) runTimer = window.setTimeout(runMeta, 2000);
    }
    runHealth();
    runStatus();
    runMeta();
    return () => {
      cancelled = true;
      if (healthTimer) window.clearTimeout(healthTimer);
      if (statusTimer) window.clearTimeout(statusTimer);
      if (runTimer) window.clearTimeout(runTimer);
    };
  }, []);

  async function runAction(path, body = null) {
    setActionMsg("Sending...");
    try {
      const payload = await fetchJson(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: body ? JSON.stringify(body) : undefined,
      });
      setActionMsg(`OK: ${payload.action || "done"}`);
    } catch (err) {
      setActionMsg(`Error: ${String(err)}`);
    }
  }

  async function resetCoverage() {
    setActionMsg("Resetting coverage...");
    try {
      await fetchJson("/api/coverage/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      setCoverageVersion((v) => v + 1);
      setActionMsg("OK: coverage reset");
    } catch (err) {
      setActionMsg(`Error: ${String(err)}`);
    }
  }

  async function startRun() {
    setActionMsg("Starting run...");
    try {
      const payload = await fetchJson("/api/runs/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scenario: { label: runScenario, map_provider: health.mapProvider },
          controller: { label: runController },
          notes: runNotes,
        }),
      });
      setRunInfo(payload || null);
      setActionMsg(`OK: run ${payload?.run_id || ""} started`);
    } catch (err) {
      setActionMsg(`Error: ${String(err)}`);
    }
  }

  async function stopRun() {
    setActionMsg("Stopping run...");
    try {
      const payload = await fetchJson("/api/runs/stop", { method: "POST" });
      setRunInfo(payload || null);
      setActionMsg(`OK: run ${payload?.run_id || ""} stopped`);
    } catch (err) {
      setActionMsg(`Error: ${String(err)}`);
    }
  }

  async function saveRunNotes() {
    if (!runInfo) return;
    setActionMsg("Saving notes...");
    try {
      const payload = await fetchJson("/api/runs/current/notes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ notes: runNotes }),
      });
      setRunInfo(payload || null);
      setActionMsg("OK: notes saved");
    } catch (err) {
      setActionMsg(`Error: ${String(err)}`);
    }
  }

  async function downloadExport(path, filename) {
    try {
      const resp = await fetch(`${BACKEND_BASE}${path}`);
      if (!resp.ok) {
        const payload = await resp.json().catch(() => ({}));
        throw new Error(payload?.detail || `HTTP ${resp.status}`);
      }
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      setActionMsg(`OK: exported ${filename}`);
    } catch (err) {
      setActionMsg(`Error: ${String(err)}`);
    }
  }

  function fmt(value, suffix = "") {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
    return `${Number(value).toFixed(1)}${suffix}`;
  }

  function fmtPct(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
    return `${Number(value).toFixed(1)}%`;
  }

  const onCoverageUpdate = useCallback((stats) => {
    setCoverageStats(stats || null);
  }, []);

  const chips = useMemo(
    () => [
      { label: "Connection", value: status.connected ? "Connected" : "Disconnected" },
      { label: "Mode", value: status.mode || "UNKNOWN" },
      { label: "Armed", value: status.armed ? "Yes" : "No" },
      { label: "Map", value: health.mapProvider },
    ],
    [status.connected, status.mode, status.armed, health.mapProvider],
  );

  return (
    <main className="app">
      <header className="top">
        <h1>Drone Thesis Dashboard</h1>
        <button className="small-btn" onClick={() => setSidebarCollapsed((v) => !v)}>
          {sidebarCollapsed ? "Show Controls" : "Hide Controls"}
        </button>
        <div className="chips">
          {chips.map((c) => (
            <span key={c.label} className="chip">
              <strong>{c.label}:</strong> {c.value}
            </span>
          ))}
        </div>
      </header>

      <section className="status-row">
        <p className={health.ok ? "ok" : "bad"}>{health.message}</p>
        <p className="hint">Backend: {BACKEND_BASE}/api/health</p>
        {actionMsg ? <p className="hint">{actionMsg}</p> : null}
      </section>

      <section className="panel run-panel">
        <h2>Experiment Run</h2>
        <div className="run-grid">
          <div><span className="hint">Run ID</span><strong>{runInfo?.run_id || "--"}</strong></div>
          <div><span className="hint">Status</span><strong>{runInfo?.status || "idle"}</strong></div>
          <label className="run-field">Scenario<input value={runScenario} onChange={(e) => setRunScenario(e.target.value)} /></label>
          <label className="run-field">Controller<input value={runController} onChange={(e) => setRunController(e.target.value)} /></label>
          <label className="run-field run-notes">Notes<textarea value={runNotes} onChange={(e) => setRunNotes(e.target.value)} rows={2} /></label>
          <div className="run-actions">
            <button className="small-btn" disabled={runInfo?.status === "running"} onClick={startRun}>Start Run</button>
            <button className="small-btn" disabled={!runInfo || runInfo?.status !== "running"} onClick={stopRun}>Stop Run</button>
            <button className="small-btn" disabled={!runInfo} onClick={saveRunNotes}>Save Notes</button>
            <button className="small-btn" disabled={!runInfo} onClick={() => downloadExport("/api/runs/current/export/json", "run_export.json")}>Export Run JSON</button>
            <button className="small-btn" disabled={!runInfo} onClick={() => downloadExport("/api/runs/current/export/path.csv", "run_path.csv")}>Export Path CSV</button>
            <button className="small-btn" disabled={!runInfo} onClick={() => downloadExport("/api/runs/current/export/path.geojson", "run_path.geojson")}>Export Path GeoJSON</button>
            <button className="small-btn" disabled={!runInfo} onClick={() => downloadExport("/api/runs/current/export/coverage.csv", "run_coverage.csv")}>Export Coverage CSV</button>
            <button className="small-btn" disabled={!runInfo} onClick={() => downloadExport("/api/runs/current/export/report.pdf", "run_report.pdf")}>Generate Report PDF</button>
          </div>
          <div className="run-events">
            <span className="hint">Event Log</span>
            <div className="event-list">
              {(runInfo?.events || []).slice(-12).reverse().map((ev, idx) => (
                <div key={`${ev.t_utc}-${idx}`} className="event-row">
                  <code>{ev.t_utc}</code> <strong>{ev.event}</strong>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className={`layout ${sidebarCollapsed ? "collapsed" : ""}`}>
        <aside className="panel">
          <h2>Controls</h2>
          <button disabled={status.connected} onClick={() => runAction("/api/connection/connect")}>Connect</button>
          <button disabled={!status.connected} onClick={() => runAction("/api/connection/disconnect")}>Disconnect</button>
          <button disabled={!status.connected} onClick={() => runAction("/api/control/arm")}>Arm</button>
          <button disabled={!status.connected} onClick={() => runAction("/api/control/disarm")}>Disarm</button>
          <button disabled={!status.connected} onClick={() => runAction("/api/control/takeoff", { alt_m: 10 })}>Takeoff</button>
          <button disabled={!status.connected} onClick={() => runAction("/api/control/land")}>Land</button>
          <button disabled={!status.connected} onClick={() => runAction("/api/control/rtl")}>RTL</button>
          <label className="hint">Set Mode</label>
          <select value={modeInput} onChange={(e) => setModeInput(e.target.value)}>
            <option value="GUIDED">GUIDED</option>
            <option value="LOITER">LOITER</option>
            <option value="AUTO">AUTO</option>
            <option value="RTL">RTL</option>
            <option value="LAND">LAND</option>
            <option value="STABILIZE">STABILIZE</option>
          </select>
          <button disabled={!status.connected} onClick={() => runAction("/api/control/set_mode", { mode: modeInput })}>Apply Mode</button>
          <button className="danger" disabled={!status.connected} onClick={() => runAction("/api/control/rtl")}>Abort</button>
        </aside>

        <section className="panel map-panel">
          <h2>Map</h2>
          <MapPanel
            telemetry={telemetry}
            mavConnected={Boolean(status.connected)}
            sidebarVersion={sidebarCollapsed ? 1 : 0}
            coverageVersion={coverageVersion}
            onCoverageUpdate={onCoverageUpdate}
          />
        </section>

        <aside className="panel">
          <h2>Telemetry</h2>
          <div className="telemetry-grid">
            <div className="metric"><span>Altitude</span><strong>{fmt(telemetry.rel_alt_m, " m")}</strong></div>
            <div className="metric"><span>Speed</span><strong>{fmt(telemetry.speed_m_s, " m/s")}</strong></div>
            <div className="metric"><span>Yaw</span><strong>{fmt(telemetry.yaw_deg, " deg")}</strong></div>
            <div className="metric"><span>Battery</span><strong>{fmt(telemetry.battery_percent, " %")}</strong></div>
            <div className="metric"><span>GPS Fix</span><strong>{telemetry.gps_fix ?? "--"}</strong></div>
            <div className="metric"><span>EKF</span><strong>{telemetry.ekf_ok === null ? "--" : telemetry.ekf_ok ? "OK" : "BAD"}</strong></div>
          </div>
          <div className="coverage-box">
            <div className="coverage-head">
              <strong>Coverage</strong>
              <button className="small-btn" onClick={resetCoverage}>Reset coverage</button>
            </div>
            <div className="coverage-grid">
              <div><span>Coverage %</span><strong>{fmtPct(coverageStats?.coverage_pct)}</strong></div>
              <div><span>Overlap %</span><strong>{fmtPct(coverageStats?.overlap_pct)}</strong></div>
              <div><span>Covered Cells</span><strong>{coverageStats?.covered_cells ?? "--"}</strong></div>
              <div><span>Total Cells</span><strong>{coverageStats?.total_cells ?? "--"}</strong></div>
              <div><span>Total Hits</span><strong>{coverageStats?.total_hits ?? "--"}</strong></div>
              <div><span>Elapsed</span><strong>{fmt(coverageStats?.time_elapsed_s, " s")}</strong></div>
            </div>
          </div>
          <p className="hint">Link: {status.connection_url || "--"}</p>
          {status.last_error ? <p className="bad hint">{status.last_error}</p> : null}
        </aside>
      </section>
    </main>
  );
}
