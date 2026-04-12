import { useEffect, useState } from "react";

const BACKEND_BASE = import.meta.env.VITE_BACKEND_BASE || "http://127.0.0.1:8000";

async function fetchJson(path) {
  const resp = await fetch(`${BACKEND_BASE}${path}`);
  const payload = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(payload?.detail ? String(payload.detail) : `HTTP ${resp.status}`);
  return payload;
}

export function Landing() {
  const [system, setSystem] = useState({
    backendOnline: false,
    runtimeMode: "--",
    mapProvider: "--",
    keyConfigured: false,
    error: "",
  });

  useEffect(() => {
    let cancelled = false;
    let timer = null;

    async function pollHealth() {
      try {
        const payload = await fetchJson("/api/health");
        if (cancelled) return;
        setSystem({
          backendOnline: Boolean(payload?.ok),
          runtimeMode: String(payload?.runtime_mode || "--"),
          mapProvider: String(payload?.map_provider || "--"),
          keyConfigured: Boolean(payload?.tencent_key_configured),
          error: "",
        });
      } catch (err) {
        if (cancelled) return;
        setSystem((prev) => ({
          ...prev,
          backendOnline: false,
          error: String(err),
        }));
      }
      if (!cancelled) timer = window.setTimeout(pollHealth, 5000);
    }

    pollHealth();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, []);

  return (
    <main className="landing">
      <div className="landing-grid-overlay" aria-hidden="true" />
      <div className="landing-radar" aria-hidden="true" />
      <div className="landing-flightpath" aria-hidden="true" />

      <section className="landing-shell">
        <header className="landing-hero">
          <span className="landing-kicker">Autonomous Flight System</span>
          <h1>Drone Mission Control</h1>
          <p>
            Enter the operations gateway to run simulation workflows or execute safety-gated real mission procedures.
          </p>
        </header>

        <section className="landing-mode-grid" aria-label="mode-selection">
          <a className="landing-mode-card sim" href="/sim">
            <div className="landing-mode-head">
              <h2>Simulation</h2>
              <span className="mode-tag">Dev/Test</span>
            </div>
            <p>Design and iterate mission geometry with rich simulator controls.</p>
            <ul>
              <li>Mission design + drawing tools</li>
              <li>Path generation experiments</li>
              <li>SITL execution controls</li>
              <li>Coverage and debug analysis</li>
            </ul>
            <span className="mode-enter">Enter Simulation →</span>
          </a>

          <a className="landing-mode-card real" href="/real-test">
            <div className="landing-mode-head">
              <h2>Real Mission</h2>
              <span className="mode-tag">Field Ops</span>
            </div>
            <p>Operate with readiness-first safety flow and operator-focused mission planning.</p>
            <ul>
              <li>Preflight readiness checks</li>
              <li>Radio and field link verification</li>
              <li>Safety intervention controls</li>
              <li>Approved mission execution path</li>
            </ul>
            <span className="mode-enter">Enter Real Mission →</span>
          </a>
        </section>

        <section className="landing-status" aria-label="system-status">
          <h3>System Status</h3>
          <div className="landing-status-row">
            <span className={`chip ${system.backendOnline ? "tone-good" : "tone-bad"}`}>
              <strong>Backend:</strong> {system.backendOnline ? "online" : "offline"}
            </span>
            <span className="chip"><strong>Runtime:</strong> {system.runtimeMode}</span>
            <span className="chip"><strong>Map:</strong> {system.mapProvider}</span>
            <span className={`chip ${system.keyConfigured ? "tone-good" : "tone-warn"}`}>
              <strong>Map Key:</strong> {system.keyConfigured ? "configured" : "missing"}
            </span>
          </div>
          {system.error ? <p className="hint bad">Health check: {system.error}</p> : null}
        </section>
      </section>
    </main>
  );
}
