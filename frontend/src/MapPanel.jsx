import { useEffect, useMemo, useRef, useState } from "react";
import {
  CircleMarker,
  MapContainer,
  Polygon,
  Polyline,
  Rectangle,
  TileLayer,
  Tooltip,
  useMap,
  useMapEvents,
} from "react-leaflet";

const BACKEND_BASE = import.meta.env.VITE_BACKEND_BASE || "http://127.0.0.1:8000";

function toLatLng(p) {
  if (!Array.isArray(p) || p.length < 2) return null;
  return [Number(p[1]), Number(p[0])];
}

function toLngLat(p) {
  if (!Array.isArray(p) || p.length < 2) return null;
  return [Number(p[1]), Number(p[0])];
}

function metersToDegLat(m) {
  return m / 111320.0;
}

function metersToDegLng(m, latDeg) {
  const c = Math.cos((latDeg * Math.PI) / 180.0);
  return m / (111320.0 * Math.max(0.1, Math.abs(c)));
}

function fmtNum(v, digits = 1, suffix = "") {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "--";
  return `${Number(v).toFixed(digits)}${suffix}`;
}

function ResizeSync() {
  const map = useMap();
  useEffect(() => {
    let raf = 0;
    const run = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => map.invalidateSize());
    };
    run();
    const onResize = () => run();
    window.addEventListener("resize", onResize);
    const ro = new ResizeObserver(() => run());
    ro.observe(map.getContainer());
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      ro.disconnect();
    };
  }, [map]);
  return null;
}

function MapCenterSync({ center }) {
  const map = useMap();
  const lastKeyRef = useRef("");
  useEffect(() => {
    if (!Array.isArray(center) || center.length < 2) return;
    const lat = Number(center[0]);
    const lng = Number(center[1]);
    if (!Number.isFinite(lat) || !Number.isFinite(lng)) return;
    const key = `${lat.toFixed(7)},${lng.toFixed(7)}`;
    if (key === lastKeyRef.current) return;
    lastKeyRef.current = key;
    map.setView([lat, lng], map.getZoom(), { animate: false });
  }, [map, center]);
  return null;
}

function MissionInteractionLock({ interactionLocked }) {
  const map = useMap();

  useEffect(() => {
    const disable = () => {
      map.dragging?.disable();
      map.doubleClickZoom?.disable();
      map.boxZoom?.disable();
      map.keyboard?.disable();
      map.scrollWheelZoom?.disable();
      map.touchZoom?.disable();
      map.dragRotate?.disable?.();
      map.touchZoomRotate?.disable?.();
      map.tap?.disable?.();
    };
    const enable = () => {
      map.dragging?.enable();
      map.doubleClickZoom?.enable();
      map.boxZoom?.enable();
      map.keyboard?.enable();
      map.scrollWheelZoom?.enable();
      map.touchZoom?.enable();
      map.dragRotate?.enable?.();
      map.touchZoomRotate?.enable?.();
      map.tap?.enable?.();
    };
    if (interactionLocked) disable();
    else enable();
    return () => enable();
  }, [map, interactionLocked]);

  return null;
}

function MissionMapEvents({ drawModeRef, startModeRef, onDrawPoint, onSetStart, onFinishDraw }) {
  useMapEvents({
    mousedown(e) {
      const active = Boolean(drawModeRef.current || startModeRef.current);
      if (!active) return;
      if (e.originalEvent?.preventDefault) e.originalEvent.preventDefault();
      if (e.originalEvent?.stopPropagation) e.originalEvent.stopPropagation();
    },
    click(e) {
      if (drawModeRef.current) {
        onDrawPoint([e.latlng.lat, e.latlng.lng]);
      } else if (startModeRef.current) {
        onSetStart([e.latlng.lat, e.latlng.lng]);
      }
    },
    dblclick() {
      if (drawModeRef.current) {
        onFinishDraw();
      }
    },
  });
  return null;
}

export function MapPanel({
  telemetry,
  mavConnected = false,
  sidebarVersion = 0,
  coverageVersion = 0,
  onCoverageUpdate = null,
}) {
  const [mapState, setMapState] = useState(null);
  const [track, setTrack] = useState([]);
  const [coverage, setCoverage] = useState(null);
  const [scanDebug, setScanDebug] = useState(null);

  const [missionArea, setMissionArea] = useState([]);
  const [missionStart, setMissionStart] = useState(null);
  const [missionPath, setMissionPath] = useState([]);
  const [missionPreview, setMissionPreview] = useState(null);
  const [drawDraft, setDrawDraft] = useState([]);
  const [missionMode, setMissionMode] = useState("none");
  const [missionMsg, setMissionMsg] = useState("");
  const [scanSpacingM, setScanSpacingM] = useState(8.0);
  const [scanSpeedMps, setScanSpeedMps] = useState(3.0);
  const [simState, setSimState] = useState({ sim_running: false, sim_paused: false, sim_done: false, pose: null });
  const [sitlState, setSitlState] = useState({
    state: "IDLE",
    scan_active: false,
    waypoint_index: 0,
    waypoint_count: 0,
    last_error: "",
  });

  const drawModeRef = useRef(false);
  const startModeRef = useRef(false);
  const simTickTimerRef = useRef(null);

  const [showBounds, setShowBounds] = useState(true);
  const [showTrail, setShowTrail] = useState(true);
  const [showCoverage, setShowCoverage] = useState(true);
  const [showFootprint, setShowFootprint] = useState(true);

  async function fetchJson(path, init) {
    const resp = await fetch(`${BACKEND_BASE}${path}`, init);
    const payload = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      const detail = payload?.detail ? String(payload.detail) : `HTTP ${resp.status}`;
      throw new Error(detail);
    }
    return payload;
  }

  function stopTickLoop() {
    if (simTickTimerRef.current) {
      window.clearInterval(simTickTimerRef.current);
      simTickTimerRef.current = null;
    }
  }

  async function refreshMissionState() {
    try {
      const [s, mp, sim, sitl] = await Promise.all([
        fetchJson("/api/map_state"),
        fetchJson("/api/mission/path"),
        fetchJson("/api/mission/sim/state"),
        fetchJson("/api/sitl/state"),
      ]);
      setMapState(s || null);
      applyMissionPayload(mp || {});
      setSimState({
        sim_running: Boolean(sim?.sim_running),
        sim_paused: Boolean(sim?.sim_paused),
        sim_done: Boolean(sim?.sim_done),
        pose: sim?.pose || null,
      });
      setSitlState({
        state: String(sitl?.state || "IDLE"),
        scan_active: Boolean(sitl?.scan_active),
        waypoint_index: Number(sitl?.waypoint_index || 0),
        waypoint_count: Number(sitl?.waypoint_count || 0),
        last_error: String(sitl?.last_error || ""),
      });
    } catch (_) {
      return;
    }
  }

  function startTickLoop() {
    stopTickLoop();
    simTickTimerRef.current = window.setInterval(async () => {
      try {
        const tick = await fetchJson("/api/mission/sim/tick", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ dt: 0.1 }),
        });
        setSimState({
          sim_running: Boolean(tick?.sim_running),
          sim_paused: Boolean(tick?.sim_paused),
          sim_done: Boolean(tick?.sim_done),
          pose: tick?.pose || null,
        });
        if (!tick?.sim_running || tick?.sim_paused || tick?.sim_done) {
          stopTickLoop();
          await refreshMissionState();
        }
      } catch (err) {
        stopTickLoop();
        setMissionMsg(`Sim tick failed: ${String(err)}`);
      }
    }, 100);
  }

  function applyMissionPayload(payload) {
    const area = Array.isArray(payload?.scan_area_polygon_lng_lat)
      ? payload.scan_area_polygon_lng_lat.map(toLatLng).filter(Boolean)
      : [];
    const startRaw = payload?.start_position_lng_lat;
    const start = Array.isArray(startRaw) && startRaw.length >= 2
      ? [Number(startRaw[1]), Number(startRaw[0])]
      : null;
    const path = Array.isArray(payload?.waypoints_lng_lat)
      ? payload.waypoints_lng_lat.map(toLatLng).filter(Boolean)
      : [];
    setMissionArea(area);
    setMissionStart(start);
    setMissionPath(path);
    setMissionPreview(payload?.coverage_preview || null);
    setSimState((prev) => ({
      sim_running: Boolean(payload?.sim_running ?? prev.sim_running),
      sim_paused: Boolean(payload?.sim_paused ?? prev.sim_paused),
      sim_done: Boolean(payload?.sim_done ?? prev.sim_done),
      pose: payload?.sim || prev.pose,
    }));
  }

  useEffect(() => {
    let cancelled = false;
    let t1 = null;
    async function tick() {
      try {
        const [s, tr, mp, sim, sitl] = await Promise.all([
          fetchJson("/api/map_state"),
          fetchJson("/api/track?limit=600"),
          fetchJson("/api/mission/path"),
          fetchJson("/api/mission/sim/state"),
          fetchJson("/api/sitl/state"),
        ]);
        if (!cancelled) {
          setMapState(s || null);
          setTrack(Array.isArray(tr?.items) ? tr.items : []);
          applyMissionPayload(mp || {});
          setSimState({
            sim_running: Boolean(sim?.sim_running),
            sim_paused: Boolean(sim?.sim_paused),
            sim_done: Boolean(sim?.sim_done),
            pose: sim?.pose || null,
          });
          setSitlState({
            state: String(sitl?.state || "IDLE"),
            scan_active: Boolean(sitl?.scan_active),
            waypoint_index: Number(sitl?.waypoint_index || 0),
            waypoint_count: Number(sitl?.waypoint_count || 0),
            last_error: String(sitl?.last_error || ""),
          });
          setMissionMsg("");
        }
      } catch (err) {
        if (!cancelled) {
          setTrack([]);
          setMissionMsg(String(err));
        }
      }
      if (!cancelled) t1 = window.setTimeout(tick, 1000);
    }
    tick();
    return () => {
      cancelled = true;
      if (t1) window.clearTimeout(t1);
      stopTickLoop();
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    let t1 = null;
    async function tick() {
      try {
        const [c, d] = await Promise.all([
          fetchJson("/api/coverage"),
          fetchJson("/api/scan/debug"),
        ]);
        if (!cancelled) {
          setCoverage(c || null);
          setScanDebug(d || null);
          if (onCoverageUpdate && c?.stats) onCoverageUpdate(c.stats);
        }
      } catch (_) {
        if (!cancelled && onCoverageUpdate) onCoverageUpdate(null);
      }
      if (!cancelled) t1 = window.setTimeout(tick, 1000);
    }
    tick();
    return () => {
      cancelled = true;
      if (t1) window.clearTimeout(t1);
    };
  }, [coverageVersion, onCoverageUpdate]);

  const center = useMemo(() => {
    if (mapState?.center_lng_lat) {
      const p = toLatLng(mapState.center_lng_lat);
      if (p) return p;
    }
    return [39.90923, 116.397428];
  }, [mapState]);

  const boundsPolygon = useMemo(() => {
    if (!Array.isArray(mapState?.bounds_polygon_lng_lat)) return [];
    return mapState.bounds_polygon_lng_lat.map(toLatLng).filter(Boolean);
  }, [mapState]);

  const trail = useMemo(
    () =>
      track
        .map((p) => [Number(p.lat), Number(p.lon)])
        .filter((p) => Number.isFinite(p[0]) && Number.isFinite(p[1])),
    [track],
  );

  const coverageRects = useMemo(() => {
    const cells = Array.isArray(coverage?.covered_cells) ? coverage.covered_cells : [];
    return cells.map((c) => ({
      bounds: [
        [Number(c.lat_min), Number(c.lng_min)],
        [Number(c.lat_max), Number(c.lng_max)],
      ],
      count: Number(c.count || 0),
    }));
  }, [coverage]);

  const footprint = useMemo(() => {
    const pts = Array.isArray(scanDebug?.footprint_polygon_lng_lat)
      ? scanDebug.footprint_polygon_lng_lat.map(toLatLng).filter(Boolean)
      : [];
    return pts.length > 2 ? pts : null;
  }, [scanDebug]);

  const origin = useMemo(() => {
    if (!mapState?.origin) return null;
    const lat = Number(mapState.origin.lat);
    const lng = Number(mapState.origin.lng);
    if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null;
    return [lat, lng];
  }, [mapState]);

  const vehicle = useMemo(() => {
    const src = mapState?.vehicle || {};
    const simPose = simState?.pose || null;
    const lat = Number(telemetry?.lat ?? src?.lat ?? simPose?.lat);
    const lng = Number(telemetry?.lon ?? src?.lng ?? simPose?.lng);
    const yaw = Number(telemetry?.yaw_deg ?? src?.yaw_deg ?? simPose?.yaw_deg);
    if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null;
    return { lat, lng, yaw: Number.isFinite(yaw) ? yaw : null };
  }, [mapState, simState, telemetry]);

  const headingLine = useMemo(() => {
    if (!vehicle || vehicle.yaw === null) return null;
    const lenM = 20;
    const yawRad = (vehicle.yaw * Math.PI) / 180.0;
    const dNorth = Math.cos(yawRad) * lenM;
    const dEast = Math.sin(yawRad) * lenM;
    const dLat = metersToDegLat(dNorth);
    const dLng = metersToDegLng(dEast, vehicle.lat);
    return [
      [vehicle.lat, vehicle.lng],
      [vehicle.lat + dLat, vehicle.lng + dLng],
    ];
  }, [vehicle]);

  async function saveAreaToBackend(pointsLatLng) {
    const polygonLngLat = pointsLatLng.map(toLngLat).filter(Boolean);
    const payload = await fetchJson("/api/mission/area", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ polygon_lng_lat: polygonLngLat }),
    });
    setMissionMsg(`Area saved (${payload.points} points)`);
  }

  async function saveStartToBackend(pointLatLng) {
    await fetchJson("/api/mission/start_position", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lng: pointLatLng[1], lat: pointLatLng[0] }),
    });
    setMissionMsg("Start position saved");
  }

  async function handleGeneratePath() {
    try {
      const payload = await fetchJson("/api/mission/generate_scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ spacing_m: Number(scanSpacingM), speed_m_s: Number(scanSpeedMps), start_scan: false }),
      });
      applyMissionPayload(payload);
      setMissionMsg("Scan path generated");
    } catch (err) {
      setMissionMsg(`Generate failed: ${String(err)}`);
    }
  }

  async function handleSimStart() {
    try {
      const state = await fetchJson("/api/mission/sim/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      setSimState({
        sim_running: Boolean(state?.sim_running),
        sim_paused: Boolean(state?.sim_paused),
        sim_done: Boolean(state?.sim_done),
        pose: state?.pose || null,
      });
      startTickLoop();
      setMissionMsg("Simulation started");
    } catch (err) {
      setMissionMsg(`Start sim failed: ${String(err)}`);
    }
  }

  async function handleSitlStart() {
    try {
      const state = await fetchJson("/api/sitl/start_scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ alt_m: 10.0, accept_radius_m: 3.0 }),
      });
      setSitlState({
        state: String(state?.state || "IDLE"),
        scan_active: Boolean(state?.scan_active),
        waypoint_index: Number(state?.waypoint_index || 0),
        waypoint_count: Number(state?.waypoint_count || 0),
        last_error: String(state?.last_error || ""),
      });
      setMissionMsg("SITL scan started");
    } catch (err) {
      setMissionMsg(`Start SITL failed: ${String(err)}`);
    }
  }

  async function handleSitlStop() {
    try {
      const state = await fetchJson("/api/sitl/stop_scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      setSitlState({
        state: String(state?.state || "STOPPED"),
        scan_active: Boolean(state?.scan_active),
        waypoint_index: Number(state?.waypoint_index || 0),
        waypoint_count: Number(state?.waypoint_count || 0),
        last_error: String(state?.last_error || ""),
      });
      setMissionMsg("SITL scan stopped");
    } catch (err) {
      setMissionMsg(`Stop SITL failed: ${String(err)}`);
    }
  }

  async function handleSimPause() {
    try {
      const state = await fetchJson("/api/mission/sim/pause", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      setSimState({
        sim_running: Boolean(state?.sim_running),
        sim_paused: Boolean(state?.sim_paused),
        sim_done: Boolean(state?.sim_done),
        pose: state?.pose || null,
      });
      stopTickLoop();
      setMissionMsg("Simulation paused");
    } catch (err) {
      setMissionMsg(`Pause sim failed: ${String(err)}`);
    }
  }

  async function handleSimStop() {
    try {
      const state = await fetchJson("/api/mission/sim/stop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      setSimState({
        sim_running: Boolean(state?.sim_running),
        sim_paused: Boolean(state?.sim_paused),
        sim_done: Boolean(state?.sim_done),
        pose: state?.pose || null,
      });
      stopTickLoop();
      await refreshMissionState();
      setMissionMsg("Simulation stopped");
    } catch (err) {
      setMissionMsg(`Stop sim failed: ${String(err)}`);
    }
  }

  async function handleClearMission() {
    try {
      await fetchJson("/api/mission/clear", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      stopTickLoop();
      setMissionArea([]);
      setMissionStart(null);
      setMissionPath([]);
      setMissionPreview(null);
      setDrawDraft([]);
      setMissionMode("none");
      setSimState({ sim_running: false, sim_paused: false, sim_done: false, pose: null });
      setMissionMsg("Mission cleared");
    } catch (err) {
      setMissionMsg(`Clear failed: ${String(err)}`);
    }
  }

  function handleDrawClick(p) {
    if (missionMode !== "draw") return;
    setDrawDraft((prev) => [...prev, p]);
  }

  async function handleDrawFinish() {
    if (missionMode !== "draw") return;
    if (drawDraft.length < 3) {
      setMissionMsg("Need at least 3 points for scan area");
      return;
    }
    const area = [...drawDraft];
    setMissionArea(area);
    setDrawDraft([]);
    setMissionMode("none");
    try {
      await saveAreaToBackend(area);
    } catch (err) {
      setMissionMsg(`Save area failed: ${String(err)}`);
    }
  }

  async function handleSetStart(p) {
    if (missionMode !== "start") return;
    setMissionStart(p);
    setMissionMode("none");
    try {
      await saveStartToBackend(p);
    } catch (err) {
      setMissionMsg(`Save start failed: ${String(err)}`);
    }
  }

  const tiles = mapState?.tile_url_template
    ? `${BACKEND_BASE}${mapState.tile_url_template}`
    : "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";

  useEffect(() => {
    drawModeRef.current = missionMode === "draw";
    startModeRef.current = missionMode === "start";
  }, [missionMode]);
  const interactionLocked = missionMode === "draw" || missionMode === "start";

  const hasPath = missionPath.length >= 2;
  const simStatusLabel = simState.sim_running
    ? (simState.sim_paused ? "PAUSED" : "RUNNING")
    : "STOPPED";

  return (
    <>
      <div className="map-toolbar">
        <button className="small-btn" onClick={() => {
          setMissionMode("draw");
          setDrawDraft([]);
          setMissionMsg("Draw mode: click to add points, double-click to finish");
        }}>Draw Scan Area</button>
        <button className="small-btn" onClick={() => {
          setMissionMode("start");
          setMissionMsg("Start mode: click map to place drone start");
        }}>Set Start Position</button>
        <button className="small-btn" disabled={simState.sim_running || sitlState.scan_active} onClick={handleGeneratePath}>Generate Scan Path</button>
        <button className="small-btn" onClick={handleClearMission}>Clear Mission</button>
        {mavConnected ? (
          <>
            <button className="small-btn" disabled={!hasPath || sitlState.scan_active} onClick={handleSitlStart}>Start SITL Scan</button>
            <button className="small-btn" disabled={!sitlState.scan_active && sitlState.state !== "ARMING" && sitlState.state !== "TAKEOFF" && sitlState.state !== "RUN_PATH"} onClick={handleSitlStop}>Stop SITL Scan</button>
            <span className="chip"><strong>Executor:</strong> {sitlState.state}</span>
          </>
        ) : (
          <>
            <button className="small-btn" disabled={!hasPath || (simState.sim_running && !simState.sim_paused)} onClick={handleSimStart}>Start Sim</button>
            <button className="small-btn" disabled={!simState.sim_running || simState.sim_paused} onClick={handleSimPause}>Pause Sim</button>
            <button className="small-btn" disabled={!hasPath} onClick={handleSimStop}>Stop Sim</button>
            <span className="chip"><strong>Sim:</strong> {simStatusLabel}</span>
          </>
        )}
        <label className="map-input">Spacing (m)
          <input type="number" min="1" step="0.5" value={scanSpacingM} onChange={(e) => setScanSpacingM(e.target.value)} />
        </label>
        <label className="map-input">Speed (m/s)
          <input type="number" min="0.2" step="0.2" value={scanSpeedMps} onChange={(e) => setScanSpeedMps(e.target.value)} />
        </label>
      </div>

      <div className="map-toolbar">
        <label><input type="checkbox" checked={showBounds} onChange={(e) => setShowBounds(e.target.checked)} /> Bounds</label>
        <label><input type="checkbox" checked={showTrail} onChange={(e) => setShowTrail(e.target.checked)} /> Trail</label>
        <label><input type="checkbox" checked={showCoverage} onChange={(e) => setShowCoverage(e.target.checked)} /> Coverage</label>
        <label><input type="checkbox" checked={showFootprint} onChange={(e) => setShowFootprint(e.target.checked)} /> Footprint</label>
      </div>

      <div className="mission-preview">
        <span>Mode: {missionMode === "none" ? "idle" : missionMode}</span>
        <span>Expected Coverage: {fmtNum(missionPreview?.expected_coverage_pct, 1, "%")}</span>
        <span>Estimated Time: {fmtNum(missionPreview?.estimated_time_s, 1, " s")}</span>
        <span>Passes: {missionPreview?.number_of_passes ?? "--"}</span>
        <span>Path Length: {fmtNum(missionPreview?.path_length_m, 1, " m")}</span>
      </div>
      {missionMsg ? <p className="hint">{missionMsg}</p> : null}

      <MapContainer
        key={`${sidebarVersion}`}
        center={center}
        zoom={Number(mapState?.zoom || 16)}
        minZoom={1}
        maxZoom={20}
        className="leaflet-map"
        preferCanvas
        doubleClickZoom={false}
      >
        <ResizeSync />
        <MapCenterSync center={center} />
        <MissionInteractionLock interactionLocked={interactionLocked} />
        <MissionMapEvents
          drawModeRef={drawModeRef}
          startModeRef={startModeRef}
          onDrawPoint={handleDrawClick}
          onSetStart={handleSetStart}
          onFinishDraw={handleDrawFinish}
        />
        <TileLayer url={tiles} maxZoom={20} tileSize={256} detectRetina={false} />

        {showBounds && boundsPolygon.length > 2 ? (
          <Polygon positions={boundsPolygon} pathOptions={{ color: "#0f766e", weight: 2, fillOpacity: 0.08 }} />
        ) : null}

        {missionArea.length > 2 ? (
          <Polygon positions={missionArea} pathOptions={{ color: "#1d4ed8", weight: 3, fillOpacity: 0.1 }} />
        ) : null}

        {drawDraft.length > 1 ? (
          <Polyline positions={drawDraft} pathOptions={{ color: "#1d4ed8", weight: 2, dashArray: "5,5" }} />
        ) : null}

        {missionPath.length > 1 ? (
          <Polyline positions={missionPath} pathOptions={{ color: "#be123c", weight: 3, opacity: 0.95 }} />
        ) : null}

        {missionStart ? (
          <CircleMarker center={missionStart} radius={7} pathOptions={{ color: "#f59e0b", fillOpacity: 0.9 }}>
            <Tooltip permanent direction="top">Drone Start</Tooltip>
          </CircleMarker>
        ) : null}

        {showTrail && trail.length > 1 ? (
          <Polyline positions={trail} pathOptions={{ color: "#2563eb", weight: 3, opacity: 0.8 }} />
        ) : null}

        {showCoverage
          ? coverageRects.map((r, idx) => {
              const alpha = Math.min(0.65, 0.15 + Math.log2(Math.max(1, r.count)) * 0.1);
              return (
                <Rectangle
                  key={`cov-${idx}`}
                  bounds={r.bounds}
                  pathOptions={{ color: "#16a34a", weight: 0.5, fillOpacity: alpha }}
                />
              );
            })
          : null}

        {showFootprint && footprint ? (
          <Polygon positions={footprint} pathOptions={{ color: "#f97316", weight: 2, fillOpacity: 0.05 }} />
        ) : null}

        {origin ? (
          <CircleMarker center={origin} radius={6} pathOptions={{ color: "#f59e0b", fillOpacity: 0.8 }}>
            <Tooltip permanent direction="top">Origin</Tooltip>
          </CircleMarker>
        ) : null}

        {vehicle ? (
          <CircleMarker center={[vehicle.lat, vehicle.lng]} radius={7} pathOptions={{ color: "#dc2626", fillOpacity: 0.9 }}>
            <Tooltip permanent direction="top">Vehicle</Tooltip>
          </CircleMarker>
        ) : null}

        {headingLine ? (
          <Polyline positions={headingLine} pathOptions={{ color: "#dc2626", weight: 3 }} />
        ) : null}
      </MapContainer>
    </>
  );
}
