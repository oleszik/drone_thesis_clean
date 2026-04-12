import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  CircleMarker,
  MapContainer,
  Marker,
  Polygon,
  Polyline,
  Rectangle,
  TileLayer,
  Tooltip,
  useMap,
  useMapEvents,
} from "react-leaflet";
import L from "leaflet";
import { LayerToggles } from "./components/LayerToggles";
import { useLiveStream } from "./hooks/useLiveStream";

const BACKEND_BASE = import.meta.env.VITE_BACKEND_BASE || "http://127.0.0.1:8000";

function isTencentProvider(mapProvider) {
  return String(mapProvider || "").toLowerCase() === "tencent";
}

function outOfChina(lng, lat) {
  return lng < 72.004 || lng > 137.8347 || lat < 0.8293 || lat > 55.8271;
}

function transformLat(x, y) {
  let ret = -100.0 + (2.0 * x) + (3.0 * y) + (0.2 * y * y) + (0.1 * x * y) + (0.2 * Math.sqrt(Math.abs(x)));
  ret += ((20.0 * Math.sin(6.0 * x * Math.PI)) + (20.0 * Math.sin(2.0 * x * Math.PI))) * (2.0 / 3.0);
  ret += ((20.0 * Math.sin(y * Math.PI)) + (40.0 * Math.sin((y / 3.0) * Math.PI))) * (2.0 / 3.0);
  ret += ((160.0 * Math.sin((y / 12.0) * Math.PI)) + (320 * Math.sin((y * Math.PI) / 30.0))) * (2.0 / 3.0);
  return ret;
}

function transformLng(x, y) {
  let ret = 300.0 + x + (2.0 * y) + (0.1 * x * x) + (0.1 * x * y) + (0.1 * Math.sqrt(Math.abs(x)));
  ret += ((20.0 * Math.sin(6.0 * x * Math.PI)) + (20.0 * Math.sin(2.0 * x * Math.PI))) * (2.0 / 3.0);
  ret += ((20.0 * Math.sin(x * Math.PI)) + (40.0 * Math.sin((x / 3.0) * Math.PI))) * (2.0 / 3.0);
  ret += ((150.0 * Math.sin((x / 12.0) * Math.PI)) + (300.0 * Math.sin((x / 30.0) * Math.PI))) * (2.0 / 3.0);
  return ret;
}

function wgs84ToGcj02(lng, lat) {
  const lon = Number(lng);
  const latitude = Number(lat);
  if (!Number.isFinite(lon) || !Number.isFinite(latitude) || outOfChina(lon, latitude)) {
    return [lon, latitude];
  }
  const a = 6378245.0;
  const ee = 0.00669342162296594323;
  let dLat = transformLat(lon - 105.0, latitude - 35.0);
  let dLng = transformLng(lon - 105.0, latitude - 35.0);
  const radLat = (latitude / 180.0) * Math.PI;
  let magic = Math.sin(radLat);
  magic = 1 - (ee * magic * magic);
  const sqrtMagic = Math.sqrt(magic);
  dLat = (dLat * 180.0) / (((a * (1 - ee)) / (magic * sqrtMagic)) * Math.PI);
  dLng = (dLng * 180.0) / ((a / sqrtMagic) * Math.cos(radLat) * Math.PI);
  return [lon + dLng, latitude + dLat];
}

function gcj02ToWgs84(lng, lat) {
  const lon = Number(lng);
  const latitude = Number(lat);
  if (!Number.isFinite(lon) || !Number.isFinite(latitude) || outOfChina(lon, latitude)) {
    return [lon, latitude];
  }
  const [mgLng, mgLat] = wgs84ToGcj02(lon, latitude);
  return [lon * 2 - mgLng, latitude * 2 - mgLat];
}

function toLatLng(p, mapProvider = "") {
  if (!Array.isArray(p) || p.length < 2) return null;
  let lng = Number(p[0]);
  let lat = Number(p[1]);
  if (isTencentProvider(mapProvider)) {
    [lng, lat] = wgs84ToGcj02(lng, lat);
  }
  return [lat, lng];
}

function toLngLat(p, mapProvider = "") {
  if (!Array.isArray(p) || p.length < 2) return null;
  let lng = Number(p[1]);
  let lat = Number(p[0]);
  if (isTencentProvider(mapProvider)) {
    [lng, lat] = gcj02ToWgs84(lng, lat);
  }
  return [lng, lat];
}

function metersToDegLat(m) {
  return m / 111320.0;
}

function metersToDegLng(m, latDeg) {
  const c = Math.cos((latDeg * Math.PI) / 180.0);
  return m / (111320.0 * Math.max(0.1, Math.abs(c)));
}

function llDistanceMeters(a, b) {
  if (!a || !b || a.length < 2 || b.length < 2) return 0;
  const lat1 = Number(a[0]);
  const lng1 = Number(a[1]);
  const lat2 = Number(b[0]);
  const lng2 = Number(b[1]);
  if (![lat1, lng1, lat2, lng2].every(Number.isFinite)) return 0;
  const dLat = (lat2 - lat1) * 111320.0;
  const dLng = (lng2 - lng1) * 111320.0 * Math.max(0.1, Math.abs(Math.cos(((lat1 + lat2) * 0.5 * Math.PI) / 180.0)));
  return Math.hypot(dLat, dLng);
}

function polylineLengthMeters(points) {
  if (!Array.isArray(points) || points.length < 2) return 0;
  let total = 0;
  for (let i = 1; i < points.length; i += 1) {
    total += llDistanceMeters(points[i - 1], points[i]);
  }
  return total;
}

function polygonAreaMeters(points) {
  if (!Array.isArray(points) || points.length < 3) return 0;
  const origin = points[0];
  const lat0 = Number(origin[0]);
  const lng0 = Number(origin[1]);
  if (!Number.isFinite(lat0) || !Number.isFinite(lng0)) return 0;
  const xy = points.map((p) => {
    const lat = Number(p[0]);
    const lng = Number(p[1]);
    return [
      (lng - lng0) * 111320.0 * Math.max(0.1, Math.abs(Math.cos((lat0 * Math.PI) / 180.0))),
      (lat - lat0) * 111320.0,
    ];
  });
  let area = 0;
  for (let i = 0; i < xy.length; i += 1) {
    const [x1, y1] = xy[i];
    const [x2, y2] = xy[(i + 1) % xy.length];
    area += (x1 * y2) - (x2 * y1);
  }
  return Math.abs(area * 0.5);
}

function ccw(a, b, c) {
  return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0]);
}

function segmentsIntersect(a, b, c, d) {
  return ccw(a, c, d) !== ccw(b, c, d) && ccw(a, b, c) !== ccw(a, b, d);
}

function polygonHasSelfIntersection(points) {
  if (!Array.isArray(points) || points.length < 4) return false;
  const segs = [];
  for (let i = 0; i < points.length; i += 1) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    segs.push([a, b, i]);
  }
  for (let i = 0; i < segs.length; i += 1) {
    const [a1, a2, idxA] = segs[i];
    for (let j = i + 1; j < segs.length; j += 1) {
      const [b1, b2, idxB] = segs[j];
      if (Math.abs(idxA - idxB) <= 1) continue;
      if ((idxA === 0 && idxB === segs.length - 1) || (idxB === 0 && idxA === segs.length - 1)) continue;
      if (segmentsIntersect(a1, a2, b1, b2)) return true;
    }
  }
  return false;
}

function createDrawVertexIcon(index, opts = {}) {
  const active = Boolean(opts.active);
  const hover = Boolean(opts.hover);
  const first = Boolean(opts.first);
  const last = Boolean(opts.last);
  const cls = ["draw-vertex", active ? "active" : "", hover ? "hover" : "", first ? "first" : "", last ? "last" : ""].filter(Boolean).join(" ");
  const label = Number(index) + 1;
  return L.divIcon({
    className: "draw-vertex-icon-wrap",
    html: `<span class="${cls}">${label}</span>`,
    iconSize: [30, 30],
    iconAnchor: [15, 15],
  });
}

function fmtNum(v, digits = 1, suffix = "") {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "--";
  return `${Number(v).toFixed(digits)}${suffix}`;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function toHex(v) {
  return Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, "0");
}

function mixColor(fromRgb, toRgb, t) {
  return `#${toHex(lerp(fromRgb[0], toRgb[0], t))}${toHex(lerp(fromRgb[1], toRgb[1], t))}${toHex(lerp(fromRgb[2], toRgb[2], t))}`;
}

function coverageColorForCount(count) {
  const green = [22, 163, 74];
  const orange = [245, 158, 11];
  const red = [220, 38, 38];
  if (count <= 2) return mixColor(green, orange, 0);
  if (count <= 6) return mixColor(green, orange, (count - 2) / 4);
  return mixColor(orange, red, Math.min(1, (count - 6) / 6));
}

function sanitizeOrbitLayers(layers, fallbackAltitude = 10, fallbackLaps = 1) {
  if (!Array.isArray(layers) || !layers.length) {
    return [{ altitude_m: Number(fallbackAltitude), laps: Math.max(1, Math.round(Number(fallbackLaps) || 1)) }];
  }
  return layers.map((layer) => ({
    altitude_m: Number.isFinite(Number(layer?.altitude_m)) ? Number(layer.altitude_m) : Number(fallbackAltitude),
    laps: Math.max(1, Math.round(Number(layer?.laps) || 1)),
  }));
}

function missionTypeLabel(missionType) {
  return missionType === "orbit_scan" ? "Orbit" : "Area";
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

function MissionMapEvents({ drawModeRef, startModeRef, orbitModeRef, onDrawPoint, onSetStart, onSetOrbitCenter, onFinishDraw, onDrawMove }) {
  useMapEvents({
    mousedown(e) {
      const active = Boolean(drawModeRef.current || startModeRef.current || orbitModeRef.current);
      if (!active) return;
      if (e.originalEvent?.preventDefault) e.originalEvent.preventDefault();
      if (e.originalEvent?.stopPropagation) e.originalEvent.stopPropagation();
    },
    click(e) {
      if (drawModeRef.current) {
        onDrawPoint([e.latlng.lat, e.latlng.lng]);
      } else if (startModeRef.current) {
        onSetStart([e.latlng.lat, e.latlng.lng]);
      } else if (orbitModeRef.current) {
        onSetOrbitCenter([e.latlng.lat, e.latlng.lng]);
      }
    },
    dblclick() {
      if (drawModeRef.current) {
        onFinishDraw();
      }
    },
    mousemove(e) {
      if (!drawModeRef.current) return;
      onDrawMove([e.latlng.lat, e.latlng.lng]);
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
  onMissionStateChange = null,
  variant = "sim",
  onPlanningStateChange = null,
}) {
  const isSim = variant === "sim";
  const isReal = variant === "real";
  const readinessPath = isReal ? "/api/real/readiness" : "/api/sim/readiness";
  const mapStatePath = isReal ? "/api/map_state" : "/api/sim/map_state";
  const missionPathPath = isReal ? "/api/real/mission/path" : "/api/sim/mission/path";
  const missionAreaPath = isReal ? "/api/mission/area" : "/api/sim/mission/area";
  const missionStartPath = isReal ? "/api/mission/start_position" : "/api/sim/mission/start_position";
  const missionOrbitCenterPath = isReal ? "/api/mission/orbit_center" : "/api/sim/mission/orbit_center";
  const missionGenerateOrbitPath = isReal ? "/api/mission/generate_orbit_scan" : "/api/sim/mission/generate_orbit_scan";
  const missionGenerateScanPath = isReal ? "/api/mission/generate_scan" : "/api/sim/mission/generate_scan";
  const missionClearPath = isReal ? "/api/mission/clear" : "/api/sim/mission/clear";

  const [mapState, setMapState] = useState(null);
  const [track, setTrack] = useState([]);
  const [coverage, setCoverage] = useState(null);
  const [scanDebug, setScanDebug] = useState(null);

  const [missionArea, setMissionArea] = useState([]);
  const [missionType, setMissionType] = useState("ground_scan");
  const [orbitCenter, setOrbitCenter] = useState(null);
  const [missionStart, setMissionStart] = useState(null);
  const [missionPath, setMissionPath] = useState([]);
  const [missionPreview, setMissionPreview] = useState(null);
  const [missionConfig, setMissionConfig] = useState(null);
  const [drawDraft, setDrawDraft] = useState([]);
  const [missionMode, setMissionMode] = useState("none");
  const [missionMsg, setMissionMsg] = useState("");
  const [scanSpacingM, setScanSpacingM] = useState(8.0);
  const [scanSpeedMps, setScanSpeedMps] = useState(3.0);
  const [autoSpacing, setAutoSpacing] = useState(true);
  const [orbitRadiusM, setOrbitRadiusM] = useState(12.0);
  const [orbitAltitudeM, setOrbitAltitudeM] = useState(10.0);
  const [orbitLaps, setOrbitLaps] = useState(1);
  const [orbitPointsPerLap, setOrbitPointsPerLap] = useState(24);
  const [orbitClockwise, setOrbitClockwise] = useState(true);
  const [orbitYawToCenter, setOrbitYawToCenter] = useState(true);
  const [orbitLayers, setOrbitLayers] = useState([{ altitude_m: 10.0, laps: 1 }]);
  const [simState, setSimState] = useState({ sim_running: false, sim_paused: false, sim_done: false, pose: null });
  const [sitlState, setSitlState] = useState({
    state: "IDLE",
    scan_active: false,
    waypoint_index: 0,
    waypoint_count: 0,
    last_error: "",
  });
  const [selectedDraftIdx, setSelectedDraftIdx] = useState(null);
  const [hoverDraftIdx, setHoverDraftIdx] = useState(null);
  const [drawCursor, setDrawCursor] = useState(null);
  const [readiness, setReadiness] = useState({ can_autonomous: false, can_manual: false, blocking_reasons: [] });

  const drawModeRef = useRef(false);
  const startModeRef = useRef(false);
  const orbitModeRef = useRef(false);
  const simTickTimerRef = useRef(null);
  const missionSyncGuardRef = useRef({ area: false, start: false, orbit: false });
  const missionConfigDirtyRef = useRef({
    scanSpacingM: false,
    scanSpeedMps: false,
    autoSpacing: false,
    orbitRadiusM: false,
    orbitAltitudeM: false,
    orbitLaps: false,
    orbitPointsPerLap: false,
    orbitClockwise: false,
    orbitYawToCenter: false,
    orbitLayers: false,
  });

  const [showBounds, setShowBounds] = useState(false);
  const [showPlannedPath, setShowPlannedPath] = useState(true);
  const [showTrack, setShowTrack] = useState(isSim);
  const [showCoverage, setShowCoverage] = useState(isSim);
  const [showBreadcrumbs, setShowBreadcrumbs] = useState(false);
  const [showStartPoint, setShowStartPoint] = useState(true);
  const [showCurrentTarget, setShowCurrentTarget] = useState(isSim);
  const [showDebug, setShowDebug] = useState(isSim);

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
      throw new Error(detail);
    }
    return payload;
  }

  useEffect(() => {
    let cancelled = false;
    let timer = null;

    async function pollReadiness() {
      try {
  const payload = await fetchJson(readinessPath);
        if (!cancelled) {
          setReadiness({
            can_autonomous: Boolean(payload?.can_autonomous),
            can_manual: Boolean(payload?.can_manual),
            blocking_reasons: Array.isArray(payload?.blocking_reasons) ? payload.blocking_reasons : [],
          });
        }
      } catch (_) {
        if (!cancelled) {
          setReadiness({ can_autonomous: false, can_manual: false, blocking_reasons: ["readiness_unavailable"] });
        }
      }
      if (!cancelled) timer = window.setTimeout(pollReadiness, 1500);
    }

    pollReadiness();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [readinessPath]);

  function stopTickLoop() {
    if (simTickTimerRef.current) {
      window.clearInterval(simTickTimerRef.current);
      simTickTimerRef.current = null;
    }
  }

  function applyMissionPayload(payload, mapProvider = mapState?.map_provider) {
    const area = Array.isArray(payload?.scan_area_polygon_lng_lat)
      ? payload.scan_area_polygon_lng_lat.map((p) => toLatLng(p, mapProvider)).filter(Boolean)
      : [];
    const orbitRaw = payload?.orbit_center_lng_lat;
    const orbit = Array.isArray(orbitRaw) && orbitRaw.length >= 2
      ? toLatLng(orbitRaw, mapProvider)
      : null;
    const startRaw = payload?.start_position_lng_lat;
    const start = Array.isArray(startRaw) && startRaw.length >= 2
      ? toLatLng(startRaw, mapProvider)
      : null;
    const path = Array.isArray(payload?.waypoints_lng_lat)
      ? payload.waypoints_lng_lat.map((p) => toLatLng(p, mapProvider)).filter(Boolean)
      : [];
    const payloadMissionType = String(payload?.mission_type || "").trim();
    const hasBackendMissionShape = area.length > 0 || Boolean(orbit) || path.length > 0;
    if (payloadMissionType && hasBackendMissionShape) {
      setMissionType(payloadMissionType);
    }
    if (area.length > 0 || !missionSyncGuardRef.current.area) {
      setMissionArea(area);
    }
    if (orbit || !missionSyncGuardRef.current.orbit) {
      setOrbitCenter(orbit);
    }
    if (start || !missionSyncGuardRef.current.start) {
      setMissionStart(start);
    }
    setMissionPath(path);
    setMissionPreview(payload?.coverage_preview || null);
    setMissionConfig(payload?.config || null);
    if (hasBackendMissionShape && Number.isFinite(Number(payload?.config?.spacing_m)) && !missionConfigDirtyRef.current.scanSpacingM) {
      setScanSpacingM(Number(payload.config.spacing_m));
    }
    if (hasBackendMissionShape && Number.isFinite(Number(payload?.config?.speed_m_s)) && !missionConfigDirtyRef.current.scanSpeedMps) {
      setScanSpeedMps(Number(payload.config.speed_m_s));
    }
    if (hasBackendMissionShape && Number.isFinite(Number(payload?.config?.radius_m)) && !missionConfigDirtyRef.current.orbitRadiusM) {
      setOrbitRadiusM(Number(payload.config.radius_m));
    }
    if (hasBackendMissionShape && Number.isFinite(Number(payload?.config?.altitude_m)) && !missionConfigDirtyRef.current.orbitAltitudeM) {
      setOrbitAltitudeM(Number(payload.config.altitude_m));
    }
    if (hasBackendMissionShape && Number.isFinite(Number(payload?.config?.laps)) && !missionConfigDirtyRef.current.orbitLaps) {
      setOrbitLaps(Number(payload.config.laps));
    }
    if (hasBackendMissionShape && Number.isFinite(Number(payload?.config?.points_per_lap)) && !missionConfigDirtyRef.current.orbitPointsPerLap) {
      setOrbitPointsPerLap(Number(payload.config.points_per_lap));
    }
    if (hasBackendMissionShape && typeof payload?.config?.clockwise === "boolean" && !missionConfigDirtyRef.current.orbitClockwise) {
      setOrbitClockwise(Boolean(payload.config.clockwise));
    }
    if (hasBackendMissionShape && typeof payload?.config?.yaw_to_center === "boolean" && !missionConfigDirtyRef.current.orbitYawToCenter) {
      setOrbitYawToCenter(Boolean(payload.config.yaw_to_center));
    }
    if (hasBackendMissionShape && Array.isArray(payload?.config?.layers) && !missionConfigDirtyRef.current.orbitLayers) {
      setOrbitLayers(sanitizeOrbitLayers(payload.config.layers, payload?.config?.altitude_m, payload?.config?.laps));
    }
    setSimState((prev) => ({
      sim_running: Boolean(payload?.sim_running ?? prev.sim_running),
      sim_paused: Boolean(payload?.sim_paused ?? prev.sim_paused),
      sim_done: Boolean(payload?.sim_done ?? prev.sim_done),
      pose: payload?.sim || prev.pose,
    }));
  }

  async function refreshMissionState() {
    try {
      const baseRequests = [
        fetchJson(mapStatePath),
        fetchJson(missionPathPath),
      ];
      const simRequests = isSim
        ? [fetchJson("/api/sim/mission/sim/state"), fetchJson("/api/sim/sitl/state")]
        : [Promise.resolve({}), Promise.resolve({})];
      const [s, mp, sim, sitl] = await Promise.all([...baseRequests, ...simRequests]);
      setMapState(s || null);
      applyMissionPayload(mp || {}, s?.map_provider);
      if (isSim) {
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
      }
    } catch (_) {
      return;
    }
  }

  const applyTrackPayload = useCallback((payload) => {
    const items = Array.isArray(payload?.items) ? payload.items : [];
    setTrack((prev) => {
      if (payload?.reset) {
        return items;
      }
      if (!items.length) {
        return prev;
      }
      return [...prev, ...items].slice(-600);
    });
  }, []);

  const applyCoveragePayload = useCallback((payload) => {
    const cov = payload?.coverage || null;
    const debug = payload?.scan_debug || null;
    setCoverage(cov);
    setScanDebug(debug);
    if (onCoverageUpdate) onCoverageUpdate(cov?.stats || null);
  }, [onCoverageUpdate]);

  const applyBridgePayload = useCallback((payload) => {
    if (payload?.map_state) setMapState(payload.map_state);
    if (payload?.mission_path) applyMissionPayload(payload.mission_path, payload?.map_state?.map_provider || mapState?.map_provider);
    if (payload?.mission_sim) {
      setSimState({
        sim_running: Boolean(payload.mission_sim?.sim_running),
        sim_paused: Boolean(payload.mission_sim?.sim_paused),
        sim_done: Boolean(payload.mission_sim?.sim_done),
        pose: payload.mission_sim?.pose || null,
      });
    }
    if (payload?.sitl_state) {
      setSitlState({
        state: String(payload.sitl_state?.state || "IDLE"),
        scan_active: Boolean(payload.sitl_state?.scan_active),
        waypoint_index: Number(payload.sitl_state?.waypoint_index || 0),
        waypoint_count: Number(payload.sitl_state?.waypoint_count || 0),
        last_error: String(payload.sitl_state?.last_error || ""),
      });
    }
  }, []);

  const bridgeStream = useLiveStream("/api/sim/stream/bridge_state", {
    event: "bridge_state",
    onMessage: applyBridgePayload,
    enabled: isSim,
  });

  const coverageStream = useLiveStream("/api/sim/stream/coverage", {
    event: "coverage",
    onMessage: applyCoveragePayload,
    resetKey: String(coverageVersion),
    enabled: isSim,
  });

  const trackStream = useLiveStream("/api/sim/stream/track?limit=600", {
    event: "track",
    onMessage: applyTrackPayload,
    enabled: isSim,
  });

  function startTickLoop() {
    stopTickLoop();
    simTickTimerRef.current = window.setInterval(async () => {
      try {
  const tick = await fetchJson("/api/sim/mission/sim/tick", {
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

  useEffect(() => {
    refreshMissionState();
    return () => {
      stopTickLoop();
    };
  }, []);

  useEffect(() => {
    drawModeRef.current = missionMode === "draw";
    startModeRef.current = missionMode === "start";
    orbitModeRef.current = missionMode === "orbit_center";
  }, [missionMode]);

  const center = useMemo(() => {
    if (mapState?.center_lng_lat) {
      const p = toLatLng(mapState.center_lng_lat, mapState?.map_provider);
      if (p) return p;
    }
    return [39.90923, 116.397428];
  }, [mapState]);

  const boundsPolygon = useMemo(() => {
    if (!Array.isArray(mapState?.bounds_polygon_lng_lat)) return [];
    return mapState.bounds_polygon_lng_lat.map((p) => toLatLng(p, mapState?.map_provider)).filter(Boolean);
  }, [mapState]);

  const hasExplicitFencePolygon = useMemo(() => {
    if (!isReal) return false;
    if (!Boolean(mapState?.operating_fence?.configured)) return false;
    return String(mapState?.operating_fence?.source || "").toLowerCase() === "polygon";
  }, [isReal, mapState?.operating_fence?.configured, mapState?.operating_fence?.source]);

  const fencePolygon = useMemo(() => {
    if (!hasExplicitFencePolygon) return [];
    if (!Array.isArray(mapState?.operating_fence?.polygon_lng_lat)) return [];
    return mapState.operating_fence.polygon_lng_lat.map((p) => toLatLng(p, mapState?.map_provider)).filter(Boolean);
  }, [hasExplicitFencePolygon, mapState]);

  const trail = useMemo(
    () =>
      track
        .map((p) => toLatLng([Number(p.lon), Number(p.lat)], mapState?.map_provider))
        .filter(Boolean),
    [mapState?.map_provider, track],
  );

  const coverageRects = useMemo(() => {
    const cells = Array.isArray(coverage?.covered_cells) ? coverage.covered_cells : [];
    return cells.map((c) => ({
      bounds: (() => {
        const a = toLatLng([Number(c.lng_min), Number(c.lat_min)], mapState?.map_provider);
        const b = toLatLng([Number(c.lng_max), Number(c.lat_max)], mapState?.map_provider);
        return [
          [Math.min(a?.[0] ?? 0, b?.[0] ?? 0), Math.min(a?.[1] ?? 0, b?.[1] ?? 0)],
          [Math.max(a?.[0] ?? 0, b?.[0] ?? 0), Math.max(a?.[1] ?? 0, b?.[1] ?? 0)],
        ];
      })(),
      count: Number(c.count || 0),
      color: coverageColorForCount(Number(c.count || 0)),
    }));
  }, [coverage, mapState?.map_provider]);

  const footprint = useMemo(() => {
    const pts = Array.isArray(scanDebug?.footprint_polygon_lng_lat)
      ? scanDebug.footprint_polygon_lng_lat.map((p) => toLatLng(p, mapState?.map_provider)).filter(Boolean)
      : [];
    return pts.length > 2 ? pts : null;
  }, [mapState?.map_provider, scanDebug]);

  const origin = useMemo(() => {
    if (!mapState?.origin) return null;
    return toLatLng([Number(mapState.origin.lng), Number(mapState.origin.lat)], mapState?.map_provider);
  }, [mapState]);

  const vehicle = useMemo(() => {
    const src = mapState?.vehicle || {};
    const simPose = simState?.pose || null;
    const lat = Number(telemetry?.lat ?? src?.lat ?? simPose?.lat);
    const lng = Number(telemetry?.lon ?? src?.lng ?? simPose?.lng);
    const yaw = Number(telemetry?.yaw_deg ?? src?.yaw_deg ?? simPose?.yaw_deg);
    if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null;
    const display = toLatLng([lng, lat], mapState?.map_provider);
    if (!display) return null;
    return { lat: display[0], lng: display[1], yaw: Number.isFinite(yaw) ? yaw : null };
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

  const breadcrumbs = useMemo(() => {
    if (trail.length < 3) return [];
    const step = Math.max(1, Math.floor(trail.length / 20));
    return trail.filter((_, idx) => idx % step === 0 || idx === trail.length - 1);
  }, [trail]);

  const hasPath = missionPath.length >= 2;
  const interactionLocked = missionMode === "draw" || missionMode === "start" || missionMode === "orbit_center";
  const simStatusLabel = simState.sim_running
    ? (simState.sim_paused ? "PAUSED" : "RUNNING")
    : "STOPPED";

  const DRAW_MIN_POINT_SPACING_M = 1.5;
  const DRAW_DUPLICATE_EPS_M = 0.35;

  const draftValidation = useMemo(() => {
    const points = drawDraft;
    const reasons = [];
    let tooCloseIdx = -1;
    for (let i = 0; i < points.length; i += 1) {
      for (let j = i + 1; j < points.length; j += 1) {
        const d = llDistanceMeters(points[i], points[j]);
        if (d < DRAW_DUPLICATE_EPS_M) {
          reasons.push(`Point ${i + 1} duplicates point ${j + 1}`);
        } else if (d < DRAW_MIN_POINT_SPACING_M) {
          reasons.push(`Points ${i + 1} and ${j + 1} are too close (${d.toFixed(2)}m)`);
          tooCloseIdx = i;
        }
      }
    }

    let selfIntersect = false;
    if (points.length >= 4) {
      selfIntersect = polygonHasSelfIntersection(points);
      if (selfIntersect) reasons.push("Shape self-intersects");
    }

    return {
      valid: reasons.length === 0,
      reasons,
      selfIntersect,
      tooCloseIdx,
      perimeterM: polylineLengthMeters(points) + (points.length > 2 ? llDistanceMeters(points[points.length - 1], points[0]) : 0),
      routeLenM: polylineLengthMeters(points),
      areaM2: points.length > 2 ? polygonAreaMeters(points) : 0,
      canFinish: points.length >= 3 && reasons.length === 0,
    };
  }, [drawDraft]);

  const missionAreaIsInvalid = useMemo(() => {
    if (missionType !== "ground_scan") return false;
    if (!Array.isArray(missionArea) || missionArea.length < 3) return false;
    return polygonHasSelfIntersection(missionArea);
  }, [missionArea, missionType]);

  const geometryInvalid = missionAreaIsInvalid || (missionMode === "draw" && !draftValidation.valid);
  const missionActive = Boolean(sitlState.scan_active || simState.sim_running);
  const currentTargetOrdinal = missionActive && sitlState.waypoint_count
    ? Math.min(sitlState.waypoint_index + 1, sitlState.waypoint_count)
    : null;
  const currentTarget = useMemo(() => {
    if (!hasPath || !missionPath.length) return null;
    const idx = Math.max(0, Math.min(sitlState.waypoint_index || 0, missionPath.length - 1));
    return missionPath[idx] || null;
  }, [hasPath, missionPath, sitlState.waypoint_index]);
  const orbitReady = Boolean(orbitCenter && missionStart);
  const validOrbitLayers = useMemo(
    () => sanitizeOrbitLayers(orbitLayers, orbitAltitudeM, orbitLaps).filter((layer) => Number(layer.altitude_m) > 0 && Number(layer.laps) > 0),
    [orbitAltitudeM, orbitLaps, orbitLayers],
  );

  useEffect(() => {
    if (!onMissionStateChange) return;
    onMissionStateChange({
      mode: missionType,
      hasMission: hasPath,
      targetIndex: currentTargetOrdinal,
      targetCount: sitlState.waypoint_count || missionPath.length,
      coverageActive: Boolean(sitlState.scan_active),
      debugAvailable: Boolean(footprint),
      vehicleVisible: Boolean(vehicle),
      simStatus: simStatusLabel,
      executorState: sitlState.state,
      missionMessage: missionMsg,
    });
  }, [currentTargetOrdinal, footprint, hasPath, missionMsg, missionPath.length, missionType, onMissionStateChange, simStatusLabel, sitlState.scan_active, sitlState.state, sitlState.waypoint_count, vehicle]);

  useEffect(() => {
    if (!onPlanningStateChange) return;
    const routeLengthM = polylineLengthMeters(missionPath);
    const areaM2 = missionArea.length >= 3 ? polygonAreaMeters(missionArea) : 0;
    const perimeterM = missionArea.length >= 3
      ? (polylineLengthMeters(missionArea) + llDistanceMeters(missionArea[missionArea.length - 1], missionArea[0]))
      : 0;
    const fenceConfigured = Boolean(mapState?.operating_fence?.configured);
    const missionInsideFence = hasPath && Boolean(missionConfig);
    const geometryValid = !geometryInvalid && (missionType === "orbit_scan" ? orbitReady : missionArea.length >= 3);
    const validForMissionAction = Boolean(geometryValid && hasPath && readiness.can_autonomous && fenceConfigured && missionInsideFence);
    onPlanningStateChange({
      geometryValid,
      hasPath,
      missionType,
      routeLengthM,
      areaM2,
      perimeterM,
      fenceConfigured,
      missionInsideFence,
      readinessCanAutonomous: Boolean(readiness.can_autonomous),
      validForMissionAction,
    });
  }, [geometryInvalid, hasPath, mapState?.operating_fence?.configured, missionArea, missionConfig, missionPath, missionType, onPlanningStateChange, orbitReady, readiness.can_autonomous]);

  async function saveAreaToBackend(pointsLatLng) {
    if (!Array.isArray(pointsLatLng) || pointsLatLng.length < 3) {
      throw new Error("Need at least 3 points");
    }
    if (polygonHasSelfIntersection(pointsLatLng)) {
      throw new Error("Shape is invalid: self-intersection detected");
    }
    const polygonLngLat = pointsLatLng.map((p) => toLngLat(p, mapState?.map_provider)).filter(Boolean);
  const payload = await fetchJson(missionAreaPath, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ polygon_lng_lat: polygonLngLat }),
    });
    setMissionMsg(`Area saved (${payload.points} points)`);
  }

  async function saveStartToBackend(pointLatLng) {
  await fetchJson(missionStartPath, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify((() => {
        const ll = toLngLat(pointLatLng, mapState?.map_provider);
        return { lng: ll?.[0], lat: ll?.[1] };
      })()),
    });
    setMissionMsg("Start position saved");
  }

  async function saveOrbitCenterToBackend(pointLatLng) {
  await fetchJson(missionOrbitCenterPath, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify((() => {
        const ll = toLngLat(pointLatLng, mapState?.map_provider);
        return { lng: ll?.[0], lat: ll?.[1] };
      })()),
    });
    setMissionMsg("Object center saved");
  }

  async function runMissionSyncGuard(kind, work) {
    missionSyncGuardRef.current[kind] = true;
    try {
      await work();
      await refreshMissionState();
    } finally {
      missionSyncGuardRef.current[kind] = false;
    }
  }

  async function handleGeneratePath() {
    if (geometryInvalid) {
      setMissionMsg("Cannot generate mission: geometry is invalid");
      return;
    }
    try {
      Object.keys(missionConfigDirtyRef.current).forEach((key) => {
        missionConfigDirtyRef.current[key] = false;
      });
      const payload = missionType === "orbit_scan"
  ? await fetchJson(missionGenerateOrbitPath, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              radius_m: Number(orbitRadiusM),
              altitude_m: Number(orbitAltitudeM),
              laps: Number(orbitLaps),
              layers: validOrbitLayers,
              points_per_lap: Number(orbitPointsPerLap),
              clockwise: Boolean(orbitClockwise),
              yaw_to_center: Boolean(orbitYawToCenter),
              speed_m_s: Number(scanSpeedMps),
              start_scan: false,
            }),
          })
  : await fetchJson(missionGenerateScanPath, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              spacing_m: Number(scanSpacingM),
              speed_m_s: Number(scanSpeedMps),
              start_scan: false,
              auto_spacing: Boolean(autoSpacing),
            }),
          });
      applyMissionPayload(payload, mapState?.map_provider);
      setMissionMsg(
        missionType === "orbit_scan"
          ? "Orbit scan generated"
          : (autoSpacing ? "Scan path generated with automatic spacing" : "Scan path generated")
      );
    } catch (err) {
      setMissionMsg(`Generate failed: ${String(err)}`);
    }
  }

  async function handleSimStart() {
    try {
  const state = await fetchJson("/api/sim/mission/sim/start", {
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
  const state = await fetchJson("/api/sim/sitl/start_scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ alt_m: Number(missionConfig?.first_altitude_m || missionConfig?.altitude_m || 10.0), accept_radius_m: 3.0 }),
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
    if (!window.confirm("Stop the active SITL scan?")) return;
    try {
  const state = await fetchJson("/api/sim/sitl/stop_scan", {
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
  const state = await fetchJson("/api/sim/mission/sim/pause", {
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
    if (!window.confirm("Stop the current simulation?")) return;
    try {
  const state = await fetchJson("/api/sim/mission/sim/stop", {
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
    if (!window.confirm("Clear the current mission geometry and path?")) return;
    try {
  await fetchJson(missionClearPath, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      stopTickLoop();
      setMissionArea([]);
      setOrbitCenter(null);
      setMissionType("ground_scan");
      setMissionStart(null);
      setMissionPath([]);
      setMissionPreview(null);
      setMissionConfig(null);
      setDrawDraft([]);
      setMissionMode("none");
      setSimState({ sim_running: false, sim_paused: false, sim_done: false, pose: null });
      setSitlState({
        state: "IDLE",
        scan_active: false,
        waypoint_index: 0,
        waypoint_count: 0,
        last_error: "",
      });
      setCoverage(null);
      setScanDebug(null);
      setTrack([]);
      if (onCoverageUpdate) onCoverageUpdate(null);
      await refreshMissionState();
      setMissionMsg("Mission cleared");
    } catch (err) {
      setMissionMsg(`Clear failed: ${String(err)}`);
    }
  }

  function handleDrawClick(p) {
    if (missionMode !== "draw") return;
    setDrawDraft((prev) => {
      for (let i = 0; i < prev.length; i += 1) {
        const d = llDistanceMeters(prev[i], p);
        if (d < DRAW_DUPLICATE_EPS_M) {
          setMissionMsg(`Point too close to existing point ${i + 1} (duplicate)`);
          return prev;
        }
        if (d < DRAW_MIN_POINT_SPACING_M) {
          setMissionMsg(`Point too close to point ${i + 1} (${d.toFixed(2)}m). Min spacing ${DRAW_MIN_POINT_SPACING_M}m.`);
          return prev;
        }
      }
      return [...prev, p];
    });
  }

  async function handleDrawFinish() {
    if (missionMode !== "draw") return;
    if (!draftValidation.canFinish) {
      const first = draftValidation.reasons?.[0];
      setMissionMsg(first ? `Cannot finish: ${first}` : "Need at least 3 points for scan area");
      return;
    }
    if (drawDraft.length < 3) {
      setMissionMsg("Need at least 3 points for scan area");
      return;
    }
    const area = [...drawDraft];
    setMissionArea(area);
    setDrawDraft([]);
    setSelectedDraftIdx(null);
    setHoverDraftIdx(null);
    setDrawCursor(null);
    setMissionMode("none");
    try {
      await runMissionSyncGuard("area", async () => {
        await saveAreaToBackend(area);
      });
    } catch (err) {
      setMissionMsg(`Save area failed: ${String(err)}`);
    }
  }

  async function handleSetStart(p) {
    if (missionMode !== "start") return;
    setMissionStart(p);
    setMissionMode("none");
    try {
      await runMissionSyncGuard("start", async () => {
        await saveStartToBackend(p);
      });
    } catch (err) {
      setMissionMsg(`Save start failed: ${String(err)}`);
    }
  }

  async function handleSetOrbitCenter(p) {
    if (missionMode !== "orbit_center") return;
    setOrbitCenter(p);
    setMissionMode("none");
    try {
      await runMissionSyncGuard("orbit", async () => {
        await saveOrbitCenterToBackend(p);
      });
    } catch (err) {
      setMissionMsg(`Save object center failed: ${String(err)}`);
    }
  }

  function updateOrbitLayer(index, field, value) {
    missionConfigDirtyRef.current.orbitLayers = true;
    setOrbitLayers((prev) => prev.map((layer, idx) => (
      idx === index
        ? {
            ...layer,
            [field]: field === "laps"
              ? Math.max(1, Math.round(Number(value) || 1))
              : Math.max(1, Number(value) || 1),
          }
        : layer
    )));
  }

  function addOrbitLayer() {
    missionConfigDirtyRef.current.orbitLayers = true;
    setOrbitLayers((prev) => {
      const last = prev[prev.length - 1];
      return [...prev, { altitude_m: Number(last?.altitude_m || orbitAltitudeM || 10), laps: Math.max(1, Math.round(Number(last?.laps || 1))) }];
    });
  }

  function removeOrbitLayer(index) {
    missionConfigDirtyRef.current.orbitLayers = true;
    setOrbitLayers((prev) => {
      if (prev.length <= 1) return prev;
      return prev.filter((_, idx) => idx !== index);
    });
  }

  const tiles = mapState?.tile_url_template
    ? `${BACKEND_BASE}${mapState.tile_url_template}`
    : "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
  const coverageEnabledForMission = missionType !== "orbit_scan";

  function handleDrawUndo() {
    if (missionMode !== "draw") return;
    setDrawDraft((prev) => prev.slice(0, -1));
    setSelectedDraftIdx((prev) => {
      if (prev === null || prev <= 0) return null;
      return prev - 1;
    });
  }

  function handleDrawClear() {
    if (missionMode !== "draw") return;
    setDrawDraft([]);
    setSelectedDraftIdx(null);
    setHoverDraftIdx(null);
    setDrawCursor(null);
    setMissionMsg("Draft cleared");
  }

  function handleDrawCancel() {
    if (missionMode !== "draw") return;
    setDrawDraft([]);
    setSelectedDraftIdx(null);
    setHoverDraftIdx(null);
    setDrawCursor(null);
    setMissionMode("none");
    setMissionMsg("Drawing cancelled");
  }

  function handleDeleteSelectedPoint() {
    if (missionMode !== "draw") return;
    if (selectedDraftIdx === null || selectedDraftIdx < 0) return;
    setDrawDraft((prev) => prev.filter((_, idx) => idx !== selectedDraftIdx));
    setMissionMsg(`Removed point ${selectedDraftIdx + 1}`);
    setSelectedDraftIdx(null);
  }

  function handleDraftPointDrag(index, latlng) {
    setDrawDraft((prev) => {
      const next = prev.map((p, idx) => (idx === index ? [Number(latlng.lat), Number(latlng.lng)] : p));
      for (let i = 0; i < next.length; i += 1) {
        if (i === index) continue;
        const d = llDistanceMeters(next[i], next[index]);
        if (d < DRAW_MIN_POINT_SPACING_M) {
          setMissionMsg(`Dragged point too close to point ${i + 1} (${d.toFixed(2)}m). Move farther apart.`);
          return prev;
        }
      }
      return next;
    });
  }
  const autonomyBlocked = !Boolean(readiness.can_autonomous);
  const autonomyBlockReason = readiness.blocking_reasons?.[0] || "readiness not green";

  const layerItems = [
    { key: "planned", label: "Planned path", checked: showPlannedPath, onChange: setShowPlannedPath },
    { key: "bounds", label: isReal ? "Fence" : "Bounds", checked: showBounds && (isReal ? hasExplicitFencePolygon : true), onChange: setShowBounds },
    { key: "start", label: "Start point", checked: showStartPoint, onChange: setShowStartPoint },
  ];
  if (isSim) {
    layerItems.push(
      { key: "track", label: "Actual track", checked: showTrack, onChange: setShowTrack },
      { key: "breadcrumbs", label: "Breadcrumbs", checked: showBreadcrumbs, onChange: setShowBreadcrumbs },
      { key: "target", label: "Current target", checked: showCurrentTarget, onChange: setShowCurrentTarget },
      { key: "debug", label: "Debug overlays", checked: showDebug, onChange: setShowDebug },
    );
  }
  if (isSim && coverageEnabledForMission) {
    layerItems.splice(2, 0, { key: "coverage", label: "Coverage overlay", checked: showCoverage, onChange: setShowCoverage });
  }

  const legendItems = [
    { label: "Mission area", swatchClass: "swatch-area" },
    { label: "Planned path", swatchClass: "swatch-path" },
    { label: "Actual track", swatchClass: "swatch-track" },
    { label: "Object center", swatchClass: "swatch-target" },
    { label: "Vehicle", swatchClass: "swatch-vehicle" },
    { label: "Current target", swatchClass: "swatch-target" },
    { label: "Start / origin", swatchClass: "swatch-start" },
  ];
  if (isSim && coverageEnabledForMission) {
    legendItems.splice(4, 0,
      { label: "Coverage: first pass", swatchClass: "swatch-coverage-low" },
      { label: "Coverage: overlap", swatchClass: "swatch-coverage-mid" },
      { label: "Coverage: heavy overlap", swatchClass: "swatch-coverage-high" },
    );
  }

  return (
    <>
      <div className="map-ops-header">
        <div className="map-toolbar compact-strip">
          {isSim ? <span className="chip"><strong>Bridge:</strong> {bridgeStream.connected ? "live" : "reconnecting"}</span> : null}
          {isSim && coverageEnabledForMission ? <span className="chip"><strong>Coverage:</strong> {coverageStream.connected ? "live" : "reconnecting"}</span> : null}
          {isSim ? <span className="chip"><strong>Track:</strong> {trackStream.connected ? "live" : "reconnecting"}</span> : null}
          {isSim ? <span className="chip"><strong>Executor:</strong> {sitlState.state}</span> : null}
          {isSim ? <span className="chip"><strong>Target:</strong> {currentTargetOrdinal || "--"} / {sitlState.waypoint_count || missionPath.length || "--"}</span> : null}
          <span className="chip"><strong>Mission Type:</strong> {missionTypeLabel(missionType)}</span>
          {isReal ? <span className="chip"><strong>Fence:</strong> {mapState?.operating_fence?.configured ? "configured" : "missing"}</span> : null}
          <span className="chip"><strong>Readiness:</strong> {readiness.can_autonomous ? "green" : (readiness.can_manual ? "manual only" : "blocked")}</span>
        </div>

        <div className="map-toolbar action-toolbar">
          <div className="map-action-group">
            <span className="group-title">Plan</span>
            <label className="map-input">Mission Type
              <select value={missionType} onChange={(e) => {
                const next = e.target.value;
                setMissionType(next);
                setMissionMode("none");
                setDrawDraft([]);
              }}>
                <option value="ground_scan">Area</option>
                <option value="orbit_scan">Orbit</option>
              </select>
            </label>
            {missionType === "ground_scan" ? (
              <button className="small-btn" onClick={() => {
                setMissionMode("draw");
                setDrawDraft([]);
                setSelectedDraftIdx(null);
                setHoverDraftIdx(null);
                setDrawCursor(null);
                setMissionMsg("Draw mode: click to add points, drag points to edit, then Finish Shape");
              }}>Draw Area</button>
            ) : (
              <button className="small-btn" onClick={() => {
                setMissionMode("orbit_center");
                setMissionMsg("Orbit center mode: click map to place the object center");
              }}>Set Object Center</button>
            )}
            <button className="small-btn" onClick={() => {
              setMissionMode("start");
              setMissionMsg("Start mode: click map to place drone start");
            }}>Set Start</button>
            <button
              className="small-btn"
              disabled={geometryInvalid || autonomyBlocked || simState.sim_running || sitlState.scan_active || (missionType === "orbit_scan" ? !orbitReady : false)}
              title={geometryInvalid ? "Geometry invalid: fix drawing before generating" : (autonomyBlocked ? `Autonomy blocked: ${autonomyBlockReason}` : "")}
              onClick={handleGeneratePath}
            >
              {missionType === "orbit_scan" ? "Generate Orbit" : "Generate Path"}
            </button>
            <button className="small-btn" onClick={handleClearMission}>Clear</button>
          </div>

          {isSim ? <div className="map-action-group run-group">
            <span className="group-title">Run</span>
            {mavConnected ? (
              <>
                <button className="small-btn" disabled={autonomyBlocked || !hasPath || sitlState.scan_active} title={autonomyBlocked ? `Autonomy blocked: ${autonomyBlockReason}` : ""} onClick={handleSitlStart}>Start SITL</button>
                <button className="small-btn" disabled={!sitlState.scan_active && sitlState.state !== "ARMING" && sitlState.state !== "TAKEOFF" && sitlState.state !== "RUN_PATH"} onClick={handleSitlStop}>Stop SITL</button>
              </>
            ) : (
              <>
                <button className="small-btn" disabled={autonomyBlocked || !hasPath || (simState.sim_running && !simState.sim_paused)} title={autonomyBlocked ? `Autonomy blocked: ${autonomyBlockReason}` : ""} onClick={handleSimStart}>Start Sim</button>
                <button className="small-btn" disabled={!simState.sim_running || simState.sim_paused} onClick={handleSimPause}>Pause</button>
                <button className="small-btn" disabled={!hasPath} onClick={handleSimStop}>Stop</button>
              </>
            )}
          </div> : null}

          <div className="map-action-group">
            <span className="group-title">Path Settings</span>
            {missionType === "ground_scan" ? (
              <>
                <label className="map-input">Spacing (m)
                  <input type="number" min="1" step="0.5" value={scanSpacingM} disabled={autoSpacing} onChange={(e) => {
                    missionConfigDirtyRef.current.scanSpacingM = true;
                    setScanSpacingM(e.target.value);
                  }} />
                </label>
                <label className="toggle-row compact-toggle">
                  <input type="checkbox" checked={autoSpacing} onChange={(e) => {
                    missionConfigDirtyRef.current.autoSpacing = true;
                    setAutoSpacing(e.target.checked);
                  }} />
                  <span>Auto spacing</span>
                </label>
              </>
            ) : (
              <>
                <label className="map-input">Radius (m)
                  <input type="number" min="1" step="0.5" value={orbitRadiusM} onChange={(e) => {
                    missionConfigDirtyRef.current.orbitRadiusM = true;
                    setOrbitRadiusM(e.target.value);
                  }} />
                </label>
                <label className="map-input">Altitude (m)
                  <input type="number" min="1" step="0.5" value={orbitAltitudeM} onChange={(e) => {
                    missionConfigDirtyRef.current.orbitAltitudeM = true;
                    const nextValue = e.target.value;
                    setOrbitAltitudeM(nextValue);
                    setOrbitLayers((prev) => (
                      prev.length === 1
                        ? [{ altitude_m: Math.max(1, Number(nextValue) || 1), laps: Math.max(1, Math.round(Number(prev[0]?.laps || orbitLaps || 1))) }]
                        : prev
                    ));
                  }} />
                </label>
                <label className="map-input">Laps
                  <input type="number" min="1" step="1" value={orbitLaps} onChange={(e) => {
                    missionConfigDirtyRef.current.orbitLaps = true;
                    const nextValue = e.target.value;
                    setOrbitLaps(nextValue);
                    setOrbitLayers((prev) => (
                      prev.length === 1
                        ? [{ altitude_m: Math.max(1, Number(prev[0]?.altitude_m || orbitAltitudeM || 10)), laps: Math.max(1, Math.round(Number(nextValue) || 1)) }]
                        : prev
                    ));
                  }} />
                </label>
                <details className="orbit-layer-editor">
                  <summary className="orbit-layer-head">
                    <span className="orbit-layer-title">Orbit Layers</span>
                    <span className="hint">advanced</span>
                  </summary>
                  <div className="collapsible-body">
                    <button type="button" className="small-btn" onClick={addOrbitLayer}>Add Layer</button>
                    {orbitLayers.map((layer, index) => (
                      <div key={`orbit-layer-${index}`} className="orbit-layer-row">
                        <label className="map-input">
                          <span>Altitude {index + 1} (m)</span>
                          <input
                            type="number"
                            min="1"
                            step="0.5"
                            value={layer.altitude_m}
                            onChange={(e) => updateOrbitLayer(index, "altitude_m", e.target.value)}
                          />
                        </label>
                        <label className="map-input">
                          <span>Laps</span>
                          <input
                            type="number"
                            min="1"
                            step="1"
                            value={layer.laps}
                            onChange={(e) => updateOrbitLayer(index, "laps", e.target.value)}
                          />
                        </label>
                        <button type="button" className="small-btn danger orbit-layer-remove" disabled={orbitLayers.length <= 1} onClick={() => removeOrbitLayer(index)}>
                          Remove
                        </button>
                      </div>
                    ))}
                  </div>
                </details>
                <label className="map-input">Pts/Lap
                  <input type="number" min="8" step="1" value={orbitPointsPerLap} onChange={(e) => {
                    missionConfigDirtyRef.current.orbitPointsPerLap = true;
                    setOrbitPointsPerLap(e.target.value);
                  }} />
                </label>
                <label className="toggle-row compact-toggle">
                  <input type="checkbox" checked={orbitClockwise} onChange={(e) => {
                    missionConfigDirtyRef.current.orbitClockwise = true;
                    setOrbitClockwise(e.target.checked);
                  }} />
                  <span>Clockwise</span>
                </label>
                <label className="toggle-row compact-toggle">
                  <input type="checkbox" checked={orbitYawToCenter} onChange={(e) => {
                    missionConfigDirtyRef.current.orbitYawToCenter = true;
                    setOrbitYawToCenter(e.target.checked);
                  }} />
                  <span>Yaw center</span>
                </label>
              </>
            )}
            <label className="map-input">Speed (m/s)
              <input type="number" min="0.2" step="0.2" value={scanSpeedMps} onChange={(e) => {
                missionConfigDirtyRef.current.scanSpeedMps = true;
                setScanSpeedMps(e.target.value);
              }} />
            </label>
          </div>
        </div>

        {missionMode === "draw" ? (
          <div className="map-toolbar draw-helper-toolbar">
            <span className="chip draw-instructions">Click map to add points • Drag point to adjust • Select point then Delete • Finish to close shape</span>
            <button className="small-btn draw-btn" disabled={drawDraft.length === 0} onClick={handleDrawUndo}>Undo</button>
            <button className="small-btn draw-btn" disabled={drawDraft.length === 0} onClick={handleDrawClear}>Clear</button>
            <button className="small-btn draw-btn danger" disabled={selectedDraftIdx === null} onClick={handleDeleteSelectedPoint}>Delete Selected</button>
            <button className="small-btn draw-btn" disabled={!draftValidation.canFinish} onClick={handleDrawFinish}>Finish Shape</button>
            <button className="small-btn draw-btn" onClick={handleDrawCancel}>Cancel</button>
            <span className="chip">Route: {fmtNum(draftValidation.routeLenM, 1, " m")}</span>
            <span className="chip">Perimeter: {fmtNum(draftValidation.perimeterM, 1, " m")}</span>
            <span className="chip">Area: {fmtNum(draftValidation.areaM2, 1, " m²")}</span>
          </div>
        ) : null}
      </div>

      <div className="mission-preview">
        <span>Mode: {missionTypeLabel(missionType)}</span>
        {isSim ? <span>Sim: {simStatusLabel}</span> : null}
        {isReal ? <span>Geometry: {geometryInvalid ? "invalid" : "valid"}</span> : null}
        {isReal ? <span>Area: {fmtNum(missionArea.length >= 3 ? polygonAreaMeters(missionArea) : 0, 1, " m²")}</span> : null}
        {isReal ? <span>Perimeter: {fmtNum(missionArea.length >= 3 ? (polylineLengthMeters(missionArea) + llDistanceMeters(missionArea[missionArea.length - 1], missionArea[0])) : 0, 1, " m")}</span> : null}
        {isReal ? <span>Route: {fmtNum(polylineLengthMeters(missionPath), 1, " m")}</span> : null}
        {isSim ? (
          <>
            {missionType === "orbit_scan"
              ? <span>Radius: {fmtNum(missionConfig?.radius_m, 1, " m")}</span>
              : <span>Spacing: {fmtNum(missionConfig?.spacing_m, 1, " m")}</span>}
            {missionType === "orbit_scan"
              ? <span>Altitude: {fmtNum(missionConfig?.altitude_m, 1, " m")}</span>
              : <span>Expected Coverage: {fmtNum(missionPreview?.expected_coverage_pct, 1, "%")}</span>}
            {missionType === "orbit_scan"
              ? <span>Laps: {missionConfig?.laps ?? "--"}</span>
              : <span>Overlap Est: {fmtNum(missionPreview?.overlap_pct_est, 1, "%")}</span>}
            {missionType === "orbit_scan" && Array.isArray(missionConfig?.layers) && missionConfig.layers.length > 1
              ? <span>Layers: {missionConfig.layers.map((layer) => `${layer.altitude_m}m x${layer.laps}`).join(", ")}</span>
              : null}
            {missionType === "orbit_scan"
              ? <span>Pts/Lap: {missionConfig?.points_per_lap ?? "--"}</span>
              : null}
            {missionType === "ground_scan" ? <span>Expected Coverage: {fmtNum(missionPreview?.expected_coverage_pct, 1, "%")}</span> : null}
            <span>Estimated Time: {fmtNum(missionPreview?.estimated_time_s, 1, " s")}</span>
            <span>Passes: {missionPreview?.number_of_passes ?? "--"}</span>
            <span>Path Length: {fmtNum(missionPreview?.path_length_m, 1, " m")}</span>
            {missionType === "ground_scan" ? <span>Sweep Angle: {fmtNum(missionPreview?.sweep_angle_deg, 1, " deg")}</span> : null}
            {missionType === "ground_scan" ? <span>Lead-In: {fmtNum(missionPreview?.lead_in_m, 1, " m")}</span> : null}
            {missionType === "ground_scan" ? <span>Return: {fmtNum(missionPreview?.return_to_home_m, 1, " m")}</span> : null}
          </>
        ) : null}
      </div>

      {missionMsg ? <p className="hint">{missionMsg}</p> : null}
      {missionMode === "draw" && !draftValidation.valid ? (
        <div className="map-banner danger">
          Invalid drawing: {draftValidation.reasons.join(" • ")}
        </div>
      ) : null}
      {missionAreaIsInvalid ? <div className="map-banner danger">Mission area is invalid (self-intersection). Edit and redraw area before generating.</div> : null}
      {!hasPath ? <div className="map-banner">{missionType === "orbit_scan" ? "No orbit mission loaded yet. Set an object center, set a start point, then generate an orbit." : "No mission loaded yet. Draw a scan area, set a start point, then generate a path."}</div> : null}
  {isSim && !mavConnected && !simState.sim_running ? <div className="map-banner subdued">No live vehicle connected. You can still prepare a mission and use local simulation.</div> : null}
  {isSim && coverageEnabledForMission && !(Array.isArray(coverage?.covered_cells) && coverage.covered_cells.length) && !sitlState.scan_active ? <div className="map-banner subdued">Coverage overlay is idle until scan motion begins.</div> : null}
  {isSim && sitlState.last_error ? <div className="map-banner danger">Mission executor error: {sitlState.last_error}</div> : null}

  <LayerToggles items={layerItems} legend={legendItems} />

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
          orbitModeRef={orbitModeRef}
          onDrawPoint={handleDrawClick}
          onSetStart={handleSetStart}
          onSetOrbitCenter={handleSetOrbitCenter}
          onFinishDraw={handleDrawFinish}
          onDrawMove={setDrawCursor}
        />
        <TileLayer url={tiles} maxZoom={20} tileSize={256} detectRetina={false} />

        {showBounds && (isReal ? fencePolygon.length > 2 : ((missionArea.length > 2 || hasPath) && boundsPolygon.length > 2)) ? (
          <Polygon positions={isReal ? fencePolygon : boundsPolygon} pathOptions={{ color: "#0f766e", weight: 2, fillOpacity: 0.08 }} />
        ) : null}

        {missionArea.length > 2 ? (
          <Polygon positions={missionArea} pathOptions={{ color: "#2563eb", weight: 3.5, fillColor: "#60a5fa", fillOpacity: 0.16 }} />
        ) : null}

        {missionType === "orbit_scan" && orbitCenter ? (
          <CircleMarker center={orbitCenter} radius={7} pathOptions={{ color: "#7c3aed", fillOpacity: 0.9 }}>
            <Tooltip permanent direction="top">Object Center</Tooltip>
          </CircleMarker>
        ) : null}

        {drawDraft.length > 1 ? (
          <Polyline positions={drawDraft} pathOptions={{ color: "#1d4ed8", weight: 3.5, dashArray: "6,6" }} />
        ) : null}

        {missionMode === "draw" && drawDraft.length > 2 ? (
          <Polygon positions={drawDraft} pathOptions={{ color: "#2563eb", weight: 2.5, fillColor: "#93c5fd", fillOpacity: 0.12 }} />
        ) : null}

        {missionMode === "draw" && drawDraft.length > 0 && drawCursor ? (
          <Polyline positions={[drawDraft[drawDraft.length - 1], drawCursor]} pathOptions={{ color: "#0ea5e9", weight: 2.5, dashArray: "4,8", opacity: 0.9 }} />
        ) : null}

        {missionMode === "draw"
          ? drawDraft.map((point, idx) => (
              <Marker
                key={`draft-point-${idx}`}
                position={point}
                icon={createDrawVertexIcon(idx, {
                  active: selectedDraftIdx === idx,
                  hover: hoverDraftIdx === idx,
                  first: idx === 0,
                  last: idx === drawDraft.length - 1,
                })}
                draggable
                eventHandlers={{
                  click: () => setSelectedDraftIdx(idx),
                  mouseover: () => setHoverDraftIdx(idx),
                  mouseout: () => setHoverDraftIdx(null),
                  dragend: (e) => {
                    const ll = e.target.getLatLng();
                    handleDraftPointDrag(idx, ll);
                  },
                }}
              >
                <Tooltip direction="top" permanent>
                  {idx === 0 ? "Start" : (idx === drawDraft.length - 1 ? "End" : `P${idx + 1}`)}
                </Tooltip>
              </Marker>
            ))
          : null}

        {showPlannedPath && missionPath.length > 1 ? (
          <Polyline positions={missionPath} pathOptions={{ color: "#be123c", weight: 3, opacity: 0.95 }} />
        ) : null}

        {showStartPoint && missionStart ? (
          <CircleMarker center={missionStart} radius={7} pathOptions={{ color: "#f59e0b", fillOpacity: 0.9 }}>
            <Tooltip permanent direction="top">Drone Start</Tooltip>
          </CircleMarker>
        ) : null}

        {showTrack && trail.length > 1 ? (
          <Polyline positions={trail} pathOptions={{ color: "#2563eb", weight: 3, opacity: 0.8 }} />
        ) : null}

        {showBreadcrumbs
          ? breadcrumbs.map((point, idx) => (
              <CircleMarker
                key={`crumb-${idx}`}
                center={point}
                radius={3}
                pathOptions={{ color: "#2563eb", fillOpacity: 0.55, opacity: 0.8 }}
              />
            ))
          : null}

        {coverageEnabledForMission && showCoverage
          ? coverageRects.map((r, idx) => {
              const alpha = Math.min(0.72, 0.22 + Math.log2(Math.max(1, r.count)) * 0.12);
              return (
                <Rectangle
                  key={`cov-${idx}`}
                  bounds={r.bounds}
                  pathOptions={{ color: r.color, fillColor: r.color, weight: 0.4, fillOpacity: alpha }}
                />
              );
            })
          : null}

        {showDebug && footprint ? (
          <Polygon positions={footprint} pathOptions={{ color: "#f97316", weight: 2, fillOpacity: 0.05 }} />
        ) : null}

        {origin && missionStart && showStartPoint ? (
          <CircleMarker center={origin} radius={6} pathOptions={{ color: "#f59e0b", fillOpacity: 0.8 }}>
            <Tooltip permanent direction="top">Origin</Tooltip>
          </CircleMarker>
        ) : null}

        {showCurrentTarget && missionActive && currentTarget ? (
          <CircleMarker center={currentTarget} radius={8} pathOptions={{ color: "#7c3aed", fillOpacity: 0.9 }}>
            <Tooltip permanent direction="top">Current Target</Tooltip>
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
