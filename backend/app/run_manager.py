from __future__ import annotations

import csv
import io
import json
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")


def _dist_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    d_lat = (lat2 - lat1) * 111320.0
    c = math.cos(math.radians((lat1 + lat2) * 0.5))
    d_lng = (lon2 - lon1) * 111320.0 * max(0.1, abs(c))
    return math.hypot(d_lat, d_lng)


def _polyline_length_m(track_lat_lon: list[tuple[float, float]]) -> float:
    if len(track_lat_lon) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(track_lat_lon)):
        total += _dist_m(track_lat_lon[i - 1], track_lat_lon[i])
    return total


def _pdf_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_simple_pdf(lines: list[str]) -> bytes:
    y0 = 780
    dy = 14
    text_lines = ["BT", "/F1 10 Tf", f"72 {y0} Td"]
    for i, line in enumerate(lines[:48]):
        if i > 0:
            text_lines.append(f"0 -{dy} Td")
        text_lines.append(f"({_pdf_escape(line)}) Tj")
    text_lines.append("ET")
    content_stream = "\n".join(text_lines).encode("latin-1", errors="replace")

    objects: list[bytes] = []
    objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objects.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
    objects.append(
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
    )
    objects.append(b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")
    objects.append((f"5 0 obj << /Length {len(content_stream)} >> stream\n").encode("ascii") + content_stream + b"\nendstream endobj\n")

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    xref: list[int] = [0]
    for obj in objects:
        xref.append(out.tell())
        out.write(obj)
    xref_start = out.tell()
    out.write(f"xref\n0 {len(xref)}\n".encode("ascii"))
    out.write(b"0000000000 65535 f \n")
    for off in xref[1:]:
        out.write(f"{off:010d} 00000 n \n".encode("ascii"))
    out.write(
        (
            "trailer\n"
            f"<< /Size {len(xref)} /Root 1 0 R >>\n"
            "startxref\n"
            f"{xref_start}\n"
            "%%EOF\n"
        ).encode("ascii")
    )
    return out.getvalue()


@dataclass
class RunState:
    run_id: str
    status: str
    started_at_utc: str
    stopped_at_utc: str | None = None
    scenario: dict[str, Any] = field(default_factory=dict)
    controller: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)


class RunManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: RunState | None = None

    def start(self, scenario: dict[str, Any] | None, controller: dict[str, Any] | None, notes: str | None) -> dict[str, Any]:
        with self._lock:
            if self._current is not None and self._current.status == "running":
                raise ValueError("a run is already active")
            s = RunState(
                run_id=_run_id(),
                status="running",
                started_at_utc=_utc_now_iso(),
                scenario=dict(scenario or {}),
                controller=dict(controller or {}),
                notes=str(notes or ""),
            )
            s.events.append({"t_utc": _utc_now_iso(), "event": "RUN_STARTED", "data": {}})
            self._current = s
            return self._as_dict_locked(s)

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._current is None:
                raise ValueError("no active run")
            self._current.status = "stopped"
            self._current.stopped_at_utc = _utc_now_iso()
            self._current.events.append({"t_utc": _utc_now_iso(), "event": "RUN_STOPPED", "data": {}})
            return self._as_dict_locked(self._current)

    def set_notes(self, notes: str) -> dict[str, Any]:
        with self._lock:
            if self._current is None:
                raise ValueError("no active run")
            self._current.notes = str(notes)
            return self._as_dict_locked(self._current)

    def log(self, event: str, data: dict[str, Any] | None = None) -> None:
        with self._lock:
            if self._current is None:
                return
            self._current.events.append({"t_utc": _utc_now_iso(), "event": str(event), "data": dict(data or {})})
            if len(self._current.events) > 2000:
                self._current.events = self._current.events[-2000:]

    def current(self) -> dict[str, Any] | None:
        with self._lock:
            if self._current is None:
                return None
            return self._as_dict_locked(self._current)

    def _as_dict_locked(self, state: RunState) -> dict[str, Any]:
        return {
            "run_id": state.run_id,
            "status": state.status,
            "started_at_utc": state.started_at_utc,
            "stopped_at_utc": state.stopped_at_utc,
            "scenario": dict(state.scenario),
            "controller": dict(state.controller),
            "notes": state.notes,
            "events": list(state.events),
        }

    def _require_current(self) -> dict[str, Any]:
        cur = self.current()
        if cur is None:
            raise ValueError("no run available")
        return cur

    def export_json(self, snapshot_getter: Callable[[], dict[str, Any]]) -> bytes:
        cur = self._require_current()
        snap = snapshot_getter()
        track_items = list((snap.get("track") or {}).get("items") or [])
        track_lat_lon = [
            (float(p.get("lat")), float(p.get("lon")))
            for p in track_items
            if p.get("lat") is not None and p.get("lon") is not None
        ]
        summary = {
            "coverage_pct": ((snap.get("coverage") or {}).get("stats") or {}).get("coverage_pct"),
            "overlap_pct": ((snap.get("coverage") or {}).get("stats") or {}).get("overlap_pct"),
            "total_hits": ((snap.get("coverage") or {}).get("stats") or {}).get("total_hits"),
            "track_points": len(track_lat_lon),
            "track_length_m": _polyline_length_m(track_lat_lon),
        }
        out = {
            "run": cur,
            "exported_at_utc": _utc_now_iso(),
            "summary": summary,
            "snapshot": snap,
        }
        return json.dumps(out, indent=2).encode("utf-8")

    def export_path_csv(self, snapshot_getter: Callable[[], dict[str, Any]]) -> bytes:
        self._require_current()
        snap = snapshot_getter()
        mission = snap.get("mission_path") or {}
        waypoints = mission.get("waypoints_lng_lat") or []
        track_items = list((snap.get("track") or {}).get("items") or [])

        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["kind", "index", "t_unix", "lat", "lng", "rel_alt_m", "speed_m_s"])
        for i, wp in enumerate(waypoints):
            if not isinstance(wp, (list, tuple)) or len(wp) < 2:
                continue
            w.writerow(["planned_wp", i, "", float(wp[1]), float(wp[0]), "", ""])
        for i, p in enumerate(track_items):
            lat = p.get("lat")
            lon = p.get("lon")
            if lat is None or lon is None:
                continue
            w.writerow([
                "flown_track",
                i,
                p.get("t_unix", ""),
                float(lat),
                float(lon),
                p.get("rel_alt_m", ""),
                p.get("speed_m_s", ""),
            ])
        return buf.getvalue().encode("utf-8")

    def export_path_geojson(self, snapshot_getter: Callable[[], dict[str, Any]]) -> bytes:
        self._require_current()
        snap = snapshot_getter()
        mission = snap.get("mission_path") or {}
        waypoints = [
            [float(wp[0]), float(wp[1])]
            for wp in (mission.get("waypoints_lng_lat") or [])
            if isinstance(wp, (list, tuple)) and len(wp) >= 2
        ]
        track = [
            [float(p.get("lon")), float(p.get("lat"))]
            for p in (snap.get("track") or {}).get("items") or []
            if p.get("lon") is not None and p.get("lat") is not None
        ]
        features = []
        if waypoints:
            features.append({
                "type": "Feature",
                "properties": {"name": "planned_path"},
                "geometry": {"type": "LineString", "coordinates": waypoints},
            })
        if track:
            features.append({
                "type": "Feature",
                "properties": {"name": "flown_track"},
                "geometry": {"type": "LineString", "coordinates": track},
            })
        gj = {"type": "FeatureCollection", "features": features}
        return json.dumps(gj, indent=2).encode("utf-8")

    def export_coverage_csv(self, snapshot_getter: Callable[[], dict[str, Any]]) -> bytes:
        self._require_current()
        snap = snapshot_getter()
        cov = snap.get("coverage") or {}
        cells = cov.get("covered_cells") or []

        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["r", "c", "count", "lat_min", "lat_max", "lng_min", "lng_max"])
        for c in cells:
            w.writerow([
                c.get("r", ""),
                c.get("c", ""),
                c.get("count", ""),
                c.get("lat_min", ""),
                c.get("lat_max", ""),
                c.get("lng_min", ""),
                c.get("lng_max", ""),
            ])
        return buf.getvalue().encode("utf-8")

    def export_report_pdf(self, snapshot_getter: Callable[[], dict[str, Any]]) -> bytes:
        cur = self._require_current()
        snap = snapshot_getter()
        cov_stats = (snap.get("coverage") or {}).get("stats") or {}
        sitl_state = snap.get("sitl_state") or {}
        lines = [
            "Drone Thesis Run Report",
            "",
            f"Run ID: {cur.get('run_id')}",
            f"Status: {cur.get('status')}",
            f"Started (UTC): {cur.get('started_at_utc')}",
            f"Stopped (UTC): {cur.get('stopped_at_utc') or '-'}",
            f"Exported (UTC): {_utc_now_iso()}",
            "",
            f"Scenario: {json.dumps(cur.get('scenario', {}), ensure_ascii=True)}",
            f"Controller: {json.dumps(cur.get('controller', {}), ensure_ascii=True)}",
            f"Notes: {cur.get('notes') or '-'}",
            "",
            f"Coverage %: {cov_stats.get('coverage_pct', '-')}",
            f"Overlap %: {cov_stats.get('overlap_pct', '-')}",
            f"Covered Cells: {cov_stats.get('covered_cells', '-')}",
            f"Total Cells: {cov_stats.get('total_cells', '-')}",
            f"Elapsed Active (s): {cov_stats.get('time_elapsed_s', '-')}",
            "",
            f"SITL State: {sitl_state.get('state', '-')}",
            f"Waypoint Progress: {sitl_state.get('waypoint_index', 0)}/{sitl_state.get('waypoint_count', 0)}",
        ]
        return build_simple_pdf(lines)
