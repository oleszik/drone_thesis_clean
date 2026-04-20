# System Data Flow Loop

This diagram is derived from the runtime paths in:
- `frontend/src/App.jsx`
- `frontend/src/MapPanel.jsx`
- `frontend/src/pages/RealTest.jsx`
- `backend/app/main.py`
- `backend/app/sitl_executor.py`
- `backend/app/mavlink_service.py`
- `scripts/bridge/runtime.py`

## 1) Mission Operations Loop (Sim + Real)

```mermaid
flowchart LR
    U[Operator] --> UI[React UI<br/>frontend/src]
    DS[Desktop Shell<br/>desktop/src/main.cjs] --> UI

    UI -->|POST/GET mission + control APIs| API[FastAPI<br/>backend/app/main.py]
    API -->|plan/update path| MS[MissionService]
    API -->|start/stop execution| EXE[SitlExecutor / Real Executor]
    API -->|arm/mode/goto/land| MAV[MavlinkService / RealRadioService]

    EXE --> MAV
    MAV -->|MAVLink setpoints| AP[ArduPilot SITL or Real Flight Controller]
    AP -->|heartbeat + telemetry| MAV

    MAV -->|status/telemetry/track| API
    API -->|coverage sampler updates| COV[CoverageService]

    API -->|SSE: telemetry| UI
    API -->|SSE: bridge_state| UI
    API -->|SSE: track| UI
    API -->|SSE: coverage| UI

    COV --> API
    UI -->|map/status/events| U
```

Main loop closure:
1. User action in UI triggers backend mission/control endpoint.
2. Backend sends MAVLink commands to SITL/real vehicle.
3. Vehicle returns telemetry and heartbeat.
4. Backend pushes SSE updates (`telemetry`, `bridge_state`, `track`, `coverage`) back to UI.
5. User adjusts mission/controls from updated state, repeating the loop.

## 2) RL Bridge Runtime Loop (`scripts/bridge/runtime.py`)

```mermaid
flowchart LR
    M[RL Model<br/>runs/.../best_model.zip] --> BR[Bridge Runtime]
    BR -->|connect + command| MC[MAVLink Client]
    MC -->|set_position_target_local_ned| AP[ArduPilot SITL/FC]
    AP -->|telemetry messages| MC
    MC --> BR
    BR -->|target update + adaptive logic| BR
    BR -->|coverage tracking| CG[CoverageGridTracker]
    CG --> BR
    BR -->|telemetry.jsonl + summary.json| OUT[runs/ardupilot_scan/...]
```

