# React + FastAPI Phase 1 (Telemetry + Control)

## Run backend

```bash
cd /mnt/data/drone_thesis_clean
source .venv-web/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

## Run frontend

```bash
cd /mnt/data/drone_thesis_clean/frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

## SITL connection defaults

- URL: `udp:127.0.0.1:14550`
- Override with `.env`:

```bash
MAVLINK_URL=udp:127.0.0.1:14550
```

## Phase 1 endpoints

- `GET /api/health`
- `GET /api/status`
- `GET /api/telemetry`
- `POST /api/connection/connect`
- `POST /api/connection/disconnect`
- `POST /api/control/arm`
- `POST /api/control/disarm`
- `POST /api/control/takeoff` with JSON `{"alt_m": 10}`
- `POST /api/control/rtl`
- `POST /api/control/land`
- `POST /api/control/set_mode` with JSON `{"mode": "GUIDED"}`

## Quick CLI test

```bash
curl http://127.0.0.1:8000/api/status
curl http://127.0.0.1:8000/api/telemetry
curl -X POST http://127.0.0.1:8000/api/control/set_mode -H 'Content-Type: application/json' -d '{"mode":"GUIDED"}'
```
