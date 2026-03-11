# React + FastAPI Phase 2 (Map)

## New backend endpoints

- `GET /api/map_state`
- `GET /api/track?limit=600`
- `GET /api/map/tile/{z}/{x}/{y}.png`

`/api/map_state` includes:

- `tile_url_template`
- `center_lng_lat`, `zoom`
- `origin`
- `bounds_polygon_lng_lat`
- `vehicle`

## Frontend map implementation

- `frontend/src/MapPanel.jsx` (Leaflet via `react-leaflet`)
- Layers:
  - vehicle marker
  - heading line
  - breadcrumb trail
  - bounds polygon
  - origin marker
- Resize reliability:
  - `ResizeObserver` + `invalidateSize()`
  - window resize handling
  - re-mount on sidebar collapse toggle

## Install frontend deps

```bash
cd /mnt/data/drone_thesis_clean/frontend
npm install
```

## Run

Backend:

```bash
cd /mnt/data/drone_thesis_clean
source .venv-web/bin/activate
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Frontend:

```bash
cd /mnt/data/drone_thesis_clean/frontend
npm run dev
```
