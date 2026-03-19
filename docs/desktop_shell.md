# Desktop Shell Scaffold

This project now includes a first-pass Electron shell under `desktop/` that wraps the existing React dashboard and starts the Python backend as a local service when needed.

## Architecture

The desktop app does **not** replace the existing web architecture.

- React dashboard remains the UI.
- FastAPI remains the backend and local control/service layer.
- REST and SSE endpoints remain unchanged.
- Electron only provides:
  - a native window
  - local backend process management
  - packaged app structure for later distribution

Flow:

```text
Electron main process
  -> checks backend /api/health
  -> reuses existing backend if already running
  -> otherwise launches: python3 -m uvicorn backend.app.main:app
  -> opens native window
  -> loads React frontend from Vite dev server in development
  -> loads built frontend assets in production
```

## Desktop Folder

Files:

- `desktop/package.json`
  - Electron dev scripts
  - packaging scaffold via `electron-builder`
- `desktop/src/main.cjs`
  - app lifecycle
  - window creation
  - backend bootstrap
- `desktop/src/backend.cjs`
  - health probe
  - backend spawn / shutdown
- `desktop/src/config.cjs`
  - desktop host/port/path config
- `desktop/src/preload.cjs`
  - minimal safe preload bridge

## Development Workflow

Existing browser workflow still works:

```bash
uvicorn backend.app.main:app --reload --port 8000
cd frontend
npm run dev
```

Desktop development is additive, not a replacement.

### Option A: Let Electron manage the backend

Terminal 1:

```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Terminal 2:

```bash
cd desktop
npm install
npm run dev
```

In this mode Electron:

- opens the native window
- checks `http://127.0.0.1:8000/api/health`
- starts the backend itself if it is not already running

### Option B: Keep backend/frontend fully separate

Terminal 1:

```bash
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2:

```bash
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

Terminal 3:

```bash
cd desktop
DESKTOP_MANAGE_BACKEND=0 npm run dev
```

In this mode Electron reuses the already-running backend and behaves as a native shell only.

## Configuration

Environment variables:

- `DESKTOP_BACKEND_HOST`
  - default: `127.0.0.1`
- `DESKTOP_BACKEND_PORT`
  - default: `8000`
- `DESKTOP_FRONTEND_URL`
  - default: `http://127.0.0.1:5173`
- `DESKTOP_PYTHON`
  - default: `python3`
- `DESKTOP_MANAGE_BACKEND`
  - default: `1`
- `DESKTOP_BACKEND_TIMEOUT_MS`
  - default: `30000`

## Production / Packaging Notes

The current scaffold supports:

- loading built frontend assets in production
- bundling backend source files as Electron resources

What is **not** fully solved yet:

- shipping an embedded Python runtime
- dependency bootstrapping inside the packaged app
- OS-specific installers beyond the Electron Builder scaffold

So the current packaging path should be treated as a scaffold for the next step, not a finished installer story.

Practical next step later:

1. build frontend with `cd frontend && npm run build`
2. install desktop dependencies with `cd desktop && npm install`
3. package with `npm run dist`

For a fully standalone desktop release, the next milestone would be bundling Python plus backend dependencies in a reproducible way.
