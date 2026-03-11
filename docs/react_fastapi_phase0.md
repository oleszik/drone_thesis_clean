# React + FastAPI Phase 0

## 1) Backend setup

```bash
cd /mnt/data/drone_thesis_clean
python -m venv .venv-web
source .venv-web/bin/activate
pip install -r backend/requirements.txt
```

Run backend:

```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/api/health
```

## 2) Frontend setup

```bash
cd /mnt/data/drone_thesis_clean/frontend
npm install
npm run dev
```

Open:

`http://127.0.0.1:5173`

Expected text:

`backend connected`

## 3) Config + secrets

- Shared config: `config.yaml`
- Secrets template: `.env.example`
- Real secrets: `.env` (ignored by git)
