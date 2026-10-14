# MindFlow Tutor

[![CI/CD](https://github.com/BaryuH/mindflow-tutor/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/BaryuH/mindflow-tutor/actions/workflows/ci-cd.yml)

Proactive Multimodal AI Learning Assistant — FastAPI backend + React/Vite frontend.

Detects distraction and confusion via webcam in real time, then proactively helps the user refocus or clarifies confusing material using AI.

## Tech Stack

| Layer    | Tech                                                |
| -------- | --------------------------------------------------- |
| Frontend | React 18, TypeScript, Vite, shadcn/ui, Tailwind CSS |
| Backend  | FastAPI, Python 3.11, Uvicorn                       |
| Database | MongoDB (Motor async driver)                        |
| AI       | OpenAI GPT-4o, ElevenLabs TTS, Tavily Search        |
| RAG      | LlamaParse + ChromaDB                               |
| Auth     | JWT + Email verification (SMTP/Gmail), Google/Facebook OAuth |
| Vision   | MediaPipe, OpenCV, hsemotion (EfficientNet-B0)      |
| CI/CD    | GitHub Actions → GHCR → Azure App Service           |

---

## Quick start

Prerequisites: **Python 3.11**, **Node.js 20**, **Docker** (for MongoDB), and API keys for OpenAI, ElevenLabs, Tavily and LlamaParse.

```bash
# 1. MongoDB
docker compose up -d mongo

# 2. Backend — web + AI deps in one pass (CPU-only torch)
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  -r requirements-web.txt -r requirements-ai.txt
cp .env.example .env        # then fill in your keys
uvicorn server:app --reload --port 8000

# 3. Frontend (new terminal)
cd client
npm ci
npm run dev
```

- App: http://localhost:5173 (the Vite dev server proxies `/api` → `:8000`)
- API docs: http://localhost:8000/docs

With `make` available, the same steps are `make install`, `make db`, `make backend` and `make client`.

To run the production image locally instead: `docker compose --profile full up --build` → http://localhost:8000.

### Configuration

All settings live in [`backend/.env.example`](backend/.env.example) — copy it to `backend/.env`. Notes:

- `JWT_SECRET_KEY` — generate with `python -c "import secrets; print(secrets.token_hex(32))"`. The server logs a warning if it is unset.
- `ADMIN_EMAIL` / `ADMIN_PASSWORD` — optional; when both are set an admin account is created on startup.
- `SMTP_PASSWORD` — a Gmail App Password, used for email verification.

Model weights (face detector, face landmarker) are downloaded automatically on first use.

---

## Development

| Task | Command |
| --- | --- |
| Backend lint | `cd backend && ruff check .` |
| Backend tests | `cd backend && pytest` |
| Frontend lint / types | `cd client && npm run lint && npm run typecheck` |
| Frontend tests | `cd client && npm test` |
| Smoke test against a running server | `python scripts/smoke_test.py` |

Git hooks (ruff, secret scanning, large-file guard): `pip install pre-commit && pre-commit install`.

### CI/CD

[`.github/workflows/ci-cd.yml`](.github/workflows/ci-cd.yml):

1. **Every PR and push** — backend (ruff + pytest) and frontend (lint, typecheck, test, build) run in parallel.
2. **Push to `main`** — builds `Dockerfile.prod` and pushes it to `ghcr.io/baryuh/mindflow-tutor`.
3. **Deploy** — runs only once the repository variable `AZURE_WEBAPP_NAME` and the secret `AZURE_WEBAPP_PUBLISH_PROFILE` are configured; it uses the `production` environment.

---

## Project Structure

```
mindflow-tutor/
├── backend/
│   ├── core/               # Config, auth, DB, prompts
│   ├── engines/
│   │   ├── attention_system/   # Distraction detection (head pose)
│   │   ├── confusion_system/   # Confusion detection (emotion recognition)
│   │   ├── combined_detector_engine.py
│   │   ├── orchestrator.py     # Vision + RAG + LLM pipeline
│   │   └── rag_retriever.py
│   ├── persistence/        # Session memory, ChromaDB vector store
│   ├── providers/          # OpenAI, Tavily, voice (Whisper + ElevenLabs)
│   ├── routers/            # API route handlers
│   ├── schemas/            # Pydantic models
│   ├── tests/              # pytest suite
│   ├── server.py           # FastAPI entrypoint
│   ├── pyproject.toml      # ruff + pytest config
│   └── requirements-{web,ai}.txt
├── client/
│   └── src/
│       ├── components/     # Feature components + ui/ (shadcn/ui)
│       ├── contexts/       # Auth context
│       ├── hooks/
│       ├── lib/            # API clients, helpers
│       └── pages/          # Index, Login, Dashboard, Session, ...
├── scripts/smoke_test.py   # Manual checks against a running server
├── docker-compose.yml
├── Dockerfile.prod
└── Makefile
```

---

## API Overview

| Method | Endpoint                        | Description                        |
| ------ | ------------------------------- | ---------------------------------- |
| GET    | `/api/health`                   | Health check                       |
| POST   | `/api/auth/register`            | Register new user                  |
| POST   | `/api/auth/login`               | Login, returns JWT                 |
| GET    | `/api/auth/verify-email`        | Verify email token                 |
| POST   | `/api/v1/ask`                   | Ask a question (RAG + AI)          |
| POST   | `/api/v1/process-trigger`       | Confusion trigger (screen + AI)    |
| POST   | `/api/v1/detector/frame`        | Webcam frame analysis              |
| POST   | `/api/v1/utils/classify-intent` | Semantic intent classification     |
| POST   | `/api/v1/documents/upload`      | Upload PDF for RAG                 |
| WS     | `/api/v1/voice`                 | Real-time voice chat (WebSocket)   |

Full interactive docs: http://localhost:8000/docs
