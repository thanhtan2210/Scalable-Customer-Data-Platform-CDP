# Deployment Guide

This document contains instructions for deploying the Platform services locally via Docker, setting up CI/CD workflows, and deploying to cloud platforms (Render, Vercel, and Streamlit Cloud).

---

## 1. Local Container Deployment & CI/CD

### API Container Setup

- **Dockerfile**: `deploy/api/Dockerfile`
- **Minimal requirements**: `deploy/api/requirements.txt`
- **Environment variables at runtime**:
  - `MLFLOW_S3_ENDPOINT_URL`
  - `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`
  - `MLFLOW_TRACKING_URI`
  - `MODEL_NAME` / `MODEL_VERSION`

### Build & Run Locally

- **Build**: `docker build -t cdp-api:latest -f deploy/api/Dockerfile .`
- **Run**: `docker run -p 8000:8000 --env-file .env cdp-api:latest`

### Compose with MinIO + MLflow

- The API service is declared in `docker-compose.yml` (service `api`).
- **Start all services**:
  - `docker compose up -d`
- **Access Addresses**:
  - API Backend: <http://localhost:8000>
  - MLflow UI: <http://localhost:5000>
  - MinIO Console: <http://localhost:9001> (Default credentials: `admin` / `password`)

#### Rebuild API after code changes

- `docker compose build api && docker compose up -d api`

### CI/CD (GitHub Container Registry)

- **Workflow file**: `.github/workflows/api-image.yml`
- Pushes to the `main` branch will build and push the production image to GHCR at `ghcr.io/<OWNER>/cdp-api`.
- **Permissions**: The workflow sets `packages: write` and logs in with `GITHUB_TOKEN`.

#### Pull & Run from Registry

- **Pull**: `docker pull ghcr.io/<OWNER>/cdp-api:main`
- **Run**: `docker run -p 8000:8000 --env-file .env ghcr.io/<OWNER>/cdp-api:main`

---

## 2. Production Cloud Deployment

### Render.com (Backend API)

- **Service Type:** Web Service
- **Runtime:** Docker (using root `backend/Dockerfile` or custom deploy dockerfile)
- **Environment Variables:**
  - `ENVIRONMENT`: `production`
  - `API_KEY`: Generate a secure API key
  - `DATABASE_URL`: Your Supabase connection string
  - `S3_ACCESS_KEY`: Cloudflare R2 Access Key
  - `S3_SECRET_KEY`: Cloudflare R2 Secret Key
  - `S3_ENDPOINT_URL`: `https://<id>.r2.cloudflarestorage.com`
  - `S3_BUCKET`: `churn-assets`
  - `MLFLOW_TRACKING_URI`: `https://dagshub.com/<user>/<repo>.mlflow`
  - `MLFLOW_TRACKING_USERNAME`: DagsHub Username
  - `MLFLOW_TRACKING_PASSWORD`: DagsHub Token
  - `GROQ_API_KEY`: For Layer 3 Profiling
  - `ENABLE_LLM_LAYER`: `true`

### Vercel (Frontend UI)

- **Framework Preset:** Vite
- **Build Command:** `npm run build`
- **Output Directory:** `dist`
- **Environment Variables:**
  - `VITE_API_URL`: Your Render service URL + `/api/v1`
  - `VITE_API_KEY`: The secure key generated for the backend API

### Streamlit Cloud (Analytics Dashboard)

- **Main file path:** `analytics/streamlit_app.py`
- **Secrets Configuration (secrets.toml):**

  ```toml
  API_URL = "https://your-render-app.onrender.com/api/v1"
  API_KEY = "your-key"
  ```
