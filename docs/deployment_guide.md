# Cloud Deployment Guide

## 1. Render.com (Backend)
- **Service Type:** Web Service
- **Runtime:** Docker
- **Env Vars:**
  - `ENVIRONMENT`: `production`
  - `API_KEY`: Generate a secure key
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

## 2. Vercel (Frontend)
- **Framework Preset:** Vite
- **Build Command:** `npm run build`
- **Output Directory:** `dist`
- **Env Vars:**
  - `VITE_API_URL`: Your Render service URL + `/api/v1`
  - `VITE_API_KEY`: The key you generated above

## 3. Streamlit Cloud (Analytics)
- **Main file path:** `analytics/streamlit_app.py`
- **Secrets (secrets.toml):**
  ```toml
  API_URL = "https://your-render-app.onrender.com/api/v1"
  API_KEY = "your-key"
  ```
