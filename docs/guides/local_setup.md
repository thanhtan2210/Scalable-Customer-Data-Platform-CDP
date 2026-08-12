# Local Development Setup Guide (Bare-Metal)

This document outlines the local development setup for the CDP platform. It is designed for fast developer iteration without requiring Docker Compose or external cloud infrastructure.

---

## 1. Prerequisites

- **Python**: 3.10 or higher
- **PowerShell** (Windows) or **Bash** (Linux/macOS)
- **Git**

---

## 2. Environment Setup

### Step 1: Clone Repository & Create Virtual Environment

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## 3. Environment Configuration (`.env`)

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

Key local development configuration variables:

```env
# Database Settings (Local SQLite Fallback)
DATABASE_URL=sqlite:///./test.db

# Storage Settings (Local Filesystem Fallback)
STORAGE_BACKEND=local
LOCAL_STORAGE_DIR=./data/storage

# MLflow Settings (Local Filesystem Registry)
MLFLOW_TRACKING_URI=file:///./mlruns
MLFLOW_EXPERIMENT_NAME=churn-prediction

# AutoML Tuning Parameters
OPTUNA_N_TRIALS=100
OPTUNA_TIMEOUT_SECONDS=3600
ENABLE_STACKING=true
```

---

## 4. Running Services Locally

### Step 1: Start the Backend REST API

Run FastAPI backend with live reloading:

```powershell
uvicorn backend.app.main:app --reload --port 8000
```

Verify backend is healthy:
- Open browser: `http://localhost:8000/health` (Expected response: `{"status": "ok"}`)
- API Documentation (Swagger): `http://localhost:8000/docs`

### Step 2: Start Local MLflow UI (Optional)

In a **separate terminal window**:

```powershell
.\.venv\Scripts\Activate.ps1
mlflow server --backend-store-uri file:///./mlruns --port 5000
```

Verify MLflow UI: Open `http://localhost:5000`

---

## 5. Executing Tests

### Run Unit & Integration Tests (pytest)

```powershell
pytest tests/ -q
```

### Run End-to-End Dataset Integration Suite

```powershell
# Fast smoke test (3 Optuna trials per model)
python scripts/test_datasets_e2e.py --layer 4 --dataset bank --fast

# Full E2E suite across all datasets
python scripts/test_datasets_e2e.py --layer 4 --dataset all
```
