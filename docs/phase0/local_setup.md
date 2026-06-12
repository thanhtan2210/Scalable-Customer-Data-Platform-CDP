# Phase 0: Local Development Setup (Lightweight)

**Date:** June 2026
**Context:** This document outlines the "Bare-Metal" local development strategy for the Universal AutoML Churn Platform. It is specifically designed for developers working on machines with limited resources (RAM/CPU) where running the full Docker Compose stack (MinIO, PostgreSQL, MLflow, FastAPI, Streamlit) simultaneously causes Out-Of-Memory (OOM) errors.

## 1. Architectural Strategy: The "Bare-Metal" Approach

While the project is fully Dockerized for production and robust local testing (`docker-compose.yml`), day-to-day development of the core AutoML engine can be done directly on the host operating system using Python Virtual Environments (`.venv`).

**Benefits of this approach:**
- **Zero Docker Overhead:** Saves 4GB-8GB of RAM.
- **Fast Iteration:** Changes to Python files are instantly reflected via `uvicorn --reload` without rebuilding containers.
- **Easy Debugging:** Python tracebacks and print statements appear directly in your primary terminal.

## 2. Dependency Management

To keep our production Docker images as lean as possible, dependencies are strictly segregated:

- `requirements.txt`: Contains **ONLY** the libraries needed to run the application in production (e.g., FastAPI, Pandas, Scikit-learn, XGBoost).
- `requirements-local.txt`: Inherits from `requirements.txt` but adds tools exclusively for local development (e.g., `pytest`, `black`, `ruff`, `mypy`).

### Installation
Ensure your virtual environment is activated, then run:

```bash
# Using Make
make install-local

# Or using pip directly
pip install -r requirements-local.txt
```

## 3. Running Services Locally (Without Docker)

To run the platform locally, we utilize fallback mechanisms (like SQLite instead of PostgreSQL).

### Step 1: Start the FastAPI Backend
Open a terminal, activate the `.venv`, and start the backend using SQLite as the database:

**Windows (PowerShell):**
```powershell
$env:DATABASE_URL="sqlite:///./churn_db.sqlite"
uvicorn backend.app.main:app --reload --port 8000
```

**Linux/Mac (Bash):**
```bash
DATABASE_URL="sqlite:///./churn_db.sqlite" uvicorn backend.app.main:app --reload --port 8000
```
*Verify: Open `http://localhost:8000/docs` in your browser to see the Swagger UI.*

### Step 2: Start the Analytics Hub
Open a **second** terminal, activate the `.venv`, and start Streamlit:

```bash
streamlit run analytics/streamlit_app.py
```
*Verify: The browser will automatically open `http://localhost:8501` showing the Analytics Hub.*

## 4. Testing Core Logic Independently

Because the system follows Clean Architecture, you do not need to upload files via the API to test the core AutoML engine. You can write simple Python scripts in the root directory to test functions directly:

```python
# test_core.py
import pandas as pd
from backend.app.core.profiler.orchestrator import run_profiling

# Load a local raw file directly
df = pd.read_csv("data/raw/cleaned_telco.csv")

# Test the profiler
profiles = run_profiling(df)
for p in profiles:
    print(f"{p.name}: {p.inferred_role}")
```

## 5. Artifact Management

To keep the root directory clean, all auto-generated test artifacts (like JSON reports or coverage HTML) must be routed to the `tests/results/` directory. This directory is explicitly ignored by git in the `.gitignore` file.
