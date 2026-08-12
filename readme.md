# Scalable Customer Data Platform (CDP)

[![CDP Pipeline CI](https://github.com/thanhtan2210/Scalable-Customer-Data-Platform-CDP/actions/workflows/ci.yml/badge.svg)](https://github.com/thanhtan2210/Scalable-Customer-Data-Platform-CDP/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.0%2B-0194E2)](https://mlflow.org)

An enterprise-grade Customer Data Platform designed for customer data processing, automated statistical/semantic profiling, production AutoML churn modelling, multi-task learning, and batch inference serving.

---

## 🌟 Key Features

- **Automated Data Profiling Engine**: 3-layer profiling (Statistical, Semantic Regex, LLM-assisted) for instant data health assessment, semantic role inference (ID, Target, Categorical, Numeric, Text), and leakage detection.
- **Advanced AutoML & Model Routing**: Multi-candidate hyperparameter search using **Optuna** across **LightGBM**, **CatBoost**, **XGBoost**, **Random Forest**, and **Logistic Regression**, integrated with **MedianPruner**, automated class imbalance handling (`class_weight='balanced'`), and post-search **Stacking Ensembles**.
- **Multi-Task Learning (MTL)**: PyTorch-based neural backbone for simultaneous Churn Prediction (binary classification) and Customer Performance Index / CPI (regression) with Continual Learning capability.
- **RESTful API Core**: Clean FastAPI backend serving dataset uploads, async job execution, real-time drift monitoring (KS-test, PSI), and batch/single inference endpoints with 10-minute double-checked model caching.
- **End-to-End Test Suite**: Automated 4-layer integration testing (`scripts/test_datasets_e2e.py`) validating Upload $\rightarrow$ Profile $\rightarrow$ Train $\rightarrow$ Batch Predict across multiple benchmark datasets.

---

## 🚀 Quick Start

### 1. Environment Setup

Ensure Python 3.10+ is installed:

```powershell
# Clone the repository
git clone https://github.com/thanhtan2210/Scalable-Customer-Data-Platform-CDP.git
cd Scalable-Customer-Data-Platform-CDP

# Create & activate virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install core & local development dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

*Default fallback configuration uses local SQLite (`test.db`) and local filesystem storage — no Docker or MinIO required for quick development.*

### 3. Launch Services

Start the backend API server:

```powershell
# Start FastAPI server on port 8000
uvicorn backend.app.main:app --reload --port 8000
```

Open interactive Swagger API Documentation at: `http://localhost:8000/docs`

### 4. Run End-to-End Dataset Integration Test Suite

With the server running in one terminal, execute the dataset test suite:

```powershell
# Quick smoke test on Bank Churn dataset (Fast 3-trial Optuna mode)
python scripts/test_datasets_e2e.py --layer 4 --dataset bank --fast

# Full E2E pipeline test on all verified datasets
python scripts/test_datasets_e2e.py --layer 4 --dataset all
```

---

## 🏗 System Architecture & Flow

```mermaid
graph TD
    A[Client / App] -->|POST /api/v1/datasets/upload| B[FastAPI Backend]
    B -->|Save raw file| C[(Storage: R2 / S3 / Local)]
    B -->|Metadata| D[(SQLite / PostgreSQL)]
    
    A -->|POST /api/v1/datasets/{id}/profile| B
    B -->|Run 3-Layer Profiling| E[Profiler Engine]
    E -->|Statistical, Semantic, Leakage Check| E
    E -->|Generate Validation Schema| F[Pandera Schema Gen]
    
    A -->|POST /api/v1/jobs/{id}/train| B
    B -->|Trigger Async Job| G[AutoML / PyTorch MTL Engine]
    G -->|Model Routing & Optuna Optimization| H[Optuna 100 Trials + Pruning]
    H -->|Candidate Models: LightGBM / CatBoost / XGB / RF| H
    H -->|Build Stacking Ensemble| I[Stacking Meta-Learner]
    I -->|Log Params, Metrics & Artifacts| J[(MLflow Registry)]
    
    A -->|POST /api/v1/predict/batch| B
    B -->|Thread-safe Model Cache| K[Serving Layer]
    K -->|Load Model & Threshold| J
    K -->|Drift Check: KS & PSI| L[Drift Detector]
```

Detailed component specifications are documented in [docs/architecture/ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md).

---

## 📁 Project Structure

```
Scalable-Customer-Data-Platform-CDP/
├── backend/app/
│   ├── api/v1/          # REST API endpoints (datasets, jobs, predict, monitoring)
│   ├── core/
│   │   ├── etl/         # Data cleaning, schema normalization, SLA checks
│   │   ├── profiler/    # Statistical & semantic profiling, CPI target synthesis
│   │   ├── pipeline/    # Scikit-learn preprocessing pipeline builder & Pandera schemas
│   │   ├── training/    # AutoML engine (Optuna), model router, PyTorch MTL trainer
│   │   └── serving/     # Prediction cache, drift detection, threshold calibration
│   └── db/              # SQLAlchemy ORM models & Alembic migrations
├── data/
│   └── dataset/         # Benchmark dataset catalog & CSV data files
├── docs/                # Architectural docs & step-by-step guides
│   ├── architecture/    # Deep dive system architecture
│   └── guides/          # Local setup, AutoML guide, E2E testing, adding datasets
├── scripts/             # Integration test scripts & benchmark tools
├── analytics/           # Analytics & visual dashboard scripts
├── docker-compose.yml   # Multi-service production stack definition
└── requirements.txt     # Python dependencies
```

---

## 📊 Dataset Catalog & Benchmarks

The CDP platform includes a standardized dataset catalog (`data/dataset/catalog.yaml`) for cross-domain model benchmark evaluation:

| Dataset ID | Domain / Source | Size | Target Column | Key Characteristics | Status |
|---|---|---|---|---|---|
| `bank` | Banking (Kaggle) | 10,000 x 14 | `Exited` | Tabular dense, target synonym testing | Verified |
| `telco` | Telecom (IBM) | 7,043 x 21 | `Churn` | Categorical & numerical mixed, spaces in numeric | Verified |
| `ecommerce` | E-Commerce | 5,630 x 20 | `Churn` | Imbalanced churn rate (~17%), missing values | Verified |
| `bank_marketing_full` | Banking Direct Marketing | 45,211 x 17 | `y` | Semicolon separator, binary target | Verified |
| `credit_card` | Credit Card Customers | 10,127 x 23 | `Attrition_Flag` | Rich behavioral features, high-AUC target (>0.93) | Pending data file |
| `ibm_hr_attrition` | HR Analytics | 1,470 x 35 | `Attrition` | HR employee attrition benchmark | Pending data file |

For instructions on adding new datasets to the catalog, see [docs/guides/adding_datasets.md](docs/guides/adding_datasets.md).

---

## 📚 Documentation & Guides

- 📐 **[System Architecture](docs/architecture/ARCHITECTURE.md)** — Detailed microservice design & ML lifecycle flow.
- 💻 **[Local Setup Guide](docs/guides/local_setup.md)** — Step-by-step lightweight local environment setup.
- 🤖 **[AutoML & Training Engine Guide](docs/guides/automl.md)** — Hyperparameter search space, model routing logic, and stacking ensemble details.
- 🧪 **[End-to-End Testing Guide](docs/guides/e2e_testing.md)** — Running & extending the multi-dataset integration test suite.
- 📥 **[Adding New Datasets Guide](docs/guides/adding_datasets.md)** — How to register raw datasets into the catalog.

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.
