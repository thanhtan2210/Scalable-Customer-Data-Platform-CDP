# Scalable Customer Data Platform (CDP)

[![CDP Pipeline CI](https://github.com/thanhtan2210/Scalable-Customer-Data-Platform-CDP/actions/workflows/ci.yml/badge.svg)](https://github.com/thanhtan2210/Scalable-Customer-Data-Platform-CDP/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/thanhtan2210/Scalable-Customer-Data-Platform-CDP/graph/badge.svg?token=YOUR_TOKEN_HERE)](https://codecov.io/gh/thanhtan2210/Scalable-Customer-Data-Platform-CDP)

A comprehensive platform designed for large-scale customer data processing (target: 50M users), catering to Data Engineering (DE), Data Science (DS), and MLOps requirements.

## 🌟 Key Features
- **Unified CLI**: Manage the entire project lifecycle via a single `launcher.py` script.
- **Hybrid Execution**: Supports full orchestration via Docker or a standalone **Local Mode** for environments without S3/MinIO infrastructure.
- **Robust ETL**: Automated data cleaning, schema normalization, and Data Quality (SLA) validation.
- **ML Lifecycle**: Experiment tracking with MLflow and model serving via FastAPI.
- **Business Dashboard**: Interactive Streamlit interface with detailed insights and data visualizations.

---

## 🚀 Quick Start

### 1. Environment Setup
- Python 3.9+
- Install dependencies: `pip install -r requirements.txt`

### 2. Infrastructure (Optional)
If Docker is available: `docker-compose up -d`
*(The project automatically falls back to local Parquet storage if Docker/MinIO is not detected).*

### 3. Execute Pipeline & Training
```powershell
# Run Data Pipeline (Cleaning & Normalization)
python launcher.py pipeline

# Train Model & Log results to MLflow
python launcher.py train
```

### 4. Launch Services
The project uses specific default ports to avoid system conflicts:
- **Churn API**: `http://localhost:8001/docs`
- **A/B Testing Service**: `http://localhost:8081/docs`
- **MLflow UI**: `http://localhost:5000`
- **Sales Dashboard**: `http://localhost:8501`

```powershell
python launcher.py churn-api
python launcher.py ab-service
python launcher.py dashboard
```

---

## 🏗 Architecture & Data Flow

The platform is designed around a modular microservice architecture. For a detailed component description and interactive Mermaid flowcharts, please refer to the [System Architecture Document](docs/architecture/ARCHITECTURE.md).

### Core Flow:
1. **Ingest & Validate**: Clients upload files → Ingestion parses formats → Pandera validates schema boundaries.
2. **Profile & Synthesize**: Statistics & semantic profiling identify types/leakage. Target synthesizer builds Customer Performance Index (CPI).
3. **AutoML & Train**: Optuna tunes model candidates (XGBoost/RF/LogReg) or PyTorch MTL train is triggered. Results logged to MLflow.
4. **Serve & Monitor**: Model cache holds model instances (10-min TTL). Drift detector processes inference logs and schedules auto-retraining.

---

## 🛠 Project Structure
- `backend/app/core/etl/`: Logic for data cleaning, normalization, lineage, and SLA checks.
- `backend/app/core/profiler/`: Statistical, semantic, and LLM-assisted metadata profiling.
- `backend/app/core/training/`: AutoML (Optuna) and Multi-Task Learning (PyTorch) pipelines.
- `backend/app/core/serving/`: Prediction cache, model versioning, drift checks, and A/B testing.
- `backend/app/api/`: FastAPI REST endpoints (`/datasets`, `/jobs`, `/predict`).
- `analytics/`: Streamlit business intelligence dashboard.
- `data/parquet/processed/`: The "Single Source of Truth" storage for processed data.
- `tests/`: 153+ automated tests (unit, integration, and E2E).

---

## 📊 Data Description
The sample Telco Churn dataset includes 7,043 records with key features:
- **CustomerID**: Unique identifier.
- **Tenure**: Months with the company.
- **MonthlyCharges**: Normalized monthly fee.
- **TotalCharges**: Cumulative charges.
- **Churn**: 1 (Churned), 0 (Retained).
- **CLTV**: Customer Lifetime Value.

---

## 📈 Business Impact (Offline Evaluation)
Based on `reports/offline_evaluation.json`:
- **Model Accuracy**: ~93.5%
- **ROC-AUC**: 0.99
- **Estimated ROI**: 263.09 (Based on a 30% outreach success rate and $5/contact cost).

### ⚡ API Latency Benchmark (Local SQLite & Cache)
Based on `backend/app/reports/benchmark_results.json` (measured over 100 requests of batch size 10):
- **p50 Latency**: 400.19 ms
- **p95 Latency**: 632.17 ms
- **p99 Latency**: 771.45 ms
- **Mean Latency**: 426.15 ms
- **Throughput**: 2.35 requests/second

---

## 📝 Technical Notes (Troubleshooting)
1. **SLA Validation**: The `Churn Reason` column has a high null rate (~73%), which is expected for customers who haven't churned. The system is configured to accept this.
2. **Local Fallback**: If MinIO connection fails, the Dashboard automatically reads from `data/parquet/processed/cleaned_telco.parquet`.
3. **MLflow**: If not using Docker, start the local server with:
   `mlflow server --backend-store-uri file:///./mlruns --port 5000`
