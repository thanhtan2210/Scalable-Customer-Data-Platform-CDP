# Scalable Customer Data Platform (CDP) Architecture

This document presents the architecture design and end-to-end data flow of the Scalable Customer Data Platform (CDP).

## System Architecture Diagram

The system consists of a modular FastAPI backend, PostgreSQL database, object storage (MinIO/Cloudflare R2), MLflow tracking registry, and a Streamlit analytics dashboard.

```mermaid
graph TD
    A[User/Client] -->|POST /datasets/upload| B[FastAPI Backend]
    B -->|Upload raw file <50MB| C[(Cloudflare R2 / MinIO)]
    B -->|Write metadata| D[(PostgreSQL)]
    
    A -->|POST /datasets/{id}/profile| B
    B -->|Load file| C
    B -->|Layer 1: Stats| E[Profiling Engine]
    E -->|Layer 2: Semantic regex| E
    E -->|Layer 3: LLM optional| E
    E -->|Pandera schema gen| F[schema_gen.py]
    
    A -->|POST /jobs/{id}/train| B
    B -->|Optuna AutoML or MTL| G[Training Engine]
    G -->|Log params/metrics/model| H[(MLflow / DagsHub)]
    G -->|Update job.roc_auc| D
    
    A -->|POST /predict| B
    B -->|ModelCache TTL 10min| I[Serving Layer]
    I -->|Load from MLflow| H
    I -->|Drift detection KS+PSI| J[Drift Monitor]
    J -->|Auto-retrain trigger| G
```

## Modular Components

### 1. Ingestion Layer (`core/ingestion/`)
Handles parsing of multiple file formats (CSV, TSV, Parquet, JSON, ODS, Excel) into pandas DataFrames. Enforces file size limitations (<50MB) and filename sanitization.

### 2. Profiling Engine (`core/profiler/`)
Calculates column-level statistical characteristics (entropy, cardinality, skewness, null percentages) and uses rule-based regex templates (and optional LLM routing) to infer column semantic roles (e.g., ID, TARGET, CATEGORICAL, NUMERIC, DATETIME). Flags potential target leakage (correlation >0.95).

### 3. Pipeline & Schema Generation (`core/pipeline/`)
Generates structural validation constraints automatically utilizing **Pandera schemas** and builds scikit-learn preprocessing pipelines mapping specific transformations (e.g., TF-IDF, One-Hot Encoding, Median Imputation) to semantic roles.

### 4. Training Engine (`core/training/`)
Orchestrates training utilizing:
- **Standard AutoML**: Optuna hyperparameter optimization of scikit-learn classifiers (XGBoost, Random Forest, Logistic Regression) evaluated on 5-fold Stratified Cross-Validation.
- **Multi-Task Learning (MTL)**: PyTorch multi-task network sharing a backbone to solve binary customer churn prediction and continuous Customer Performance Index (CPI) regression simultaneously.

Logs execution, params, artifacts, metrics, and models directly to **MLflow**.

### 5. Serving & MLOps (`core/serving/`)
Serves model predictions using FastAPI with thread-safe model caching (Double-Checked Locking, 10-minute TTL). Includes automated data drift detection (Kolmogorov-Smirnov, Population Stability Index, Chi-squared) triggering auto-retraining.
