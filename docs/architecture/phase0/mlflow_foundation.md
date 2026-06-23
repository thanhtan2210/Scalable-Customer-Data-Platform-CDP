# Phase 0: MLflow Foundation & MLOps Infrastructure

**Date:** June 2026
**Scope:** Foundation & Infrastructure (Remote Tracking)

This document defines the foundational infrastructure for experiment tracking and model management using **DagsHub MLflow**. By using a remote server, we eliminate the need to run heavy MLflow containers locally, saving CPU and RAM for development.

## 1. Core Objectives of MLOps Infrastructure

In Phase 0, the MLOps layer must ensure that every model trained is reproducible, versioned, and accessible to both the API and the Analytics Hub.

- **Zero-Local Footprint:** No MLflow server or database runs on the host machine. All tracking metadata and artifacts are stored in the Cloud.
- **Centralized Registry:** Serves as the Single Source of Truth (SSOT) for which model version is currently in "Staging" or "Production".
- **Domain-Agnostic Experimentation:** Experiments are organized by `dataset_id` to ensure isolation between different industry datasets.

## 2. Infrastructure Setup (DagsHub)

We utilize **DagsHub** as the managed MLflow provider because it is free for personal projects and integrates seamlessly with Git.

### Connection Parameters
- **Tracking URI:** `https://dagshub.com/{username}/{repo_name}.mlflow`
- **Authentication:** Basic Auth via DagsHub Username and API Token.
- **Artifact Storage:** Managed by DagsHub (S3-compatible underlying storage).

## 3. Organizational Strategy

To align with the domain-agnostic requirement of the Universal AutoML CDP, tracking is structured as follows:

- **Experiments:** Each industries' run or major AutoML sweep is logged under an Experiment named after the `{dataset_id}`.
- **Model Naming:** The registered model will be named `universal_churn_model`. Different industries will be distinguished by **Tags** (e.g., `industry: telco`) and **Versions**.

## 4. Environment Configuration

The infrastructure is activated by injecting the following variables into the environment:

```env
MLFLOW_TRACKING_URI=https://dagshub.com/...
MLFLOW_TRACKING_USERNAME=...
MLFLOW_TRACKING_PASSWORD=... # DagsHub Token
```

## 5. Integration Strategy

The platform uses a singleton `ModelRegistryClient` (or standard `mlflow` library calls configured via `config.py`).

**Key Lifecycle Foundation:**
1. **Initialize:** `mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)`
2. **Log:** `mlflow.log_params()`, `mlflow.log_metrics()`, and `mlflow.sklearn.log_model()`.
3. **Register:** Automatically register the best model from the AutoML sweep.
4. **Transition:** Moving versions between `None` -> `Staging` -> `Production`.

## 6. Local Fallback (Offline Mode)

If `MLFLOW_TRACKING_URI` is not provided, the system defaults to a local SQLite-backed tracking server (`mlflow.db`) and local artifact storage (`mlruns/`). This is used strictly for development without internet access.
