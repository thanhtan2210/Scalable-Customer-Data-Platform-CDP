# End-to-End Dataset Integration Testing Guide

This guide explains how to use `scripts/test_datasets_e2e.py` to run automated end-to-end integration tests across all registered datasets.

---

## 1. Overview of the E2E Test Suite

The E2E test runner validates the entire platform workflow against real tabular datasets. It executes tests across **4 sequential layers**:

```
Layer 1: Upload    ──> POST /api/v1/datasets/upload (File validation & Storage)
Layer 2: Profile   ──> POST /api/v1/datasets/{id}/profile (3-Layer Profiler & Schema)
Layer 3: Train     ──> POST /api/v1/jobs/{id}/train (Async AutoML / Optuna HPO / MLflow)
Layer 4: Predict   ──> POST /api/v1/predict/batch (Batch Inference & Risk Tiering)
```

---

## 2. Command Usage

### Basic Syntax

```powershell
python scripts/test_datasets_e2e.py [OPTIONS]
```

### Options & Flags

| Flag | Default | Description |
|---|---|---|
| `--layer {1,2,3,4}` | `4` | Maximum test layer to execute (1=Upload, 2=Profile, 3=Train, 4=Predict) |
| `--dataset NAME` | `all` | Specific dataset ID (`bank`, `telco`, `ecommerce`, `bank_marketing_full`, or `all`) |
| `--fast` | `False` | Smoke test mode: reduces Optuna HPO to 3 trials / 60 seconds |
| `--base-url URL` | `http://localhost:8000` | Target CDP REST API base URL |

---

## 3. Example Workflows

### 1. Fast Smoke Test (Recommended during active code changes)
Run all 4 layers on the `bank` dataset with fast 3-trial HPO:

```powershell
python scripts/test_datasets_e2e.py --layer 4 --dataset bank --fast
```

### 2. Verify Profiler Engine Only
Validate file upload and 3-layer semantic profiling across ALL catalog datasets:

```powershell
python scripts/test_datasets_e2e.py --layer 2 --dataset all
```

### 3. Full Production Training Benchmark
Run full 100-trial HPO training & batch prediction across all verified datasets:

```powershell
python scripts/test_datasets_e2e.py --layer 4 --dataset all
```

---

## 4. Test Suite Output Interpretation

Sample successful test run:

```
============================================================
  CDP END-TO-END DATASET TEST SUITE
============================================================
  Base URL  : http://localhost:8000
  Max Layer : 4 (Predict)
  Datasets  : bank

  [OK] Backend is up at http://localhost:8000

============================================================
  Dataset: BANK — Bank Customer Churn (Churn Modelling)
============================================================
  [STEP 1] Upload
  [OK] Uploaded -> dataset_id=7cc700d6-7fee-4303-9c1c-1a95ba5c3d83
  [INFO] rows=10000  cols=14  format=csv

  [STEP 2] Profile dataset
  [OK] Profiling complete
  [INFO] Suggested target    : Exited

  [STEP 3] Train AutoML
  [OK] Training completed -> status=completed  duration=45.2s
  [INFO] Best ROC AUC        : 0.8324
  [INFO] Model URI           : runs:/a1b2c3d4/model

  [STEP 4] Batch Predict
  [OK] Batch prediction completed -> total_predictions=10000
  [INFO] Risk Distribution   : HIGH=2037  MEDIUM=1200  LOW=6763
```

---

## 5. Troubleshooting Common Failures

- **Connection Refused (`[WinError 10061]`)**: Backend server is not running. Start it with `uvicorn backend.app.main:app --port 8000`.
- **Job Status `FAILED`**: Inspect `error_message` in the output or inspect backend server logs for the exact traceback.
- **Dataset File Not Found**: Verify dataset data files exist under `data/dataset/`.
