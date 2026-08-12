"""
End-to-End Dataset Testing Script
===================================
Chiến thuật test theo 4 layer:
  Layer 1 – Upload & Parse        (test parser + file handling)
  Layer 2 – Profiling             (test profiler + target detection + leakage)
  Layer 3 – AutoML Training       (test full training pipeline)
  Layer 4 – Prediction            (test batch predict từ trained model)

Chạy:
  python scripts/test_datasets_e2e.py --layer 2 --dataset all
  python scripts/test_datasets_e2e.py --layer 3 --dataset bank
  python scripts/test_datasets_e2e.py --layer 3 --dataset bank --fast  # Quick smoke test (3 trials)
  python scripts/test_datasets_e2e.py --layer 4 --dataset telco

Yêu cầu: Backend đang chạy tại BASE_URL (mặc định http://localhost:8000)
"""

import argparse
import time
import sys
import os

# Fix Windows console UTF-8 output encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import requests

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "test-api-key")
TIMEOUT = 120  # Increased: server may be busy during training


def load_catalog(catalog_path="data/dataset/catalog.yaml"):
    if os.path.exists(catalog_path):
        try:
            import yaml
            with open(catalog_path, "r", encoding="utf-8") as f:
                catalog = yaml.safe_load(f)
            datasets = {}
            for item in catalog.get("datasets", []):
                if item.get("status") in ("verified", "pending"):
                    datasets[item["id"]] = {
                        "file": item["file"],
                        "target": item["target"],
                        "description": f"{item['name']} ({item.get('notes', '')})",
                        "id_cols": item.get("id_columns", []),
                        "expected_nulls": item.get("expected_nulls", False),
                    }
            if datasets:
                return datasets
        except Exception as e:
            print(f"Warning: Could not parse {catalog_path}: {e}")

    return {
        "telco": {
            "file": "data/dataset/telco-customer-churn/Telco_Customer_Churn.csv",
            "target": "Churn",
            "description": "Telco Customer Churn (7043 rows, 21 cols)",
            "id_cols": ["customerID"],
            "expected_nulls": True,
        },
        "bank": {
            "file": "data/dataset/bank-customer-churn/Churn_Modelling.csv",
            "target": "Exited",
            "description": "Bank Customer Churn (10000 rows, 14 cols)",
            "id_cols": ["RowNumber", "CustomerId", "Surname"],
            "expected_nulls": False,
        },
        "ecommerce": {
            "file": "data/dataset/ecommerce-churn/E_Commerce_Dataset.csv",
            "target": "Churn",
            "description": "E-Commerce Customer Churn (5630 rows, 20 cols)",
            "id_cols": ["CustomerID"],
            "expected_nulls": True,
        },
    }


DATASETS = load_catalog()

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
SESSION = requests.Session()
SESSION.headers.update({"X-API-Key": API_KEY})


def log(level: str, msg: str):
    icons = {"OK": "[OK]", "FAIL": "[FAIL]", "INFO": "[INFO]", "WARN": "[WARN]", "STEP": "[STEP]"}
    icon = icons.get(level, f"[{level}]")
    try:
        print(f"  {icon} {msg}")
    except Exception:
        print(f"  [{level}] {msg}")


def separator(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def do_post(path: str, **kwargs) -> requests.Response:
    return SESSION.post(f"{BASE_URL}{path}", timeout=TIMEOUT, **kwargs)


def do_get(path: str, **kwargs) -> requests.Response:
    return SESSION.get(f"{BASE_URL}{path}", timeout=TIMEOUT, **kwargs)


# ──────────────────────────────────────────────
# LAYER 1: UPLOAD & PARSE
# ──────────────────────────────────────────────
def test_upload(name: str, cfg: dict):
    """Returns dataset_id on success, None on failure."""
    log("STEP", f"Upload: {cfg['description']}")
    filepath = cfg["file"]

    if not os.path.exists(filepath):
        log("FAIL", f"File not found: {filepath}")
        return None

    with open(filepath, "rb") as f:
        resp = do_post(
            "/api/v1/datasets/upload",
            files={"file": (os.path.basename(filepath), f, "text/csv")},
        )

    if resp.status_code != 200:
        log("FAIL", f"Upload failed: {resp.status_code} — {resp.text[:200]}")
        return None

    data = resp.json()
    dataset_id = data.get("dataset_id")
    r2_path = data.get("r2_path")
    log("OK", f"Uploaded -> dataset_id={dataset_id}")
    log("INFO", f"rows={data.get('row_count')}  cols={data.get('col_count')}  format={data.get('detected_format')}")

    if data.get("requires_sheet_selection"):
        log("WARN", "Requires sheet selection — use CSV version instead")
        return None

    return dataset_id, r2_path


# ──────────────────────────────────────────────
# LAYER 2: PROFILING
# ──────────────────────────────────────────────
def test_profile(name: str, cfg: dict, dataset_id: str):
    log("STEP", f"Profile dataset: {name}")
    resp = do_post(f"/api/v1/datasets/{dataset_id}/profile")

    if resp.status_code != 200:
        log("FAIL", f"Profile failed: {resp.status_code} — {resp.text[:300]}")
        return None

    data = resp.json()
    suggested_target = data.get("suggested_target")
    candidate_targets = data.get("candidate_targets", [])
    leakage_suspects = data.get("leakage_suspects", [])
    composite = data.get("composite_target")
    warnings = data.get("warnings", [])

    log("OK", "Profiling complete")
    log("INFO", f"Suggested target    : {suggested_target}")
    log("INFO", f"Candidate targets   : {candidate_targets}")
    log("INFO", f"Leakage suspects    : {leakage_suspects}")
    log("INFO", f"Composite target    : {composite}")
    log("INFO", f"Column warnings     : {len(warnings)} ({warnings[:2]})" if warnings else "Column warnings : none")

    # Validate target detection
    expected = cfg["target"]
    if suggested_target and expected.lower() in str(suggested_target).lower():
        log("OK", f"Target detection correct: '{suggested_target}' matches expected '{expected}'")
    elif any(expected.lower() in str(t).lower() for t in candidate_targets):
        log("WARN", f"Target '{expected}' in candidates but NOT as suggested_target")
    else:
        log("FAIL", f"Expected target '{expected}' NOT found. Got suggested='{suggested_target}'")

    return data


# ──────────────────────────────────────────────
# LAYER 3: TRAINING
# ──────────────────────────────────────────────
def test_training(name: str, cfg: dict, dataset_id: str, profile_data: dict):
    log("STEP", f"Start AutoML Training: {name}")

    target_col = profile_data.get("suggested_target") or cfg["target"]
    confirmed_profiles = profile_data.get("profiles", [])  # API returns key "profiles"

    if not confirmed_profiles:
        log("FAIL", "No confirmed_profiles returned from profiling step — cannot start training")
        return None

    payload = {
        "confirmed_target": target_col,
        "confirmed_profiles": confirmed_profiles,
    }

    resp = do_post(f"/api/v1/jobs/datasets/{dataset_id}/train", json=payload)

    if resp.status_code != 200:
        log("FAIL", f"Train failed: {resp.status_code} — {resp.text[:300]}")
        return None

    data = resp.json()
    job_id = data.get("job_id")
    log("OK", f"Training started -> job_id={job_id}")
    return job_id


def poll_job(job_id: str, max_wait_sec: int = 600):
    """Poll job until done or timeout."""
    log("STEP", f"Polling job {job_id} (max {max_wait_sec}s)")
    start = time.time()
    poll_interval = 10

    while time.time() - start < max_wait_sec:
        resp = do_get(f"/api/v1/jobs/{job_id}/status")
        if resp.status_code != 200:
            log("FAIL", f"Job poll error: {resp.status_code} — {resp.text[:200]}")
            return None

        job = resp.json()
        status = job.get("status")
        elapsed = int(time.time() - start)
        print(f"    ... [{elapsed}s] status={status}", end="\r")

        if status == "completed":
            print()
            log("OK", f"Job completed in {elapsed}s")
            log("INFO", f"roc_auc={job.get('roc_auc')}  threshold={job.get('optimal_threshold')}")
            log("INFO", f"model_uri={job.get('model_uri', '')[:60]}")
            return job

        if status == "failed":
            print()
            log("FAIL", f"Job failed: {job.get('error_message', 'unknown error')}")
            return None

        time.sleep(poll_interval)

    print()
    log("WARN", f"Timeout after {max_wait_sec}s — job still running")
    return None


# ──────────────────────────────────────────────
# LAYER 4: PREDICTION
# ──────────────────────────────────────────────
def test_prediction(name: str, cfg: dict, dataset_id: str, r2_path: str) -> bool:
    log("STEP", f"Batch Predict: {name}")

    payload = {
        "dataset_id": dataset_id,
        "file_path": r2_path,
    }
    resp = do_post("/api/v1/predict/batch", json=payload)

    if resp.status_code != 200:
        log("FAIL", f"Predict failed: {resp.status_code} — {resp.text[:300]}")
        return False

    data = resp.json()
    log("OK", "Prediction complete")
    log("INFO", f"total records    : {data.get('total_records')}")
    log("INFO", f"high risk count  : {data.get('high_risk')}")
    log("INFO", f"medium risk count: {data.get('medium_risk')}")
    log("INFO", f"low risk count   : {data.get('low_risk')}")
    log("INFO", f"threshold used   : {data.get('threshold_used')}")
    log("INFO", f"model type       : {data.get('model_type')}")
    return True


# ──────────────────────────────────────────────
# HEALTH CHECK
# ──────────────────────────────────────────────
def health_check() -> bool:
    try:
        resp = do_get("/health")
        if resp.status_code == 200:
            log("OK", f"Backend is up at {BASE_URL}")
            return True
    except Exception as e:
        log("FAIL", f"Cannot connect to {BASE_URL}: {e}")
    return False


# ──────────────────────────────────────────────
# MAIN RUNNER
# ──────────────────────────────────────────────
def run_dataset(name: str, cfg: dict, max_layer: int):
    separator(f"Dataset: {name.upper()} — {cfg['description']}")
    results = {"upload": None, "profile": None, "train": None, "predict": None}

    # Layer 1 – Upload
    upload_res = test_upload(name, cfg)
    if not upload_res:
        print("\n  [STOP] Stopped at Layer 1 — Upload failed")
        return results
    dataset_id, r2_path = upload_res
    results["upload"] = True
    if max_layer < 2:
        return results

    # Layer 2 – Profile
    profile_data = test_profile(name, cfg, dataset_id)
    results["profile"] = profile_data is not None
    if not profile_data:
        print("\n  [STOP] Stopped at Layer 2 — Profile failed")
        return results
    if max_layer < 3:
        return results

    # Layer 3 – Training
    job_id = test_training(name, cfg, dataset_id, profile_data)
    results["train"] = job_id is not None
    model_version_id = None
    if job_id:
        job_result = poll_job(job_id, max_wait_sec=900)
        results["train"] = job_result is not None
        # JobStatusResponse returns model_uri, not model_version_id
        model_version_id = job_result.get("model_uri") if job_result else None
    if max_layer < 4 or not model_version_id:
        return results

    # Layer 4 – Prediction
    results["predict"] = test_prediction(name, cfg, dataset_id, r2_path)

    return results


def main():
    parser = argparse.ArgumentParser(description="CDP E2E Dataset Test")
    parser.add_argument(
        "--layer", type=int, default=2, choices=[1, 2, 3, 4],
        help="Max test layer: 1=Upload, 2=Profile, 3=Train, 4=Predict",
    )
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["all"] + list(DATASETS.keys()),
        help="Which dataset to test",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Fast mode: reduce Optuna to 3 trials/60s timeout for quick smoke tests",
    )
    args = parser.parse_args()

    if args.fast:
        os.environ["OPTUNA_N_TRIALS"] = "3"
        os.environ["OPTUNA_TIMEOUT_SECONDS"] = "60"
        print("  [FAST] Optuna set to 3 trials / 60s timeout per model")

    layer_names = {1: "Upload", 2: "Profile", 3: "Train", 4: "Predict"}

    print("\n" + "=" * 60)
    print("  CDP END-TO-END DATASET TEST SUITE")
    print("=" * 60)
    print(f"  Base URL  : {BASE_URL}")
    print(f"  Max Layer : {args.layer} ({layer_names[args.layer]})")
    print(f"  Datasets  : {args.dataset}")
    print()

    if not health_check():
        print("\n[ERROR] Backend is not running. Start with:")
        print("   uvicorn backend.app.main:app --reload --port 8000")
        sys.exit(1)

    datasets_to_test = (
        DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}
    )
    all_results = {}

    for name, cfg in datasets_to_test.items():
        all_results[name] = run_dataset(name, cfg, args.layer)

    # Summary
    separator("SUMMARY")
    keys_order = ["upload", "profile", "train", "predict"][:args.layer]
    header = f"  {'Dataset':<15}" + "".join(f"{'Layer'+str(i):<14}" for i in range(1, args.layer + 1))
    print(header)
    print("  " + "-" * (15 + 14 * args.layer))
    for name, res in all_results.items():
        row = f"  {name:<15}"
        for key in keys_order:
            val = res.get(key)
            icon = "PASS" if val else "FAIL" if val is False else "SKIP"
            row += f"{icon:<14}"
        print(row)

    total_passed = sum(1 for r in all_results.values() for k in keys_order for v in [r.get(k)] if v is True)
    total_ran = sum(1 for r in all_results.values() for k in keys_order if r.get(k) is not None)
    print(f"\n  Result: {total_passed}/{total_ran} checks passed\n")


if __name__ == "__main__":
    main()
