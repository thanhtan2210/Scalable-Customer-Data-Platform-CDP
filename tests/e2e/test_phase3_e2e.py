import os
import time
import pytest
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")
HEADERS = {"X-API-Key": os.getenv("API_KEY", "test-api-key")}
FILE_PATH = "data/raw/cleaned_telco.csv"

# Shared state between sequential steps
state = {
    "dataset_id": None,
    "suggested_target": None,
    "profiles": None,
    "composite_target": None,
    "job_id": None,
    "optimal_threshold": None
}

def test_01_upload():
    assert os.path.exists(FILE_PATH), f"File {FILE_PATH} does not exist."
    with open(FILE_PATH, "rb") as f:
        files = {"file": (os.path.basename(FILE_PATH), f, "text/csv")}
        resp = requests.post(f"{API_URL}/datasets/upload", headers=HEADERS, files=files)
        
    assert resp.status_code == 200, f"Upload failed: {resp.text}"
    data = resp.json()
    print("UPLOAD_RESPONSE_DATA:", data)
    assert "dataset_id" in data
    assert data["status"] == "uploaded"
    assert data["detected_format"] == "csv"
    assert "row_count" in data
    assert "col_count" in data
    
    state["dataset_id"] = data["dataset_id"]

def test_02_profile():
    dataset_id = state["dataset_id"]
    assert dataset_id is not None, "Dataset ID is missing."
    
    resp = requests.post(f"{API_URL}/datasets/{dataset_id}/profile", headers=HEADERS)
    assert resp.status_code == 200, f"Profiling failed: {resp.text}"
    
    data = resp.json()
    assert "suggested_target" in data
    assert "candidate_targets" in data
    assert "profiles" in data
    
    suggested = data["suggested_target"]
    assert suggested != "", "suggested_target is empty"

    target_profile = next((p for p in data["profiles"] if p["name"] == suggested), None)
    assert target_profile is not None, f"suggested_target '{suggested}' not in profiles"
    assert target_profile["inferred_role"] in ["TARGET", "CATEGORICAL"], f"Bad target role: {target_profile['inferred_role']}"
    
    state["suggested_target"] = suggested
    state["profiles"] = data["profiles"]
    state["composite_target"] = data.get("composite_target")

def test_03_train():
    dataset_id = state["dataset_id"]
    suggested_target = state["suggested_target"]
    profiles = state["profiles"]
    composite_target = state["composite_target"]
    
    # Exclude leakage suspect columns by setting DataRole.IGNORE before training to ensure realistic E2E model
    for p in profiles:
        if p.get("potential_leakage"):
            p["inferred_role"] = "IGNORE"

    payload = {
        "confirmed_target": suggested_target,
        "confirmed_profiles": profiles,
        "composite_config": composite_target,
        "prior_model_uri": None
    }
    headers = HEADERS.copy()
    headers["Content-Type"] = "application/json"
    
    resp = requests.post(f"{API_URL}/jobs/datasets/{dataset_id}/train", headers=headers, json=payload)
    assert resp.status_code == 200, f"Train failed: {resp.text}"
    
    data = resp.json()
    assert "job_id" in data
    assert "status" in data
    
    state["job_id"] = data["job_id"]

def test_04_poll_status():
    job_id = state["job_id"]
    assert job_id is not None, "Job ID is missing."
    
    start_time = time.time()
    max_wait = 300  # 5 minutes
    completed = False
    
    while time.time() - start_time < max_wait:
        resp = requests.get(f"{API_URL}/jobs/{job_id}/status", headers=HEADERS)
        assert resp.status_code == 200, f"Get status failed: {resp.text}"
        data = resp.json()
        status = data.get("status")
        
        if status == "completed":
            assert data["roc_auc"] > 0.6, f"Low ROC AUC: {data['roc_auc']}"
            assert data["model_uri"] not in [None, ""], "Empty model_uri"
            assert data["optimal_threshold"] is not None
            assert 0.0 < data["optimal_threshold"] < 1.0
            
            state["optimal_threshold"] = data["optimal_threshold"]
            completed = True
            break
        elif status == "failed":
            pytest.fail(f"Job failed: {data.get('error_message')}")
            
        time.sleep(2)
        
    assert completed, "Polling timed out."

def test_05_idempotency():
    dataset_id = state["dataset_id"]
    job_id = state["job_id"]
    suggested_target = state["suggested_target"]
    profiles = state["profiles"]
    composite_target = state["composite_target"]
    
    payload = {
        "confirmed_target": suggested_target,
        "confirmed_profiles": profiles,
        "composite_config": composite_target,
        "prior_model_uri": None
    }
    headers = HEADERS.copy()
    headers["Content-Type"] = "application/json"
    
    resp = requests.post(f"{API_URL}/jobs/datasets/{dataset_id}/train", headers=headers, json=payload)
    assert resp.status_code == 200, f"Idempotency check failed: {resp.text}"
    data = resp.json()
    assert data["job_id"] == job_id, f"Different job ID returned: {data['job_id']} != {job_id}"

def test_06_batch_predict():
    dataset_id = state["dataset_id"]
    r2_processed_path = f"raw/default_user/{dataset_id}/cleaned_telco.parquet"
    payload = {
        "dataset_id": dataset_id,
        "file_path": r2_processed_path
    }
    headers = HEADERS.copy()
    headers["Content-Type"] = "application/json"
    
    resp = requests.post(f"{API_URL}/predict/batch", headers=headers, json=payload)
    assert resp.status_code == 200, f"Batch predict failed: {resp.text}"
    
    data = resp.json()
    assert "total_records" in data
    assert "predictions" in data
    
    total = data["total_records"]
    assert total > 0
    assert data["high_risk"] + data["medium_risk"] + data["low_risk"] == total
    assert data["threshold_source"] in ["optimal", "default"]
    assert len(data["predictions"]) == total

def test_07_re_evaluate_leakage():
    dataset_id = state["dataset_id"]
    suggested_target = state["suggested_target"]
    payload = {
        "confirmed_target": suggested_target
    }
    headers = HEADERS.copy()
    headers["Content-Type"] = "application/json"
    
    resp = requests.post(f"{API_URL}/datasets/{dataset_id}/re-evaluate-leakage", headers=headers, json=payload)
    assert resp.status_code == 200, f"Re-evaluate leakage failed: {resp.text}"
    
    data = resp.json()
    assert "profiles_updated_in_db" in data
    assert "updated_profiles" in data
