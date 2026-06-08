import requests
import time
import os
import json
import pandas as pd
import io

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/api/v1")
API_KEY = os.getenv("API_KEY", "test-api-key")
HEADERS = {"X-API-Key": API_KEY}

DATASETS = {
    "telco": {
        "file": "data/raw/cleaned_telco.csv",
        "target": "Churn"
    },
    "bank": {
        "url": "https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Bank%20Churn%20Modelling.csv",
        "target": "Exited"
    },
    "hr": {
        "url": "https://raw.githubusercontent.com/aditya-bhatt/HR-Analytics-Employee-Attrition-Prediction/master/HR-Employee-Attrition.csv",
        "target": "Attrition"
    }
}

def run_e2e_for_dataset(name, config):
    print(f"🚀 Starting E2E test for: {name}")
    
    # 1. Prepare Data
    if "url" in config:
        df = pd.read_csv(config["url"])
    else:
        df = pd.read_csv(config["file"])
    
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    # 2. Upload
    files = {"file": (f"{name}.csv", csv_buffer, "text/csv")}
    resp = requests.post(f"{BASE_URL}/datasets/upload", files=files, headers=HEADERS)
    dataset_id = resp.json()["dataset_id"]
    print(f"✅ Uploaded. Dataset ID: {dataset_id}")

    # 3. Profile
    resp = requests.post(f"{BASE_URL}/datasets/{dataset_id}/profile", headers=HEADERS)
    profile_data = resp.json()
    print(f"✅ Profiled. Suggested Target: {profile_data['suggested_target']}")

    # 4. Train
    train_payload = {
        "confirmed_target": config["target"],
        "confirmed_profiles": profile_data["profiles"]
    }
    resp = requests.post(f"{BASE_URL}/jobs/datasets/{dataset_id}/train", json=train_payload, headers=HEADERS)
    job_id = resp.json()["job_id"]
    print(f"⏳ Training job {job_id} started...")

    # 5. Poll
    status = "training"
    while status == "training":
        time.sleep(10)
        resp = requests.get(f"{BASE_URL}/jobs/{job_id}/status", headers=HEADERS)
        job_info = resp.json()
        status = job_info["status"]
        print(f"   Status: {status}")

    if status == "completed":
        print(f"🏆 Training Success! ROC-AUC: {job_info['roc_auc']}")
        
        # 6. Predict
        sample_records = df.head(5).to_dict(orient="records")
        predict_payload = {"dataset_id": dataset_id, "records": sample_records}
        resp = requests.post(f"{BASE_URL}/predict", json=predict_payload, headers=HEADERS)
        predictions = resp.json()["predictions"]
        print(f"🔮 Predictions received: {len(predictions)} records")
        
        return {
            "name": name,
            "status": "success",
            "roc_auc": job_info["roc_auc"],
            "dataset_id": dataset_id
        }
    else:
        print(f"❌ Training Failed for {name}")
        return {"name": name, "status": "failed"}

if __name__ == "__main__":
    results = []
    for name, config in DATASETS.items():
        try:
            results.append(run_e2e_for_dataset(name, config))
        except Exception as e:
            print(f"💥 Error testing {name}: {e}")
            results.append({"name": name, "status": "error", "error": str(e)})

    os.makedirs("tests/results", exist_ok=True)
    result_file_path = os.path.join("tests", "results", "e2e_results.json")
    with open(result_file_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📝 All results saved to {result_file_path}")
