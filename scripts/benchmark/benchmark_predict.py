"""
Self-contained latency and performance benchmark script for POST /predict endpoint.
Sets up a mock database, overrides model cache with a dummy model, and measures latencies.
"""

import os
import sys
import time
import json
import statistics
from pathlib import Path
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from backend.app.main import app
from backend.app.db.session import get_db
from backend.app.db.models import Base, TrainingJob, Dataset, Profile
from backend.app.core.serving.model_loader import model_cache

# 1. Setup file-based SQLite database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_bench.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Clean up any leftover database file
if os.path.exists("./test_bench.db"):
    try:
        os.remove("./test_bench.db")
    except Exception:
        pass

Base.metadata.create_all(bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db
app.state.limiter.enabled = False  # Disable rate limiter for benchmark accuracy

# 2. Setup Mock Model
class DummyModel:
    def __init__(self):
        self.feature_names_in_ = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", 
            "PhoneService", "InternetService", "MonthlyCharges", "TotalCharges"
        ]
        
    def predict_proba(self, X):
        import numpy as np
        # Return randomized churn probabilities
        n_samples = len(X)
        probs = np.random.uniform(0.1, 0.9, size=(n_samples, 2))
        probs = probs / probs.sum(axis=1, keepdims=True)
        # Add artificial computation latency
        time.sleep(0.012)  # Simulate 12ms inference CPU time
        return probs

# Initialize dummy database state
db = TestingSessionLocal()
dataset_id = "bench_dataset_999"
model_uri = "runs:/bench_run_123/model"

# Save mock dataset and job info in SQLite
mock_dataset = Dataset(
    id=dataset_id, 
    user_id="bench_user",
    filename="telco.csv", 
    r2_path="bench/path/telco.csv", 
    status="uploaded"
)
mock_job = TrainingJob(
    id="job_bench_123",
    dataset_id=dataset_id,
    status="completed",
    target_column="Churn",
    model_uri=model_uri,
    roc_auc=0.935,
    optimal_threshold=0.38,
    is_active=True
)
mock_profile = Profile(
    dataset_id=dataset_id,
    profiles_json=[],
    suggested_target='{"recommended_target": "Churn"}'
)
db.add(mock_dataset)
db.add(mock_job)
db.add(mock_profile)
db.commit()
db.close()

# Put dummy model in warm cache directly
model_cache._models[model_uri] = {
    "model": DummyModel(),
    "model_type": "sklearn",
    "loaded_at": time.time() + 99999,  # Never expire during benchmark
    "model_uri": model_uri
}

client = TestClient(app)
API_KEY = "test-api-key"
HEADERS = {"X-API-Key": API_KEY}

def run_benchmark():
    print("Starting POST /predict Latency Benchmark...")
    payload = {
        "dataset_id": dataset_id,
        "records": [
            {
                "id": f"cust_{i}",
                "gender": "Male" if i % 2 == 0 else "Female",
                "SeniorCitizen": i % 5 == 0,
                "Partner": "Yes" if i % 3 == 0 else "No",
                "Dependents": "No",
                "tenure": 1 + (i % 72),
                "PhoneService": "Yes",
                "InternetService": "Fiber Optic",
                "MonthlyCharges": 65.0 + (i % 50),
                "TotalCharges": 100.0 + (i * 50)
            } for i in range(10) # 10 records per request
        ]
    }

    # Warmup
    print("Warming up client...")
    for _ in range(5):
        resp = client.post("/api/v1/predict", headers=HEADERS, json=payload)
        assert resp.status_code == 200

    # Measure
    print("Measuring 100 request latencies...")
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        resp = client.post("/api/v1/predict", headers=HEADERS, json=payload)
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0) # in ms

    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(0.95 * len(latencies))]
    p99 = sorted(latencies)[int(0.99 * len(latencies))]
    mean_lat = statistics.mean(latencies)
    rps = 1000.0 / mean_lat

    results = {
        "endpoint": "POST /api/v1/predict",
        "requests": len(latencies),
        "batch_size": len(payload["records"]),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "mean_ms": round(mean_lat, 2),
        "throughput_rps": round(rps, 2)
    }

    print("\nBenchmark Results:")
    print(json.dumps(results, indent=2))

    # Save to reports
    reports_dir = Path(BASE_DIR) / "backend" / "app" / "reports"
    os.makedirs(reports_dir, exist_ok=True)
    report_file = reports_dir / "benchmark_results.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {report_file}")
    
    # Cleanup file database
    try:
        if os.path.exists("./test_bench.db"):
            os.remove("./test_bench.db")
    except Exception:
        pass

if __name__ == "__main__":
    run_benchmark()
