import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
from backend.app.main import app
from backend.app.db.models import Base, Dataset, TrainingJob
from backend.app.db.session import get_db
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Mock DB setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_predict_jobs_status.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

@pytest.fixture(autouse=True)
def setup_db_override():
    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.clear()

client = TestClient(app)
API_KEY = "test-api-key"
HEADERS = {"X-API-Key": API_KEY}

@pytest.fixture(autouse=True)
def clean_db():
    db = TestingSessionLocal()
    db.query(TrainingJob).delete()
    db.query(Dataset).delete()
    db.commit()
    db.close()

def test_get_job_status_not_found():
    response = client.get("/api/v1/jobs/non-existent-job-id/status", headers=HEADERS)
    assert response.status_code == 404
    assert "Job not found" in response.json()["detail"]

def test_get_job_status_queued_and_training():
    db = TestingSessionLocal()
    
    # Create dataset
    dataset_id = "test-ds-status"
    new_dataset = Dataset(
        id=dataset_id,
        user_id="default_user",
        filename="test.csv",
        r2_path=f"raw/default_user/{dataset_id}/test.csv",
        status="uploaded"
    )
    
    # Create job
    job_id = "test-job-status-id"
    new_job = TrainingJob(
        id=job_id,
        dataset_id=dataset_id,
        status="queued",
        target_column="churn"
    )
    db.add(new_dataset)
    db.add(new_job)
    db.commit()
    
    # Check status is queued
    response = client.get(f"/api/v1/jobs/{job_id}/status", headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    
    # Update to training
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    job.status = "training"
    db.commit()
    
    # Check status is training
    response = client.get(f"/api/v1/jobs/{job_id}/status", headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["status"] == "training"

def test_single_predict_missing_dataset_job():
    payload = {
        "dataset_id": "missing-ds",
        "records": [{"feat1": 1.0, "feat2": 2.0}]
    }
    response = client.post("/api/v1/predict", json=payload, headers=HEADERS)
    assert response.status_code == 404
    assert "No completed training job found" in response.json()["detail"]

@patch("backend.app.api.v1.predict.model_cache")
def test_single_predict_happy_path(mock_model_cache):
    db = TestingSessionLocal()
    
    # Create dataset and completed job
    dataset_id = "test-ds-predict-single"
    new_dataset = Dataset(
        id=dataset_id,
        user_id="default_user",
        filename="test.csv",
        r2_path=f"raw/default_user/{dataset_id}/test.csv",
        status="completed"
    )
    new_job = TrainingJob(
        id="job-single-123",
        dataset_id=dataset_id,
        status="completed",
        model_uri="runs:/mock-run/model",
        target_column="churn",
        roc_auc=0.85
    )
    db.add(new_dataset)
    db.add(new_job)
    db.commit()
    
    # Mock model
    import numpy as np
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([
        [0.2, 0.8],  # record 1 -> High (0.8 > 0.7)
        [0.8, 0.2]   # record 2 -> Low (0.2 < 0.4)
    ])
    mock_model_cache.get_model.return_value = mock_model
    
    # Call single predict
    payload = {
        "dataset_id": dataset_id,
        "records": [
            {"id": "rec-1", "feat1": 10.0},
            {"id": "rec-2", "feat1": 2.0}
        ]
    }
    response = client.post("/api/v1/predict", json=payload, headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    
    predictions = data["predictions"]
    assert len(predictions) == 2
    
    p1 = next(p for p in predictions if p["record_id"] == "rec-1")
    p2 = next(p for p in predictions if p["record_id"] == "rec-2")
    
    assert p1["churn_probability"] == 0.8
    assert p1["risk_level"] == "High"
    
    assert p2["churn_probability"] == 0.2
    assert p2["risk_level"] == "Low"

@patch("backend.app.api.v1.predict.model_cache")
def test_single_predict_inference_failure(mock_model_cache):
    db = TestingSessionLocal()
    
    # Create dataset and completed job
    dataset_id = "test-ds-predict-fail"
    new_dataset = Dataset(
        id=dataset_id,
        user_id="default_user",
        filename="test.csv",
        r2_path=f"raw/default_user/{dataset_id}/test.csv",
        status="completed"
    )
    new_job = TrainingJob(
        id="job-single-fail-123",
        dataset_id=dataset_id,
        status="completed",
        model_uri="runs:/mock-run/model",
        target_column="churn",
        roc_auc=0.85
    )
    db.add(new_dataset)
    db.add(new_job)
    db.commit()
    
    # Mock model to throw exception
    mock_model = MagicMock()
    mock_model.predict_proba.side_effect = Exception("Model weights load error")
    mock_model_cache.get_model.return_value = mock_model
    
    # Call single predict
    payload = {
        "dataset_id": dataset_id,
        "records": [{"feat1": 1.0}]
    }
    response = client.post("/api/v1/predict", json=payload, headers=HEADERS)
    assert response.status_code == 500
    assert "Inference failed: Model weights load error" in response.json()["detail"]
