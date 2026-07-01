import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import io
import json
from backend.app.main import app
from backend.app.db.models import Base, Dataset, Profile, TrainingJob
from backend.app.db.session import get_db
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Mock DB setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_leakage_predict_batch.db"
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
    db.query(Profile).delete()
    db.query(Dataset).delete()
    db.commit()
    db.close()

@patch("backend.app.api.v1.datasets.storage")
def test_re_evaluate_leakage(mock_storage):
    db = TestingSessionLocal()
    
    # 1. Create dataset and profile
    dataset_id = "test-ds-leakage"
    new_dataset = Dataset(
        id=dataset_id,
        user_id="default_user",
        filename="test.csv",
        r2_path=f"raw/default_user/{dataset_id}/test.csv",
        status="uploaded"
    )
    
    # Simple profiles_json, churn has status=TARGET
    profiles = [
        {
            "name": "feat1",
            "inferred_dtype": "float64",
            "inferred_role": "NUMERIC",
            "confidence_score": 1.0,
            "null_pct": 0.0,
            "unique_count": 3,
            "entropy": 1.0,
            "potential_leakage": False,
            "transform_strategy": "standard",
            "impute_strategy": "median"
        },
        {
            "name": "old_target",
            "inferred_dtype": "float64",
            "inferred_role": "TARGET",
            "confidence_score": 1.0,
            "null_pct": 0.0,
            "unique_count": 2,
            "entropy": 0.5,
            "potential_leakage": False,
            "transform_strategy": "passthrough",
            "impute_strategy": "drop"
        },
        {
            "name": "new_target",
            "inferred_dtype": "float64",
            "inferred_role": "NUMERIC",
            "confidence_score": 1.0,
            "null_pct": 0.0,
            "unique_count": 2,
            "entropy": 0.5,
            "potential_leakage": False,
            "transform_strategy": "standard",
            "impute_strategy": "median"
        }
    ]
    
    new_profile = Profile(
        dataset_id=dataset_id,
        profiles_json=profiles,
        suggested_target="{}"
    )
    
    db.add(new_dataset)
    db.add(new_profile)
    db.commit()
    
    # Mock R2 data
    df = pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0],
        "old_target": [1.0, 0.0, 1.0],
        "new_target": [0.0, 1.0, 0.0]
    })
    mock_storage.download_file.return_value = df.to_csv(index=False).encode("utf-8")
    
    # Call re-evaluate leakage
    payload = {"confirmed_target": "new_target"}
    resp = client.post(f"/api/v1/datasets/{dataset_id}/re-evaluate-leakage", json=payload, headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    
    # Check that roles have switched
    new_target_profile = next(p for p in data if p["name"] == "new_target")
    old_target_profile = next(p for p in data if p["name"] == "old_target")
    
    assert new_target_profile["inferred_role"] == "TARGET"
    assert old_target_profile["inferred_role"] == "NUMERIC"
    assert old_target_profile["transform_strategy"] == "standard"
    assert old_target_profile["impute_strategy"] == "median"

@patch("backend.app.api.v1.predict.storage")
@patch("backend.app.api.v1.predict.model_cache")
@patch("mlflow.tracking.MlflowClient")
def test_predict_batch(mock_mlflow_client, mock_model_cache, mock_storage):
    db = TestingSessionLocal()
    
    # 1. Create dataset and completed training job
    dataset_id = "test-ds-predict"
    new_dataset = Dataset(
        id=dataset_id,
        user_id="default_user",
        filename="test.parquet",
        r2_path=f"raw/default_user/{dataset_id}/test.parquet",
        status="completed"
    )
    
    new_job = TrainingJob(
        id="job-123",
        dataset_id=dataset_id,
        status="completed",
        model_uri="runs:/mock-run-id/model",
        target_column="churn",
        optimal_threshold=0.35,
        roc_auc=0.78
    )
    
    db.add(new_dataset)
    db.add(new_job)
    db.commit()
    
    # 2. Mock R2 parquet data
    df = pd.DataFrame({
        "customerID": ["c1", "c2", "c3"],
        "feat1": [1.0, 2.0, 3.0],
        "churn": [0, 1, 0]
    })
    
    pq_buffer = io.BytesIO()
    df.to_parquet(pq_buffer, index=False)
    mock_storage.download_file.return_value = pq_buffer.getvalue()
    
    # 3. Mock cached model and predict_proba returning probabilities
    mock_model = MagicMock()
    import numpy as np
    mock_model.predict_proba.return_value = np.array([
        [0.9, 0.1],  # c1 -> Low (0.1 < 0.35)
        [0.6, 0.4],  # c2 -> High (0.4 >= 0.35)
        [0.8, 0.2]   # c3 -> Medium (0.2 >= 0.175)
    ])
    mock_model_cache.get_model.return_value = mock_model
    
    # Mock MLflow threshold artifact download
    mock_mlflow_client.return_value.download_artifacts.side_effect = Exception("Artifact not found")
    
    # 4. Call batch predict
    payload = {
        "dataset_id": dataset_id,
        "file_path": f"processed/{dataset_id}/with_cpi.parquet"
    }
    
    resp = client.post("/api/v1/predict/batch", json=payload, headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    
    assert data["total_records"] == 3
    assert data["high_risk"] == 1
    assert data["medium_risk"] == 1
    assert data["low_risk"] == 1
    assert data["threshold_used"] == 0.35
    
    # Check individual risk mappings
    preds = data["predictions"]
    c1_pred = next(p for p in preds if p["record_id"] == "c1")
    c2_pred = next(p for p in preds if p["record_id"] == "c2")
    c3_pred = next(p for p in preds if p["record_id"] == "c3")
    
    assert c1_pred["risk_level"] == "Low"
    assert c2_pred["risk_level"] == "High"
    assert c3_pred["risk_level"] == "Medium"
