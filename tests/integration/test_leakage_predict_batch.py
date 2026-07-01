import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import io
import json
from backend.app.main import app
from backend.app.db.models import Base, Dataset, Profile, TrainingJob
from backend.app.db.session import get_db
from backend.app.core.profiler.column_profile import ColumnProfile
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
    
    preds = data["predictions"]
    c1_pred = next(p for p in preds if p["record_id"] == "c1")
    c2_pred = next(p for p in preds if p["record_id"] == "c2")
    c3_pred = next(p for p in preds if p["record_id"] == "c3")
    
    assert c1_pred["risk_level"] == "Low"
    assert c2_pred["risk_level"] == "High"
    assert c3_pred["risk_level"] == "Medium"

@patch("backend.app.api.v1.datasets.run_profiling")
@patch("backend.app.api.v1.datasets.storage")
def test_profile_endpoint_success(mock_storage, mock_run_profiling):
    db = TestingSessionLocal()
    
    dataset_id = "test-ds-profile"
    new_dataset = Dataset(
        id=dataset_id,
        user_id="default_user",
        filename="test.csv",
        r2_path=f"raw/default_user/{dataset_id}/test.csv",
        status="uploaded"
    )
    db.add(new_dataset)
    db.commit()
    
    df = pd.DataFrame({"feat1": [1.0, 2.0, 3.0], "churn": [0, 1, 0]})
    mock_storage.download_file.return_value = df.to_csv(index=False).encode("utf-8")
    
    mock_profiles = [
        ColumnProfile(
            name="feat1",
            inferred_dtype="float64",
            inferred_role="NUMERIC",
            confidence_score=1.0,
            null_pct=0.0,
            unique_count=3,
            entropy=1.0,
            transform_strategy="standard",
            impute_strategy="median"
        ),
        ColumnProfile(
            name="churn",
            inferred_dtype="int64",
            inferred_role="TARGET",
            confidence_score=1.0,
            null_pct=0.0,
            unique_count=2,
            entropy=0.5,
            transform_strategy="passthrough",
            impute_strategy="drop"
        )
    ]
    
    from backend.app.core.profiler.target_analysis import TargetAnalysis, CandidateTarget, TargetSignals, TargetRole
    mock_target_analysis = TargetAnalysis(
        recommended_target="churn",
        candidate_targets=[
            CandidateTarget(
                name="churn",
                rank=1,
                score=0.9,
                signals=TargetSignals(
                    is_binary=True,
                    entropy=0.5,
                    entropy_score=0.9,
                    keyword_match=True,
                    position_bonus=0.0
                ),
                suggested_role=TargetRole.TARGET
            )
        ],
        churn_column_group=[],
        recommended_auxiliary=[],
        leakage_suspects=["leak_col"],
        composite_target=None
    )
    
    mock_run_profiling.return_value = (mock_profiles, mock_target_analysis)
    
    resp = client.post(f"/api/v1/datasets/{dataset_id}/profile", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    
    assert data["dataset_id"] == dataset_id
    assert data["suggested_target"] == "churn"
    assert len(data["profiles"]) == 2
    assert "leak_col" in data["leakage_suspects"]

def test_start_training_missing_dataset():
    payload = {
        "confirmed_target": "churn",
        "confirmed_profiles": [],
        "composite_config": None,
        "prior_model_uri": None
    }
    response = client.post("/api/v1/jobs/datasets/non-existent-ds/train", json=payload, headers=HEADERS)
    assert response.status_code == 404
    assert "Dataset not found" in response.json()["detail"]

@patch("backend.app.api.v1.jobs.storage")
def test_start_training_missing_target_column(mock_storage):
    db = TestingSessionLocal()
    dataset_id = "test-ds-no-col"
    new_dataset = Dataset(
        id=dataset_id,
        user_id="default_user",
        filename="test.csv",
        r2_path=f"raw/default_user/{dataset_id}/test.csv",
        status="uploaded"
    )
    db.add(new_dataset)
    db.commit()
    
    df = pd.DataFrame({"feat1": [1, 2, 3]})
    mock_storage.download_file.return_value = df.to_csv(index=False).encode("utf-8")
    
    payload = {
        "confirmed_target": "non_existent_target",
        "confirmed_profiles": [],
        "composite_config": None,
        "prior_model_uri": None
    }
    response = client.post(f"/api/v1/jobs/datasets/{dataset_id}/train", json=payload, headers=HEADERS)
    assert response.status_code == 400
    assert "Confirmed target 'non_existent_target' not found" in response.json()["detail"]

def test_re_evaluate_leakage_missing_dataset():
    payload = {"confirmed_target": "churn"}
    response = client.post("/api/v1/datasets/non-existent-ds/re-evaluate-leakage", json=payload, headers=HEADERS)
    assert response.status_code == 404
    assert "Dataset not found" in response.json()["detail"]

def test_re_evaluate_leakage_missing_profile():
    db = TestingSessionLocal()
    dataset_id = "test-ds-no-prof"
    new_dataset = Dataset(
        id=dataset_id,
        user_id="default_user",
        filename="test.csv",
        r2_path=f"raw/default_user/{dataset_id}/test.csv",
        status="uploaded"
    )
    db.add(new_dataset)
    db.commit()
    
    payload = {"confirmed_target": "churn"}
    response = client.post(f"/api/v1/datasets/{dataset_id}/re-evaluate-leakage", json=payload, headers=HEADERS)
    assert response.status_code == 404
    assert "Profile not found" in response.json()["detail"]

def test_predict_batch_no_completed_job():
    db = TestingSessionLocal()
    dataset_id = "test-ds-no-job"
    new_dataset = Dataset(
        id=dataset_id,
        user_id="default_user",
        filename="test.csv",
        r2_path=f"raw/default_user/{dataset_id}/test.csv",
        status="uploaded"
    )
    db.add(new_dataset)
    db.commit()
    
    payload = {
        "dataset_id": dataset_id,
        "file_path": "processed/test.parquet"
    }
    response = client.post("/api/v1/predict/batch", json=payload, headers=HEADERS)
    assert response.status_code == 404
    assert "No completed training job found" in response.json()["detail"]
