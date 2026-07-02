import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import io
from datetime import datetime
from backend.app.main import app
from backend.app.db.models import Base, Dataset, TrainingJob
from backend.app.db.session import get_db
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Mock DB setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./tests/test_jobs_api.db"
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

@patch("backend.app.api.v1.jobs.storage")
@patch("backend.app.api.v1.jobs.run_automl")
@patch("mlflow.tracking.MlflowClient")
@patch("backend.app.api.v1.jobs.SessionLocal", new=TestingSessionLocal)
def test_start_training_happy_path(mock_mlflow_client, mock_run_automl, mock_storage):
    db = TestingSessionLocal()
    
    # 1. Create a dummy dataset
    dataset_id = "test-dataset-id"
    new_dataset = Dataset(
        id=dataset_id,
        user_id="default_user",
        filename="test.csv",
        r2_path=f"raw/default_user/{dataset_id}/test.csv",
        status="uploaded"
    )
    db.add(new_dataset)
    db.commit()
    
    # 2. Mock R2 download returning a small df with target column
    df = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6], "churn": [0, 1, 0]})
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    mock_storage.download_file.return_value = csv_bytes
    
    # Mock run_automl and mlflow client
    mock_run_automl.return_value = ("runs:/test-run-123/model", (MagicMock(), {}))
    
    mock_run = MagicMock()
    mock_run.data.metrics = {"best_roc_auc": 0.82, "optimal_threshold": 0.38}
    mock_mlflow_client.return_value.get_run.return_value = mock_run
    
    # Mock generate_schema & save_schema in schema_gen module
    with patch("backend.app.core.pipeline.schema_gen.generate_schema") as mock_gen, \
         patch("backend.app.core.pipeline.schema_gen.save_schema") as mock_save:
        mock_gen.return_value = (MagicMock(), {})
        mock_save.return_value = ("schema.json", "metadata.json")
        
        # 3. Post to train endpoint
        payload = {
            "confirmed_target": "churn",
            "confirmed_profiles": [
                {
                    "name": "feat1",
                    "inferred_dtype": "int64",
                    "inferred_role": "NUMERIC",
                    "confidence_score": 1.0,
                    "null_pct": 0.0,
                    "unique_count": 3,
                    "entropy": 1.0,
                    "transform_strategy": "standard",
                    "impute_strategy": "median"
                },
                {
                    "name": "feat2",
                    "inferred_dtype": "int64",
                    "inferred_role": "NUMERIC",
                    "confidence_score": 1.0,
                    "null_pct": 0.0,
                    "unique_count": 3,
                    "entropy": 1.0,
                    "transform_strategy": "standard",
                    "impute_strategy": "median"
                },
                {
                    "name": "churn",
                    "inferred_dtype": "int64",
                    "inferred_role": "TARGET",
                    "confidence_score": 1.0,
                    "null_pct": 0.0,
                    "unique_count": 2,
                    "entropy": 0.5,
                    "transform_strategy": "passthrough",
                    "impute_strategy": "drop"
                }
            ],
            "composite_config": None,
            "prior_model_uri": None
        }
        
        response = client.post(f"/api/v1/jobs/datasets/{dataset_id}/train", json=payload, headers=HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        
        # 4. Wait for background task to complete (it runs synchronously in TestClient)
        job_id = data["job_id"]
        status_response = client.get(f"/api/v1/jobs/{job_id}/status", headers=HEADERS)
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["status"] == "completed"
        assert status_data["roc_auc"] == 0.82
        assert status_data["optimal_threshold"] == 0.38
        assert status_data["model_uri"] == "runs:/test-run-123/model"

@patch("backend.app.api.v1.jobs.storage")
@patch("backend.app.api.v1.jobs.run_automl")
@patch("mlflow.tracking.MlflowClient")
@patch("backend.app.api.v1.jobs.SessionLocal", new=TestingSessionLocal)
def test_idempotency_and_schema_saving_failures(mock_mlflow_client, mock_run_automl, mock_storage):
    db = TestingSessionLocal()
    
    # 1. Create dataset
    dataset_id = "test-ds-2"
    new_dataset = Dataset(
        id=dataset_id,
        user_id="default_user",
        filename="test.csv",
        r2_path=f"raw/default_user/{dataset_id}/test.csv",
        status="uploaded"
    )
    db.add(new_dataset)
    
    # 2. Add an already completed job with prior_model_uri = None, roc_auc = 0.70
    old_job = TrainingJob(
        id="old-job-id",
        dataset_id=dataset_id,
        status="completed",
        target_column="churn",
        roc_auc=0.70,
        prior_model_uri=None
    )
    db.add(old_job)
    db.commit()
    
    df = pd.DataFrame({"feat1": [1, 2, 3], "churn": [0, 1, 0]})
    mock_storage.download_file.return_value = df.to_csv(index=False).encode("utf-8")
    
    # payload with prior_model_uri = None
    payload = {
        "confirmed_target": "churn",
        "confirmed_profiles": [],
        "composite_config": None,
        "prior_model_uri": None
    }
    
    # 3. Post to train with prior_model_uri=None -> should return old_job due to idempotency
    response1 = client.post(f"/api/v1/jobs/datasets/{dataset_id}/train", json=payload, headers=HEADERS)
    assert response1.status_code == 200
    assert response1.json()["job_id"] == "old-job-id"
    assert response1.json()["status"] == "completed"
    
    # 4. Post with prior_model_uri="runs:/prior_run/model" -> should train again (bypass idempotency)
    payload["prior_model_uri"] = "runs:/prior_run/model"
    mock_run_automl.return_value = ("runs:/new_run_1/model", (MagicMock(), {}))
    
    mock_run = MagicMock()
    mock_run.data.metrics = {"best_roc_auc": 0.75, "optimal_threshold": 0.5}
    mock_mlflow_client.return_value.get_run.return_value = mock_run
    
    with patch("backend.app.core.pipeline.schema_gen.generate_schema") as mock_gen, \
         patch("backend.app.core.pipeline.schema_gen.save_schema") as mock_save:
        mock_gen.return_value = (MagicMock(), {})
        mock_save.return_value = ("schema.json", "metadata.json")
        
        response2 = client.post(f"/api/v1/jobs/datasets/{dataset_id}/train", json=payload, headers=HEADERS)
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["status"] == "queued"
        assert data2["job_id"] != "old-job-id"
        
        # 5. Post to train with save_schema throwing an exception -> job status should be "failed"
        # Let's clean db first for this subtest
        db.query(TrainingJob).delete()
        db.commit()
        
    mock_save_error = MagicMock(side_effect=Exception("R2 upload timed out"))
    payload["prior_model_uri"] = None
    
    with patch("backend.app.core.pipeline.schema_gen.generate_schema") as mock_gen, \
         patch("backend.app.core.pipeline.schema_gen.save_schema", mock_save_error):
        mock_gen.return_value = (MagicMock(), {})
        
        response3 = client.post(f"/api/v1/jobs/datasets/{dataset_id}/train", json=payload, headers=HEADERS)
        assert response3.status_code == 200
        job_id3 = response3.json()["job_id"]
        
        # Check job status -> should be failed, and error_message should record the failure
        status_resp = client.get(f"/api/v1/jobs/{job_id3}/status", headers=HEADERS)
        assert status_resp.json()["status"] == "failed"
        
        # Verify db contains error message
        failed_job = db.query(TrainingJob).filter(TrainingJob.id == job_id3).first()
        assert failed_job is not None
        assert "Schema save failed" in failed_job.error_message
