import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import io
import numpy as np
from backend.app.main import app
from backend.app.db.models import Base, Dataset, Profile, TrainingJob
from backend.app.db.session import get_db
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Mock DB setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./tests/test_drift.db"
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

HEADERS = {"X-API-Key": "test-api-key"}

@pytest.fixture(autouse=True)
def clean_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield

@patch("backend.app.api.v1.predict.storage")
@patch("backend.app.api.v1.predict.model_cache")
def test_drift_detection_no_drift(mock_model_cache, mock_storage):
    db = TestingSessionLocal()

    # 1. Create dataset, profile and completed training job
    dataset_id = "test-ds-drift"
    new_dataset = Dataset(
        id=dataset_id,
        user_id="default_user",
        filename="train.parquet",
        r2_path=f"raw/default_user/{dataset_id}/train.parquet",
        status="completed"
    )

    new_profile = Profile(
        dataset_id=dataset_id,
        profiles_json=[
            {"name": "feat_num", "inferred_role": "NUMERIC"},
            {"name": "feat_cat", "inferred_role": "CATEGORICAL"}
        ],
        suggested_target="churn"
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
    db.add(new_profile)
    db.add(new_job)
    db.commit()

    # 2. Mock identical reference (train) and target (inference) datasets
    ref_df = pd.DataFrame({
        "feat_num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feat_cat": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        "churn": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })

    target_df = pd.DataFrame({
        "feat_num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feat_cat": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        "churn": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })

    ref_buffer = io.BytesIO()
    ref_df.to_parquet(ref_buffer, index=False)
    
    target_buffer = io.BytesIO()
    target_df.to_parquet(target_buffer, index=False)

    # Mock storage downloads
    def mock_download(path):
        if "train.parquet" in path:
            return ref_buffer.getvalue()
        else:
            return target_buffer.getvalue()
            
    mock_storage.download_file.side_effect = mock_download

    # 3. Mock model
    mock_model = MagicMock()
    mock_model.feature_names_in_ = ["feat_num", "feat_cat"]
    mock_model_cache.get_model.return_value = mock_model

    # 4. Request drift detection
    payload = {
        "target_file_path": f"processed/{dataset_id}/inference.parquet"
    }

    resp = client.post(f"/api/v1/predict/datasets/{dataset_id}/drift", json=payload, headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()

    assert data["dataset_id"] == dataset_id
    assert data["reference_rows"] == 10
    assert data["target_rows"] == 10
    assert data["drift_detected"] is False

    metrics = data["metrics"]
    assert metrics["feat_num"]["is_drifted"] is False
    assert metrics["feat_num"]["drift_level"] == "low"
    assert metrics["feat_cat"]["is_drifted"] is False
    assert metrics["feat_cat"]["drift_level"] == "low"


@patch("backend.app.api.v1.predict.storage")
@patch("backend.app.api.v1.predict.model_cache")
def test_drift_detection_high_drift(mock_model_cache, mock_storage):
    db = TestingSessionLocal()

    # 1. Create dataset, profile and completed training job
    dataset_id = "test-ds-drift-high"
    new_dataset = Dataset(
        id=dataset_id,
        user_id="default_user",
        filename="train.parquet",
        r2_path=f"raw/default_user/{dataset_id}/train.parquet",
        status="completed"
    )

    new_profile = Profile(
        dataset_id=dataset_id,
        profiles_json=[
            {"name": "feat_num", "inferred_role": "NUMERIC"},
            {"name": "feat_cat", "inferred_role": "CATEGORICAL"}
        ],
        suggested_target="churn"
    )

    new_job = TrainingJob(
        id="job-1234",
        dataset_id=dataset_id,
        status="completed",
        model_uri="runs:/mock-run-id/model",
        target_column="churn",
        optimal_threshold=0.35,
        roc_auc=0.78
    )

    db.add(new_dataset)
    db.add(new_profile)
    db.add(new_job)
    db.commit()

    # 2. Mock reference (train) and drifted target (inference) datasets
    ref_df = pd.DataFrame({
        "feat_num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feat_cat": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        "churn": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })

    # Shift numerical values by a large constant, and change category distribution completely
    target_df = pd.DataFrame({
        "feat_num": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        "feat_cat": ["X", "Y", "Z", "X", "Y", "Z", "X", "Y", "Z", "X"],
        "churn": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })

    ref_buffer = io.BytesIO()
    ref_df.to_parquet(ref_buffer, index=False)
    
    target_buffer = io.BytesIO()
    target_df.to_parquet(target_buffer, index=False)

    # Mock storage downloads
    def mock_download(path):
        if "train.parquet" in path:
            return ref_buffer.getvalue()
        else:
            return target_buffer.getvalue()
            
    mock_storage.download_file.side_effect = mock_download

    # 3. Mock model
    mock_model = MagicMock()
    mock_model.feature_names_in_ = ["feat_num", "feat_cat"]
    mock_model_cache.get_model.return_value = mock_model

    # 4. Request drift detection
    payload = {
        "target_file_path": f"processed/{dataset_id}/inference.parquet"
    }

    resp = client.post(f"/api/v1/predict/datasets/{dataset_id}/drift", json=payload, headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()

    assert data["dataset_id"] == dataset_id
    assert data["drift_detected"] is True

    metrics = data["metrics"]
    assert metrics["feat_num"]["is_drifted"] is True
    assert metrics["feat_num"]["drift_level"] == "high"
    assert metrics["feat_cat"]["is_drifted"] is True
    assert metrics["feat_cat"]["drift_level"] == "high"
