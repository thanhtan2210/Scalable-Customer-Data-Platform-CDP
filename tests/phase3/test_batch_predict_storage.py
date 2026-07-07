import pytest
import io
import pandas as pd
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.db.models import Base, Dataset, TrainingJob
from backend.app.db.session import get_db
from backend.app.core.storage import StorageClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Mock DB setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./tests/test_batch_predict_storage.db"
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

# Mock model
class DummyModel:
    def predict_proba(self, X):
        import numpy as np
        # predict_proba returns [prob_0, prob_1] for each record
        return np.array([[0.9, 0.1]] * len(X))

@patch("backend.app.api.v1.predict.storage")
@patch("backend.app.api.v1.predict.model_cache")
def test_batch_predict_saves_inference_data(mock_model_cache, mock_storage):
    db = TestingSessionLocal()
    db.query(TrainingJob).delete()
    db.query(Dataset).delete()
    db.commit()

    dataset_id = "test-ds-123"
    job = TrainingJob(
        id="job-123",
        dataset_id=dataset_id,
        status="completed",
        model_uri="runs:/test-run/model",
        target_column="churn_label",
        roc_auc=0.9
    )
    db.add(job)
    db.commit()

    # Mock storage download
    csv_content = b"cust_id,feature_1,feature_2,churn_label\ncust_01,0.5,1.2,0\ncust_02,0.8,0.9,1"
    mock_storage.download_file.return_value = csv_content

    # Mock model cache return: (model_object, model_type)
    mock_model_cache.get_model.return_value = (DummyModel(), "sklearn")
    mock_model_cache.load_model.return_value = (DummyModel(), "sklearn")

    # Call batch predict
    payload = {
        "dataset_id": dataset_id,
        "file_path": "raw/user/test.csv"
    }
    resp = client.post("/api/v1/predict/batch", headers=HEADERS, json=payload)
    assert resp.status_code == 200, resp.text

    # Verify storage.upload_file was called to save inference data
    assert mock_storage.upload_file.called
    args, kwargs = mock_storage.upload_file.call_args
    # First argument should be bytes
    upload_bytes = args[0]
    # Second argument should be path
    upload_path = args[1]
    assert upload_path.startswith(f"ml_artifacts/{dataset_id}/inference/")
    assert upload_path.endswith(".parquet")

@patch("backend.app.api.v1.predict.storage")
@patch("backend.app.api.v1.predict.model_cache")
def test_inference_data_correct_format(mock_model_cache, mock_storage):
    db = TestingSessionLocal()
    db.query(TrainingJob).delete()
    db.query(Dataset).delete()
    db.commit()

    dataset_id = "test-ds-123"
    job = TrainingJob(
        id="job-123",
        dataset_id=dataset_id,
        status="completed",
        model_uri="runs:/test-run/model",
        target_column="churn_label",
        roc_auc=0.9
    )
    db.add(job)
    db.commit()

    csv_content = b"cust_id,feature_1,feature_2,churn_label,churn\ncust_01,0.5,1.2,0,0\ncust_02,0.8,0.9,1,1"
    mock_storage.download_file.return_value = csv_content
    mock_model_cache.get_model.return_value = (DummyModel(), "sklearn")
    mock_model_cache.load_model.return_value = (DummyModel(), "sklearn")

    payload = {
        "dataset_id": dataset_id,
        "file_path": "raw/user/test.csv"
    }
    resp = client.post("/api/v1/predict/batch", headers=HEADERS, json=payload)
    assert resp.status_code == 200, resp.text

    # Retrieve upload call args
    args, _ = mock_storage.upload_file.call_args
    upload_bytes = args[0]

    # Read uploaded parquet content back to pandas
    df = pd.read_parquet(io.BytesIO(upload_bytes))
    
    # Assertions on contents
    assert len(df) == 2
    # Target columns must be dropped to prevent leakage
    assert "churn_label" not in df.columns
    assert "churn" not in df.columns
    # Other features must be preserved
    assert "cust_id" in df.columns
    assert "feature_1" in df.columns
    assert "feature_2" in df.columns
