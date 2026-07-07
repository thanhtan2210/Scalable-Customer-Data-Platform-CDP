import pytest
import io
import asyncio
import pandas as pd
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.db.models import Base, Dataset, TrainingJob, DriftReport
from backend.app.db.session import get_db
from backend.app.core.storage import StorageClient
from backend.app.core.serving.retrain_loop import run_drift_check_loop
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Mock DB setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./tests/test_drift_retrain.db"
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
    feature_names_in_ = ["feature_1", "feature_2"]

@patch("backend.app.api.v1.predict.storage")
@patch("backend.app.api.v1.predict.model_cache")
def test_drift_refactored_endpoint_saves_report(mock_model_cache, mock_storage):
    db = TestingSessionLocal()
    db.query(DriftReport).delete()
    db.query(TrainingJob).delete()
    db.query(Dataset).delete()
    db.commit()

    dataset_id = "test-ds-drift"
    # Create dataset, profile, and training job
    ds = Dataset(
        id=dataset_id,
        user_id="user_1",
        filename="test.csv",
        r2_path=f"raw/user_1/{dataset_id}/test.csv",
        status="uploaded"
    )
    db.add(ds)
    
    from backend.app.db.models import Profile
    db.query(Profile).delete()
    profile = Profile(
        id="profile-drift",
        dataset_id=dataset_id,
        profiles_json=[
            {"name": "feature_1", "inferred_role": "NUMERIC"},
            {"name": "feature_2", "inferred_role": "NUMERIC"},
            {"name": "churn_label", "inferred_role": "CATEGORICAL"}
        ]
    )
    db.add(profile)

    job = TrainingJob(
        id="job-drift",
        dataset_id=dataset_id,
        status="completed",
        model_uri="runs:/test-run/model",
        target_column="churn_label",
        roc_auc=0.9
    )
    db.add(job)
    db.commit()

    # Mock baseline download
    mock_storage.download_file.side_effect = [
        # Reference dataset (CSV format)
        b"feature_1,feature_2,churn_label\n0.1,1.0,0\n0.2,2.0,1",
        # Target dataset (CSV format for manual target_file_path)
        b"feature_1,feature_2,churn_label\n0.1,1.0,0\n0.2,2.0,1"
    ]

    mock_model_cache.get_model.return_value = (DummyModel(), "sklearn")
    mock_model_cache.load_model.return_value = (DummyModel(), "sklearn")

    payload = {
        "target_file_path": "raw/user_1/target.csv"
    }
    resp = client.post(f"/api/v1/predict/{dataset_id}/drift", headers=HEADERS, json=payload)
    assert resp.status_code == 200, resp.text

    # Verify report row was added to DB
    reports = db.query(DriftReport).filter(DriftReport.dataset_id == dataset_id).all()
    assert len(reports) == 1
    assert reports[0].reference_rows == 2
    assert reports[0].target_rows == 2
    assert reports[0].drift_detected is False

@patch("backend.app.api.v1.predict.storage")
@patch("backend.app.api.v1.predict.model_cache")
def test_drift_refactored_endpoint_auto_aggregate(mock_model_cache, mock_storage):
    db = TestingSessionLocal()
    db.query(DriftReport).delete()
    db.query(TrainingJob).delete()
    db.query(Dataset).delete()
    db.commit()

    dataset_id = "test-ds-drift"
    ds = Dataset(
        id=dataset_id,
        user_id="user_1",
        filename="test.csv",
        r2_path=f"raw/user_1/{dataset_id}/test.csv",
        status="uploaded"
    )
    db.add(ds)
    
    from backend.app.db.models import Profile
    db.query(Profile).delete()
    profile = Profile(
        id="profile-drift",
        dataset_id=dataset_id,
        profiles_json=[
            {"name": "feature_1", "inferred_role": "NUMERIC"},
            {"name": "feature_2", "inferred_role": "NUMERIC"},
            {"name": "churn_label", "inferred_role": "CATEGORICAL"}
        ]
    )
    db.add(profile)

    job = TrainingJob(
        id="job-drift",
        dataset_id=dataset_id,
        status="completed",
        model_uri="runs:/test-run/model",
        target_column="churn_label",
        roc_auc=0.9
    )
    db.add(job)
    db.commit()

    # Create dummy parquet files in memory
    df1 = pd.DataFrame({"feature_1": [0.1], "feature_2": [1.0]})
    df2 = pd.DataFrame({"feature_1": [0.2], "feature_2": [2.0]})
    
    buf1 = io.BytesIO()
    df1.to_parquet(buf1, index=False)
    bytes1 = buf1.getvalue()

    buf2 = io.BytesIO()
    df2.to_parquet(buf2, index=False)
    bytes2 = buf2.getvalue()

    # Mock storage.list_files returning two files
    mock_storage.list_files.return_value = [
        f"ml_artifacts/{dataset_id}/inference/2026-07-07/file1.parquet",
        f"ml_artifacts/{dataset_id}/inference/2026-07-07/file2.parquet"
    ]

    # Mock storage downloads: first reference download, then two target parquet downloads
    mock_storage.download_file.side_effect = [
        # Reference dataset (CSV format)
        b"feature_1,feature_2,churn_label\n0.1,1.0,0\n0.2,2.0,1",
        # Target parquet 1
        bytes1,
        # Target parquet 2
        bytes2
    ]

    mock_model_cache.get_model.return_value = (DummyModel(), "sklearn")
    mock_model_cache.load_model.return_value = (DummyModel(), "sklearn")

    payload = {
        "date": "2026-07-07"
    }
    resp = client.post(f"/api/v1/predict/{dataset_id}/drift", headers=HEADERS, json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["reference_rows"] == 2
    assert data["target_rows"] == 2
    assert data["drift_detected"] is False

    # Check database report entry
    reports = db.query(DriftReport).filter(DriftReport.dataset_id == dataset_id).all()
    assert len(reports) == 1
    assert reports[0].target_rows == 2

@patch("backend.app.core.serving.retrain_loop.httpx.AsyncClient")
def test_auto_retrain_trigger_flow(mock_async_client):
    db = TestingSessionLocal()
    db.query(TrainingJob).delete()
    db.query(Dataset).delete()
    db.commit()

    # Setup database with one completed training job
    dataset_id = "test-ds-retrain"
    ds = Dataset(
        id=dataset_id,
        user_id="user_1",
        filename="test.csv",
        r2_path=f"raw/user_1/{dataset_id}/test.csv",
        status="uploaded"
    )
    db.add(ds)
    
    job = TrainingJob(
        id="job-retrain",
        dataset_id=dataset_id,
        status="completed",
        model_uri="runs:/test-run/model",
        target_column="churn_label",
        roc_auc=0.9
    )
    db.add(job)
    db.commit()

    mock_client_inst = MagicMock()
    mock_async_client.return_value.__aenter__.return_value = mock_client_inst
    
    mock_drift_resp = MagicMock()
    mock_drift_resp.status_code = 200
    mock_drift_resp.json.return_value = {"drift_detected": True}
    
    mock_train_resp = MagicMock()
    mock_train_resp.status_code = 200
    mock_train_resp.json.return_value = {"job_id": "new-job-123", "status": "training"}

    async_drift_future = asyncio.Future()
    async_drift_future.set_result(mock_drift_resp)
    
    async_train_future = asyncio.Future()
    async_train_future.set_result(mock_train_resp)
    
    mock_client_inst.post.side_effect = [async_drift_future, async_train_future]

    # Patch config variables to ensure auto retraining runs once immediately
    with patch("backend.app.core.serving.retrain_loop.config") as mock_config, \
         patch("backend.app.core.serving.retrain_loop.SessionLocal") as mock_session_local:
        mock_config.DRIFT_CHECK_INTERVAL_SEC = 1
        mock_config.API_KEY = "test-api-key"
        mock_session_local.side_effect = lambda: TestingSessionLocal()
        
        # Override asyncio.sleep to break the loop or run only once
        async def mock_sleep(seconds):
            if seconds == 10:
                return
            else:
                raise GeneratorExit("Loop Break")
                
        with patch("backend.app.core.serving.retrain_loop.asyncio.sleep", mock_sleep):
            try:
                # Run the loop (will run once then raise GeneratorExit)
                asyncio.run(run_drift_check_loop())
            except GeneratorExit:
                pass

    # Verify that local train POST request was triggered
    assert mock_client_inst.post.call_count == 2
    call_args_list = mock_client_inst.post.call_args_list
    assert call_args_list[0][0][0] == f"http://localhost:8000/api/v1/predict/{dataset_id}/drift"
    assert call_args_list[1][0][0] == "http://localhost:8000/api/v1/jobs/train"
    payload = call_args_list[1][1]["json"]
    assert payload["dataset_id"] == dataset_id
    assert payload["target_column"] == "churn_label"
