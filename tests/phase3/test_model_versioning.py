import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.db.models import TrainingJob
from backend.app.db.session import get_db

HEADERS = {"X-API-Key": "test-api-key"}
client = TestClient(app)

def test_list_models_returns_completed_only():
    mock_db = MagicMock()
    
    # 1 completed job
    job1 = TrainingJob(
        id="job1", 
        dataset_id="ds1", 
        status="completed", 
        model_uri="runs:/job1", 
        target_column="churn", 
        roc_auc=0.85, 
        started_at=datetime.utcnow(), 
        is_active=False,
        tags={}
    )
    mock_q = MagicMock()
    mock_q.filter.return_value = mock_q
    mock_q.order_by.return_value = mock_q
    mock_q.all.return_value = [job1]
    
    mock_db.query.return_value = mock_q
    
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        resp = client.get("/api/v1/models/ds1", headers=HEADERS)
        assert resp.status_code == 200, f"Error: {resp.text}"
        data = resp.json()
        assert data["dataset_id"] == "ds1"
        assert data["total"] == 1
        assert data["models"][0]["job_id"] == "job1"
        assert data["models"][0]["status"] == "completed"
    finally:
        app.dependency_overrides.pop(get_db, None)

def test_promote_model_deactivates_others():
    mock_db = MagicMock()
    
    job_to_promote = TrainingJob(
        id="job1", 
        dataset_id="ds1", 
        status="completed", 
        target_column="churn",
        is_active=False
    )
    current_active = TrainingJob(
        id="job2", 
        dataset_id="ds1", 
        status="completed", 
        target_column="churn",
        is_active=True
    )
    
    mock_q = MagicMock()
    mock_q.filter.return_value = mock_q
    mock_q.first.side_effect = [job_to_promote, current_active]
    mock_db.query.return_value = mock_q
    
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        resp = client.post(
            "/api/v1/models/ds1/promote", 
            headers=HEADERS, 
            json={"job_id": "job1"}
        )
        assert resp.status_code == 200, f"Error: {resp.text}"
        data = resp.json()
        assert data["promoted_job_id"] == "job1"
        assert data["previous_active"] == "job2"
        assert data["dataset_id"] == "ds1"
        assert job_to_promote.is_active is True
        mock_q.update.assert_called_once_with({"is_active": False})
    finally:
        app.dependency_overrides.pop(get_db, None)

@patch("backend.app.api.v1.models.model_cache")
def test_promote_invalidates_cache(mock_model_cache):
    mock_db = MagicMock()
    job_to_promote = TrainingJob(
        id="job1", 
        dataset_id="ds1", 
        status="completed", 
        target_column="churn",
        is_active=False
    )
    mock_q = MagicMock()
    mock_q.filter.return_value = mock_q
    mock_q.first.side_effect = [job_to_promote, None]
    mock_db.query.return_value = mock_q
    
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        resp = client.post(
            "/api/v1/models/ds1/promote", 
            headers=HEADERS, 
            json={"job_id": "job1"}
        )
        assert resp.status_code == 200, f"Error: {resp.text}"
        mock_model_cache.invalidate.assert_called_once_with(dataset_id="ds1")
    finally:
        app.dependency_overrides.pop(get_db, None)

def test_compare_models_winner_logic():
    mock_db = MagicMock()
    job_a = TrainingJob(
        id="job_a", 
        dataset_id="ds1", 
        status="completed", 
        target_column="churn", 
        roc_auc=0.85, 
        started_at=datetime.utcnow(), 
        is_active=False
    )
    job_b = TrainingJob(
        id="job_b", 
        dataset_id="ds1", 
        status="completed", 
        target_column="churn", 
        roc_auc=0.78, 
        started_at=datetime.utcnow(), 
        is_active=False
    )
    
    mock_q = MagicMock()
    mock_q.filter.return_value = mock_q
    mock_q.first.side_effect = [job_a, job_b]
    mock_db.query.return_value = mock_q
    
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        resp = client.get(
            "/api/v1/models/ds1/compare?job_id_a=job_a&job_id_b=job_b", 
            headers=HEADERS
        )
        assert resp.status_code == 200, f"Error: {resp.text}"
        data = resp.json()
        assert data["winner"] == "a"
        assert data["delta_roc_auc"] == 0.07
    finally:
        app.dependency_overrides.pop(get_db, None)

def test_compare_models_tie():
    mock_db = MagicMock()
    job_a = TrainingJob(
        id="job_a", 
        dataset_id="ds1", 
        status="completed", 
        target_column="churn", 
        roc_auc=0.801, 
        started_at=datetime.utcnow(), 
        is_active=False
    )
    job_b = TrainingJob(
        id="job_b", 
        dataset_id="ds1", 
        status="completed", 
        target_column="churn", 
        roc_auc=0.800, 
        started_at=datetime.utcnow(), 
        is_active=False
    )
    
    mock_q = MagicMock()
    mock_q.filter.return_value = mock_q
    mock_q.first.side_effect = [job_a, job_b]
    mock_db.query.return_value = mock_q
    
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        resp = client.get(
            "/api/v1/models/ds1/compare?job_id_a=job_a&job_id_b=job_b", 
            headers=HEADERS
        )
        assert resp.status_code == 200, f"Error: {resp.text}"
        data = resp.json()
        assert data["winner"] == "tie"
        assert abs(data["delta_roc_auc"]) == 0.001
    finally:
        app.dependency_overrides.pop(get_db, None)

@patch("backend.app.api.v1.predict.model_cache")
def test_predict_uses_active_model(mock_model_cache):
    # Setup mock model cache instance
    import numpy as np
    mock_model_inst = MagicMock()
    # Mock predict_proba to return a numpy array
    mock_model_inst.predict_proba.return_value = np.array([[0.9, 0.1]])
    mock_model_cache.get_model.return_value = mock_model_inst
    
    mock_db = MagicMock()
    
    job_b = TrainingJob(
        id="job_b", 
        dataset_id="ds1", 
        status="completed", 
        target_column="churn",
        model_uri="runs:/job_b", 
        roc_auc=0.70, 
        optimal_threshold=0.5, 
        is_active=True
    )
    
    mock_q = MagicMock()
    mock_q.filter.return_value = mock_q
    mock_q.first.side_effect = [job_b, None]
    
    mock_db.query.return_value = mock_q
    
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        resp = client.post(
            "/api/v1/predict", 
            headers=HEADERS, 
            json={
                "dataset_id": "ds1",
                "records": [{"col1": 1, "col2": 2}]
            }
        )
        assert resp.status_code == 200, f"Error: {resp.text}"
        mock_model_cache.get_model.assert_called_with("runs:/job_b")
    finally:
        app.dependency_overrides.pop(get_db, None)
