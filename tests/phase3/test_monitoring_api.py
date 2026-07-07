import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.db.models import TrainingJob

HEADERS = {"X-API-Key": "test-api-key"}
client = TestClient(app)

@patch("backend.app.api.v1.monitoring.ping_with_timeout")
def test_health_returns_200_with_services(mock_ping):
    mock_ping.side_effect = [
        {"status": "up", "latency_ms": 10.0},
        {"status": "up", "latency_ms": 12.0},
        {"status": "up", "latency_ms": 15.0}
    ]
    resp = client.get("/api/v1/monitoring/health", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data["services"]["database"]["status"] == "up"
    assert data["services"]["storage"]["status"] == "up"
    assert data["services"]["mlflow"]["status"] == "up"

@patch("backend.app.api.v1.monitoring.ping_with_timeout")
def test_health_overall_status_logic(mock_ping):
    # Case degraded: database up, but storage down
    mock_ping.side_effect = [
        {"status": "up", "latency_ms": 10.0},     # database
        {"status": "down", "error": "connection failed"}, # storage
        {"status": "up", "latency_ms": 15.0}      # mlflow
    ]
    resp = client.get("/api/v1/monitoring/health", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "degraded"

    # Case unhealthy: all down (specifically database down)
    mock_ping.side_effect = [
        {"status": "down", "error": "timeout"},   # database
        {"status": "down", "error": "timeout"},   # storage
        {"status": "down", "error": "timeout"}    # mlflow
    ]
    resp = client.get("/api/v1/monitoring/health", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "unhealthy"

@patch("backend.app.api.v1.monitoring.StorageClient")
def test_metrics_has_job_fields(mock_storage_client):
    mock_storage_inst = MagicMock()
    mock_storage_inst.list_files.return_value = ["file1.csv", "file2.csv"]
    mock_storage_client.return_value = mock_storage_inst

    mock_db = MagicMock()
    mock_q = MagicMock()
    mock_q.filter.return_value = mock_q
    mock_q.count.side_effect = [1, 2, 5, 0]
    mock_db.query.return_value = mock_q

    from backend.app.db.session import get_db
    app.dependency_overrides[get_db] = lambda: mock_db

    try:
        resp = client.get("/api/v1/monitoring/metrics", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "timestamp" in data
        assert data["jobs"]["queued"] == 1
        assert data["jobs"]["training"] == 2
        assert data["jobs"]["completed_24h"] == 5
        assert data["jobs"]["failed_24h"] == 0
        assert data["storage"]["total_files"] == 2
        assert "predict_endpoint" in data
    finally:
        app.dependency_overrides.pop(get_db, None)

def test_jobs_summary_structure():
    mock_db = MagicMock()
    
    job1 = TrainingJob(id="job1", dataset_id="ds1", status="completed", roc_auc=0.85, started_at=datetime.utcnow())
    job2 = TrainingJob(id="job2", dataset_id="ds2", status="failed", roc_auc=None, started_at=datetime.utcnow())
    
    mock_q = MagicMock()
    mock_q.filter.return_value = mock_q
    mock_q.all.return_value = [job1, job2]
    mock_db.query.return_value = mock_q

    from backend.app.db.session import get_db
    app.dependency_overrides[get_db] = lambda: mock_db

    try:
        resp = client.get("/api/v1/monitoring/jobs/summary", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["period"] == "last_7_days"
        assert data["total_jobs"] == 2
        assert data["by_status"]["completed"] == 1
        assert data["by_status"]["failed"] == 1
        assert data["best_model"]["job_id"] == "job1"
        assert data["best_model"]["roc_auc"] == 0.85
    finally:
        app.dependency_overrides.pop(get_db, None)
