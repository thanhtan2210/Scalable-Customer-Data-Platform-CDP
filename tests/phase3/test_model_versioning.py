import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, call
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.db.models import TrainingJob, User
from backend.app.db.session import get_db

HEADERS = {"X-API-Key": "test-api-key"}
client = TestClient(app)


def make_mock_system_user():
    """Return a mock system User to satisfy get_current_user dependency."""
    user = MagicMock(spec=User)
    user.id = "system-service-user"
    user.email = "system@cdp.internal"
    user.is_active = True
    user.is_admin = True
    return user


def make_mock_db(*job_returns):
    """
    Build a mock_db where:
    - First db.query(User) call (from get_current_user) returns a system user mock.
    - Subsequent db.query(TrainingJob) calls return values from job_returns in order.

    Uses model-type dispatch: if query is called with User, return user mock;
    otherwise return job mocks in sequence.
    """
    mock_db = MagicMock()
    system_user = make_mock_system_user()
    job_iter = iter(job_returns)

    def _query(model_class):
        q = MagicMock()
        q.filter.return_value = q
        q.order_by.return_value = q
        q.update.return_value = 0

        if model_class is User:
            # Auth dependency: return system user
            q.first.return_value = system_user
            q.all.return_value = [system_user]
        else:
            # Endpoint: return next job value
            val = next(job_iter, None)
            q.first.return_value = val
            q.all.return_value = [val] if val is not None else []

        return q

    mock_db.query.side_effect = _query
    return mock_db


def test_list_models_returns_completed_only():
    mock_db = MagicMock()

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

    # promote_model endpoint calls db.query(TrainingJob) 3 times:
    # 1. find job_to_promote by job_id
    # 2. find current_active
    # 3. bulk deactivate (update)
    mock_db = make_mock_db(job_to_promote, current_active, None)

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
    finally:
        app.dependency_overrides.pop(get_db, None)


@patch("backend.app.api.v1.models.model_cache")
def test_promote_invalidates_cache(mock_model_cache):
    job_to_promote = TrainingJob(
        id="job1",
        dataset_id="ds1",
        status="completed",
        target_column="churn",
        is_active=False
    )

    # 1. find job_to_promote, 2. no current active, 3. bulk update
    mock_db = make_mock_db(job_to_promote, None, None)

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

    # compare_models calls get_job() twice
    mock_db = make_mock_db(job_a, job_b)

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

    mock_db = make_mock_db(job_a, job_b)

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
    import numpy as np

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

    # Mock predict_proba
    mock_model_inst = MagicMock()
    mock_model_inst.predict_proba.return_value = np.array([[0.9, 0.1]])
    # Return a tuple (model, type) for endpoint unpacking
    mock_model_cache.get_model.return_value = (mock_model_inst, "sklearn")

    # predict endpoint calls db.query(TrainingJob):
    # 1. active job → job_b
    # 2. profile → None
    mock_db = make_mock_db(job_b, None)

    app.dependency_overrides[get_db] = lambda: mock_db

    # Patch get_optimal_threshold to bypass MLflow network
    with patch("backend.app.api.v1.predict.get_optimal_threshold") as mock_thresh:
        mock_thresh.return_value = (0.5, "stored")
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
