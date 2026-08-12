import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import numpy as np

from backend.app.main import app
from backend.app.db.models import TrainingJob
from backend.app.db.session import get_db

client = TestClient(app)
HEADERS = {"X-API-Key": "test-api-key"}


@patch("mlflow.tracking.MlflowClient")
@patch("backend.app.api.v1.models.model_cache")
def test_sklearn_feature_importance(mock_model_cache, mock_mlflow_client):
    mock_db = MagicMock()
    mock_job = TrainingJob(
        id="job-rf-123",
        dataset_id="ds-1",
        status="completed",
        model_uri="runs:/run-1/model",
        target_column="target",
        roc_auc=0.85,
        started_at=datetime.utcnow(),
        is_active=True,
        tags={},
    )
    mock_q = MagicMock()
    mock_q.filter.return_value = mock_q
    mock_q.order_by.return_value = mock_q
    mock_q.first.return_value = mock_job
    mock_db.query.return_value = mock_q

    mock_estimator = MagicMock()
    mock_importances = np.linspace(0.01, 0.25, 25)
    mock_estimator.feature_importances_ = mock_importances

    mock_pipeline = MagicMock()
    mock_pipeline.steps = [("preprocessor", MagicMock()), ("model", mock_estimator)]
    mock_model_cache.get_or_load.return_value = (mock_pipeline, "sklearn")
    mock_mlflow_client.return_value.download_artifacts.side_effect = Exception("No mlflow")

    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        resp = client.get(
            "/api/v1/models/ds-1/feature-importance",
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["dataset_id"] == "ds-1"
        assert data["job_id"] == "job-rf-123"
        assert data["method"] == "feature_importances"
        assert len(data["feature_importance"]) == 20
        importances = [f["importance"] for f in data["feature_importance"]]
        assert importances == sorted(importances, reverse=True)
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_feature_importance_no_model_404():
    mock_db = MagicMock()
    mock_q = MagicMock()
    mock_q.filter.return_value = mock_q
    mock_q.order_by.return_value = mock_q
    mock_q.first.return_value = None
    mock_db.query.return_value = mock_q

    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        resp = client.get(
            "/api/v1/models/non-existent-ds-9999/feature-importance",
            headers=HEADERS,
        )
        assert resp.status_code == 404
    finally:
        app.dependency_overrides.pop(get_db, None)


@patch("mlflow.tracking.MlflowClient")
@patch("backend.app.api.v1.models.model_cache")
def test_feature_importance_uses_active_model(mock_model_cache, mock_mlflow_client):
    mock_db = MagicMock()
    mock_active_job = TrainingJob(
        id="active-job-999",
        dataset_id="ds-2",
        status="completed",
        model_uri="runs:/run-active/model",
        target_column="target",
        roc_auc=0.90,
        started_at=datetime.utcnow(),
        is_active=True,
        tags={},
    )
    mock_q = MagicMock()
    mock_q.filter.return_value = mock_q
    mock_q.order_by.return_value = mock_q
    mock_q.first.return_value = mock_active_job
    mock_db.query.return_value = mock_q

    mock_estimator = MagicMock()
    mock_estimator.feature_importances_ = np.array([0.8, 0.2])
    mock_pipeline = MagicMock()
    mock_pipeline.steps = [("pre", MagicMock()), ("model", mock_estimator)]
    mock_model_cache.get_or_load.return_value = (mock_pipeline, "sklearn")
    mock_mlflow_client.return_value.download_artifacts.side_effect = Exception("No mlflow")

    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        resp = client.get(
            "/api/v1/models/ds-2/feature-importance",
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "active-job-999"
    finally:
        app.dependency_overrides.pop(get_db, None)


@patch("mlflow.tracking.MlflowClient")
@patch("backend.app.api.v1.models.model_cache")
@patch("backend.app.core.storage.storage")
def test_shap_fallback_when_no_inference_data(mock_storage, mock_model_cache, mock_mlflow_client):
    mock_db = MagicMock()
    mock_job = TrainingJob(
        id="job-mtl-1",
        dataset_id="ds-mtl",
        status="completed",
        model_uri="runs:/run-mtl/model",
        target_column="target",
        roc_auc=0.88,
        started_at=datetime.utcnow(),
        is_active=True,
        tags={},
    )
    mock_q = MagicMock()
    mock_q.filter.return_value = mock_q
    mock_q.order_by.return_value = mock_q
    mock_q.first.return_value = mock_job
    mock_db.query.return_value = mock_q

    mock_estimator = MagicMock(spec=[])
    mock_model = MagicMock()
    mock_model.steps = [("pre", MagicMock()), ("model", mock_estimator)]

    mock_model_cache.get_or_load.return_value = (mock_model, "mtl_sklearn")
    mock_storage.list_files.return_value = []
    mock_mlflow_client.return_value.download_artifacts.side_effect = Exception("No mlflow")

    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        resp = client.get(
            "/api/v1/models/ds-mtl/feature-importance",
            headers=HEADERS,
        )
        assert resp.status_code == 422
    finally:
        app.dependency_overrides.pop(get_db, None)
