import pytest
from unittest.mock import MagicMock, patch
from backend.app.core.serving.model_loader import ModelCache

def test_detect_sklearn_model_type():
    cache = ModelCache()
    mock_pipeline = MagicMock()
    mock_pipeline.steps = [
        ("preprocessor", MagicMock()),
        ("model", MagicMock(spec=["fit", "predict_proba"]))
    ]
    result = cache._detect_model_type(mock_pipeline)
    assert result == "sklearn"

def test_detect_mtl_model_type():
    cache = ModelCache()
    class DummyMTLChurnModel:
        pass
        
    with patch("backend.app.core.training.mtl_trainer.MTLChurnModel", DummyMTLChurnModel):
        mock_pipeline = MagicMock()
        mock_pipeline.steps = [
            ("preprocessor", MagicMock()),
            ("model", DummyMTLChurnModel())
        ]
        result = cache._detect_model_type(mock_pipeline)
        assert result == "mtl_sklearn"

def test_invalidate_all():
    cache = ModelCache()
    cache._models = {
        "key1": {"model": MagicMock(),
                 "model_type": "sklearn",
                 "loaded_at": 0,
                 "model_uri": "runs:/abc/model"},
        "key2": {"model": MagicMock(),
                 "model_type": "mtl_sklearn",
                 "loaded_at": 0,
                 "model_uri": "runs:/def/model"}
    }
    cache._last_loaded = {
        "key1": 0, "key2": 0}
    count = cache.invalidate()
    assert count == 2
    assert len(cache._models) == 0

def test_invalidate_by_dataset_id():
    cache = ModelCache()
    dataset_id = "test-dataset-123"
    cache._models = {
        f"{dataset_id}_v1": {
            "model": MagicMock(),
            "model_type": "sklearn",
            "loaded_at": 0,
            "model_uri": "runs:/abc/model"
        },
        "other_dataset_v1": {
            "model": MagicMock(),
            "model_type": "sklearn",
            "loaded_at": 0,
            "model_uri": "runs:/def/model"
        }
    }
    cache._last_loaded = {
        f"{dataset_id}_v1": 0,
        "other_dataset_v1": 0
    }
    count = cache.invalidate(dataset_id=dataset_id)
    assert count == 1
    assert f"{dataset_id}_v1" not in cache._models
    assert "other_dataset_v1" in cache._models

def test_get_model_returns_tuple():
    cache = ModelCache()
    mock_model = MagicMock()
    cache._models["test_key"] = {
        "model": mock_model,
        "model_type": "sklearn",
        "loaded_at": 0,
        "model_uri": "runs:/abc/model"
    }
    model, model_type = cache.get_model("test_key")
    assert model is mock_model
    assert model_type == "sklearn"
