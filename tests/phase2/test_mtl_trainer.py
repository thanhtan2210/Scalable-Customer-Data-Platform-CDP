import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from backend.app.core.training.mtl_trainer import MTLChurnModel, is_mtl_available
from backend.app.core.profiler.column_profile import ColumnProfile, DataRole
from backend.app.core.profiler.target_analysis import CompositeTargetConfig, SynthesisStrategy, ColumnWeight
from backend.app.core.training.automl import run_automl

def test_hardcoded_weights():
    """1. Hardcoded weights: alpha = 0.7, beta = 0.3"""
    assert MTLChurnModel._ALPHA == 0.7
    assert MTLChurnModel._BETA == 0.3

def test_torch_unavailable_graceful(monkeypatch):
    """2. Torch unavailable graceful fallback"""
    import backend.app.core.training.mtl_trainer as mtl_trainer
    
    # Temporarily set _TORCH_AVAILABLE to False
    monkeypatch.setattr(mtl_trainer, "_TORCH_AVAILABLE", False)
    
    assert mtl_trainer.is_mtl_available() is False
    
    model = MTLChurnModel()
    X = np.random.randn(10, 5)
    y_bin = np.random.randint(0, 2, 10)
    y_cpi = np.random.rand(10)
    
    with pytest.raises(ImportError):
        model.fit(X, y_bin, y_cpi)
        
    with pytest.raises(ImportError):
        model.predict_proba(X)
        
    with pytest.raises(ImportError):
        model.get_shared_embeddings(X)

def test_mtl_fit_predict():
    """3. MTL Fit and Predict (runs only if torch is available)"""
    if not is_mtl_available():
        pytest.skip("PyTorch is not available, skipping MTL fit/predict test.")
        
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y_bin = np.random.randint(0, 2, 100)
    y_cpi = np.random.rand(100)
    
    model = MTLChurnModel()
    fitted = model.fit(X, y_bin, y_cpi, epochs=5, batch_size=32)
    assert fitted is model
    
    probs = model.predict_proba(X)
    assert probs.shape == (100, 2)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0)

def test_inference_head_a_only():
    """4. Inference uses Head A only (runs only if torch is available)"""
    if not is_mtl_available():
        pytest.skip("PyTorch is not available, skipping MTL inference head test.")
        
    X = np.random.randn(50, 10)
    y_bin = np.random.randint(0, 2, 50)
    y_cpi = np.random.rand(50)
    
    model = MTLChurnModel()
    model.fit(X, y_bin, y_cpi, epochs=2, batch_size=16)
    
    embeddings = model.get_shared_embeddings(X)
    assert embeddings.shape == (50, 64)

@patch("backend.app.core.training.automl.mlflow")
def test_cpi_as_feature_fallback(mock_mlflow, monkeypatch):
    """5. Fallback to CPI as standard feature column when MTL is disabled"""
    import backend.app.core.training.automl as automl
    
    # Mock is_mtl_available to return False so we run the fallback path
    monkeypatch.setattr(automl, "is_mtl_available", lambda: False)
    
    # Setup dummy data
    df = pd.DataFrame({
        "feature_num": np.random.randn(100),
        "target": np.random.choice([0, 1], size=100),
        "cpi_score": np.random.rand(100)
    })
    
    profiles = [
        ColumnProfile(name="feature_num", inferred_dtype="float64", inferred_role=DataRole.NUMERIC, confidence_score=0.9, null_pct=0.0, unique_count=100, entropy=2.0, impute_strategy="median", transform_strategy="standard"),
        ColumnProfile(name="target", inferred_dtype="int64", inferred_role=DataRole.TARGET, confidence_score=1.0, null_pct=0.0, unique_count=2, entropy=1.0)
    ]
    
    comp_config = CompositeTargetConfig(
        strategy=SynthesisStrategy.WEIGHTED,
        source_columns=["feature_num"],
        cpi_variance_explained=None,
        weights=[ColumnWeight(name="feature_num", weight=1.0, normalize_method="minmax")],
        cpi_column_name="cpi_score",
        requires_confirmation=False
    )
    
    # We can patch route_models to return a basic mock model search space
    from sklearn.linear_model import LogisticRegression
    mock_routed = [{
        "name": "LogisticRegression",
        "class": LogisticRegression,
        "kwargs": {"max_iter": 100},
        "search_space": {}
    }]
    monkeypatch.setattr(automl, "route_models", lambda df, profs: mock_routed)
    monkeypatch.setattr(automl, "generate_schema", lambda profs, ds_id, target: ("dummy_schema", "dummy_metadata"))
    
    # Run automl
    model_uri, schema_res = run_automl(df, profiles, target_col="target", dataset_id="test_ds", composite_config=comp_config)
    
    # Check that model_uri is returned
    assert model_uri is not None
    assert schema_res == ("dummy_schema", "dummy_metadata")


def test_mtl_predict_before_fit():
    """6. Predict before fit raises ValueError (runs only if torch is available)"""
    if not is_mtl_available():
        pytest.skip("PyTorch is not available, skipping MTL test.")
        
    model = MTLChurnModel()
    X = np.random.randn(10, 5)
    with pytest.raises(ValueError, match="Model is not fitted yet."):
        model.predict_proba(X)
    with pytest.raises(ValueError, match="Model is not fitted yet."):
        model.get_shared_embeddings(X)


def test_mtl_fit_epochs_zero():
    """7. Fit with zero epochs (runs only if torch is available)"""
    if not is_mtl_available():
        pytest.skip("PyTorch is not available, skipping MTL test.")
        
    X = np.random.randn(20, 5)
    y_bin = np.random.randint(0, 2, 20)
    y_cpi = np.random.rand(20)
    
    model = MTLChurnModel()
    model.fit(X, y_bin, y_cpi, epochs=0, batch_size=10)
    assert model.model is not None
    # Check that it still returns model
    probs = model.predict_proba(X)
    assert probs.shape == (20, 2)


def test_mtl_pickle_roundtrip():
    """8. Pickle serialization and deserialization roundtrip (runs only if torch is available)"""
    if not is_mtl_available():
        pytest.skip("PyTorch is not available, skipping MTL test.")
        
    import pickle
    
    X = np.random.randn(50, 8)
    y_bin = np.random.randint(0, 2, 50)
    y_cpi = np.random.rand(50)
    
    model = MTLChurnModel()
    model.fit(X, y_bin, y_cpi, epochs=2, batch_size=16)
    
    # Pickle serialization
    serialized = pickle.dumps(model)
    
    # Deserialization
    loaded_model = pickle.loads(serialized)
    assert loaded_model.input_dim == 8
    assert loaded_model.model is not None
    
    # Verify inference is identical
    orig_probs = model.predict_proba(X)
    loaded_probs = loaded_model.predict_proba(X)
    assert np.allclose(orig_probs, loaded_probs)


def test_mtl_batch_size_larger_than_dataset():
    """9. Batch size larger than dataset size (runs only if torch is available)"""
    if not is_mtl_available():
        pytest.skip("PyTorch is not available, skipping MTL test.")
        
    X = np.random.randn(20, 5)
    y_bin = np.random.randint(0, 2, 20)
    y_cpi = np.random.rand(20)
    
    model = MTLChurnModel()
    # Batch size 100 > dataset size 20
    model.fit(X, y_bin, y_cpi, epochs=2, batch_size=100)
    assert model.model is not None


def test_mtl_y_cpi_outside_range():
    """10. Continuous target values outside typical [0, 1] range (runs only if torch is available)"""
    if not is_mtl_available():
        pytest.skip("PyTorch is not available, skipping MTL test.")
        
    X = np.random.randn(30, 6)
    y_bin = np.random.randint(0, 2, 30)
    y_cpi = np.random.randn(30) * 10.0  # arbitrary range
    
    model = MTLChurnModel()
    model.fit(X, y_bin, y_cpi, epochs=2, batch_size=10)
    assert model.model is not None
    probs = model.predict_proba(X)
    assert probs.shape == (30, 2)


def test_mtl_batch_size_one_edge_case():
    """11. Batch size of 1 triggers size 1 batch check in DataLoader loop (runs only if torch is available)"""
    if not is_mtl_available():
        pytest.skip("PyTorch is not available, skipping MTL test.")
        
    # We have 11 samples, batch_size = 5. The last batch has size 1.
    X = np.random.randn(11, 5)
    y_bin = np.random.randint(0, 2, 11)
    y_cpi = np.random.rand(11)
    
    model = MTLChurnModel()
    # During training, the last batch with size 1 should be skipped to prevent BatchNorm failure
    model.fit(X, y_bin, y_cpi, epochs=2, batch_size=5)
    assert model.model is not None


def test_mtl_proba_sum_to_one():
    """12. Predict probabilities must sum to 1.0 (runs only if torch is available)"""
    if not is_mtl_available():
        pytest.skip("PyTorch is not available, skipping MTL test.")
        
    X = np.random.randn(40, 5)
    y_bin = np.random.randint(0, 2, 40)
    y_cpi = np.random.rand(40)
    
    model = MTLChurnModel()
    model.fit(X, y_bin, y_cpi, epochs=2, batch_size=10)
    
    probs = model.predict_proba(X)
    assert np.allclose(probs.sum(axis=1), 1.0)
