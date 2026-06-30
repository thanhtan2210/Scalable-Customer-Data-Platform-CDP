import os
import pytest
import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
except (ImportError, OSError):
    torch = None
    nn = None
from backend.app.core.training.continual_trainer import FisherCalculator, ReplayBuffer, ContinualMTLTrainer
from backend.app.core.training.mtl_trainer import _MTLPyTorchModel, MTLChurnModel, is_mtl_available
from backend.app.core.profiler.target_analysis import CompositeTargetConfig, SynthesisStrategy

def test_fisher_calculator():
    if not is_mtl_available():
        pytest.skip("PyTorch is not available or blocked by policy.")
    input_dim = 10
    model = _MTLPyTorchModel(input_dim)
    
    # Generate dummy inputs
    n_samples = 50
    X = np.random.randn(n_samples, input_dim)
    y_binary = np.random.randint(0, 2, size=(n_samples,))
    y_cpi = np.random.rand(n_samples)
    
    fisher = FisherCalculator.calculate_fisher(model, X, y_binary, y_cpi)
    
    # Check that we computed weights for all trainable parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert name in fisher
            assert fisher[name].shape == param.shape
            # Fisher matrix elements must be non-negative
            assert torch.all(fisher[name] >= 0.0)

def test_replay_buffer_stratified_sampling(tmp_path, monkeypatch):
    # Mock R2 upload/download using local filesystem for test isolation
    local_storage = {}
    
    class MockStorage:
        def download_file(self, path):
            if path in local_storage:
                return local_storage[path]
            raise FileNotFoundError()
            
        def upload_file(self, content, path):
            local_storage[path] = content

    mock_storage_instance = MockStorage()
    # Monkeypatch storage in continual_trainer
    import backend.app.core.training.continual_trainer as ct
    monkeypatch.setattr(ct, "storage", mock_storage_instance)

    dataset_id = "test_dataset"
    buffer = ReplayBuffer(max_size=10)
    
    # Generate synthetic target label imbalance (80% 0s, 20% 1s)
    df_new = pd.DataFrame({
        "feature_1": np.random.randn(100),
        "Churn": [0] * 80 + [1] * 20
    })
    
    # Initial save/update
    buffer.update(dataset_id, df_new, "Churn")
    
    df_loaded = buffer.load_from_r2(dataset_id)
    assert df_loaded is not None
    assert len(df_loaded) == 10
    
    # Ensure stratified ratios are approximately preserved
    # In 10 samples, we expect ~8 zeros and ~2 ones
    val_counts = df_loaded["Churn"].value_counts().to_dict()
    assert val_counts.get(0, 0) == 8
    assert val_counts.get(1, 0) == 2

def test_continual_mtl_trainer_flow(tmp_path, monkeypatch):
    if not is_mtl_available():
        pytest.skip("PyTorch is not available or blocked by policy.")
    # Mock R2 storage
    local_storage = {}
    class MockStorage:
        def download_file(self, path):
            if path in local_storage:
                return local_storage[path]
            raise FileNotFoundError()
            
        def upload_file(self, content, path):
            local_storage[path] = content

    mock_storage_instance = MockStorage()
    import backend.app.core.training.continual_trainer as ct
    monkeypatch.setattr(ct, "storage", mock_storage_instance)
    
    # Mock MLflow sklearn load/log
    # Let's mock mlflow sklearn load_model to return a Pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    input_dim = 5
    mtl_model = MTLChurnModel()
    # Fit initial dummy
    X_init = np.random.randn(100, input_dim)
    y_init_bin = np.random.randint(0, 2, size=(100,))
    y_init_cpi = np.random.rand(100)
    mtl_model.fit(X_init, y_init_bin, y_init_cpi, epochs=2)
    
    preprocessor = StandardScaler()
    preprocessor.fit(X_init)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', mtl_model)
    ])
    
    # Mock mlflow methods
    class MockMlflow:
        def start_run(self, *args, **kwargs):
            class MockRun:
                info = type('obj', (object,), {'run_id': 'mock-run-123'})
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return MockRun()
            
        def set_tags(self, *args, **kwargs): pass
        def log_metric(self, *args, **kwargs): pass
        def log_artifact(self, *args, **kwargs): pass
        
        class Sklearn:
            def load_model(self, model_uri):
                return pipeline
                
            def log_model(self, *args, **kwargs): pass
            
        def __init__(self):
            self.sklearn = self.Sklearn()
            self.tracking = type('obj', (object,), {
                'MlflowClient': lambda: type('obj', (object,), {
                    'get_run': lambda r_id: type('obj', (object,), {
                        'data': type('obj', (object,), {
                            'metrics': {'best_roc_auc': 0.85}
                        })
                    })
                })
            })

    monkeypatch.setattr("mlflow.sklearn", MockMlflow.Sklearn())
    monkeypatch.setattr("mlflow.start_run", MockMlflow().start_run)
    monkeypatch.setattr("mlflow.set_tags", MockMlflow().set_tags)
    monkeypatch.setattr("mlflow.log_metric", MockMlflow().log_metric)
    monkeypatch.setattr("mlflow.log_artifact", MockMlflow().log_artifact)

    df_new = pd.DataFrame({
        "feat_1": np.random.randn(50),
        "feat_2": np.random.randn(50),
        "feat_3": np.random.randn(50),
        "feat_4": np.random.randn(50),
        "feat_5": np.random.randn(50),
        "Churn": np.random.randint(0, 2, size=(50,)),
        "cpi_score": np.random.rand(50)
    })
    
    trainer = ContinualMTLTrainer(lambda_ewc=10.0, mixing_ratio=0.2)
    # Populate initial buffer
    trainer.replay_buffer.update("test_id", df_new, "Churn")
    
    # Train
    model, final_pipeline, best_roc_auc, optimal_threshold = trainer.train(
        prior_model_uri="runs:/mock/model",
        dataset_id="test_id",
        df_new=df_new,
        feature_cols=["feat_1", "feat_2", "feat_3", "feat_4", "feat_5"],
        target_col="Churn",
        cpi_col="cpi_score",
        epochs=3
    )
    
    assert model is not None
    assert final_pipeline is not None
    assert 0.0 <= best_roc_auc <= 1.0
    assert 0.0 <= optimal_threshold <= 1.0
