import pytest
import pandas as pd
import numpy as np
from backend.app.core.profiler.target_synthesizer import synthesize_target
from backend.app.core.profiler.target_analysis import (
    SynthesisStrategy, ChurnColumnGroupItem, GroupRole
)
import backend.app.core.config as config

def test_pca_strategy_selection():
    """1. PCA Strategy Selection: >= 2 numeric aux cols -> strategy = PCA"""
    # Create two highly correlated columns
    t = np.array([0, 1] * 50)
    aux1 = t * 1.5 + np.random.normal(0, 0.1, 100)
    aux2 = t * 2.0 + np.random.normal(0, 0.1, 100)
    df = pd.DataFrame({
        "target": t,
        "aux1": aux1,
        "aux2": aux2
    })

    group = [
        ChurnColumnGroupItem(name="target", correlation_with_target=1.0, group_role=GroupRole.PRIMARY),
        ChurnColumnGroupItem(name="aux1", correlation_with_target=0.8, group_role=GroupRole.AUXILIARY),
        ChurnColumnGroupItem(name="aux2", correlation_with_target=0.7, group_role=GroupRole.AUXILIARY)
    ]

    cfg, cpi = synthesize_target(df, group, "target")
    assert cfg.strategy == SynthesisStrategy.PCA
    assert cfg.requires_confirmation is False  # 2 cols <= threshold (2)
    assert cpi is not None
    assert len(cpi) == 100

def test_weighted_fallback_single_col():
    """2. Weighted fallback for single col: 1 aux col -> strategy = WEIGHTED"""
    df = pd.DataFrame({
        "target": [0, 1] * 50,
        "aux1": [0, 2] * 50
    })

    group = [
        ChurnColumnGroupItem(name="target", correlation_with_target=1.0, group_role=GroupRole.PRIMARY),
        ChurnColumnGroupItem(name="aux1", correlation_with_target=0.8, group_role=GroupRole.AUXILIARY)
    ]

    cfg, cpi = synthesize_target(df, group, "target")
    assert cfg.strategy == SynthesisStrategy.WEIGHTED
    assert cfg.requires_confirmation is False
    assert cpi is not None
    assert cfg.weights is not None
    assert len(cfg.weights) == 1
    assert cfg.weights[0].name == "aux1"

def test_no_synthesis_zero_aux():
    """3. No synthesis with zero aux: 0 aux cols -> strategy = NONE"""
    df = pd.DataFrame({
        "target": [0, 1] * 50
    })

    group = [
        ChurnColumnGroupItem(name="target", correlation_with_target=1.0, group_role=GroupRole.PRIMARY)
    ]

    cfg, cpi = synthesize_target(df, group, "target")
    assert cfg.strategy == SynthesisStrategy.NONE
    assert cfg.requires_confirmation is False
    assert cpi is None

def test_cpi_output_range():
    """4. CPI Output Range: All CPI values in [0, 1]"""
    t = np.array([0, 1] * 50)
    aux1 = t * 10.0 + 5.0
    aux2 = t * -5.0 + 20.0
    df = pd.DataFrame({
        "target": t,
        "aux1": aux1,
        "aux2": aux2
    })

    group = [
        ChurnColumnGroupItem(name="target", correlation_with_target=1.0, group_role=GroupRole.PRIMARY),
        ChurnColumnGroupItem(name="aux1", correlation_with_target=0.8, group_role=GroupRole.AUXILIARY),
        ChurnColumnGroupItem(name="aux2", correlation_with_target=0.7, group_role=GroupRole.AUXILIARY)
    ]

    cfg, cpi = synthesize_target(df, group, "target")
    assert cpi is not None
    assert float(cpi.min()) >= 0.0
    assert float(cpi.max()) <= 1.0

def test_pca_low_variance_fallback():
    """5. PCA Low Variance Fallback: variance < threshold -> WEIGHTED"""
    # Independent random noise
    df = pd.DataFrame({
        "target": [0, 1] * 50,
        "aux1": np.random.normal(0, 1, 100),
        "aux2": np.random.normal(0, 1, 100)
    })

    group = [
        ChurnColumnGroupItem(name="target", correlation_with_target=1.0, group_role=GroupRole.PRIMARY),
        ChurnColumnGroupItem(name="aux1", correlation_with_target=0.5, group_role=GroupRole.AUXILIARY),
        ChurnColumnGroupItem(name="aux2", correlation_with_target=0.5, group_role=GroupRole.AUXILIARY)
    ]

    orig_threshold = config.CPI_VARIANCE_THRESHOLD
    config.CPI_VARIANCE_THRESHOLD = 0.99  # Require extremely high variance
    try:
        cfg, cpi = synthesize_target(df, group, "target")
        assert cfg.strategy == SynthesisStrategy.WEIGHTED
        assert cpi is not None
    finally:
        config.CPI_VARIANCE_THRESHOLD = orig_threshold

def test_auto_synthesize_threshold():
    """6. Auto Synthesize Threshold: <= 2 cols -> requires_confirmation = False, CPI returned"""
    df = pd.DataFrame({
        "target": [0, 1] * 50,
        "aux1": [0, 1] * 50,
        "aux2": [1, 0] * 50
    })

    group = [
        ChurnColumnGroupItem(name="target", correlation_with_target=1.0, group_role=GroupRole.PRIMARY),
        ChurnColumnGroupItem(name="aux1", correlation_with_target=0.8, group_role=GroupRole.AUXILIARY),
        ChurnColumnGroupItem(name="aux2", correlation_with_target=0.6, group_role=GroupRole.AUXILIARY)
    ]

    orig_auto_threshold = config.CPI_AUTO_THRESHOLD
    config.CPI_AUTO_THRESHOLD = 2
    try:
        cfg, cpi = synthesize_target(df, group, "target")
        assert cfg.requires_confirmation is False
        assert cpi is not None
    finally:
        config.CPI_AUTO_THRESHOLD = orig_auto_threshold

def test_user_confirm_many_cols():
    """7. User Confirm Many Cols: >= 3 cols -> requires_confirmation = True, CPI = None"""
    df = pd.DataFrame({
        "target": [0, 1] * 50,
        "aux1": [0, 1] * 50,
        "aux2": [1, 0] * 50,
        "aux3": [0, 1] * 50
    })

    group = [
        ChurnColumnGroupItem(name="target", correlation_with_target=1.0, group_role=GroupRole.PRIMARY),
        ChurnColumnGroupItem(name="aux1", correlation_with_target=0.8, group_role=GroupRole.AUXILIARY),
        ChurnColumnGroupItem(name="aux2", correlation_with_target=0.6, group_role=GroupRole.AUXILIARY),
        ChurnColumnGroupItem(name="aux3", correlation_with_target=0.5, group_role=GroupRole.AUXILIARY)
    ]

    orig_auto_threshold = config.CPI_AUTO_THRESHOLD
    config.CPI_AUTO_THRESHOLD = 2
    try:
        cfg, cpi = synthesize_target(df, group, "target")
        assert cfg.requires_confirmation is True
        assert cpi is None
    finally:
        config.CPI_AUTO_THRESHOLD = orig_auto_threshold

def test_weight_proportional_to_correlation():
    """8. Weight proportional to correlation: weights match correlation_with_target"""
    df = pd.DataFrame({
        "target": [0, 1] * 50,
        "aux1": [0, 1] * 50
    })

    group = [
        ChurnColumnGroupItem(name="target", correlation_with_target=1.0, group_role=GroupRole.PRIMARY),
        # Weighted is selected because only 1 aux col is present
        ChurnColumnGroupItem(name="aux1", correlation_with_target=0.45, group_role=GroupRole.AUXILIARY)
    ]

    cfg, cpi = synthesize_target(df, group, "target")
    assert cfg.strategy == SynthesisStrategy.WEIGHTED
    assert cfg.weights is not None
    assert cfg.weights[0].weight == 0.45


def test_pca_synthesis_constant_column():
    """9. PCA constant column: verify max_val <= min_val handles constant values gracefully"""
    df = pd.DataFrame({
        "target": [0, 1] * 10,
        "aux1": [5.0] * 20,
        "aux2": [1.0, 2.0] * 10
    })
    group = [
        ChurnColumnGroupItem(name="target", correlation_with_target=1.0, group_role=GroupRole.PRIMARY),
        ChurnColumnGroupItem(name="aux1", correlation_with_target=0.8, group_role=GroupRole.AUXILIARY),
        ChurnColumnGroupItem(name="aux2", correlation_with_target=0.7, group_role=GroupRole.AUXILIARY)
    ]
    cfg, cpi = synthesize_target(df, group, "target")
    # Should still execute PCA and return values in range [0, 1]
    assert cfg.strategy == SynthesisStrategy.PCA
    assert cpi is not None
    assert len(cpi) == 20
    assert float(cpi.min()) >= 0.0
    assert float(cpi.max()) <= 1.0


def test_pca_synthesis_cpi_constant(monkeypatch):
    """10. PCA CPI constant: verify cpi_max <= cpi_min handles constant CPI gracefully"""
    df = pd.DataFrame({
        "target": [0, 1] * 10,
        "aux1": [1.0, 2.0] * 10,
        "aux2": [5.0, 10.0] * 10
    })
    group = [
        ChurnColumnGroupItem(name="target", correlation_with_target=1.0, group_role=GroupRole.PRIMARY),
        ChurnColumnGroupItem(name="aux1", correlation_with_target=0.8, group_role=GroupRole.AUXILIARY),
        ChurnColumnGroupItem(name="aux2", correlation_with_target=0.7, group_role=GroupRole.AUXILIARY)
    ]
    
    class MockPCA:
        def __init__(self, n_components):
            self.n_components = n_components
            self.explained_variance_ratio_ = np.array([1.0])
        def fit_transform(self, X):
            return np.zeros((len(X), 1))
            
    monkeypatch.setattr("backend.app.core.profiler.target_synthesizer.PCA", MockPCA)
    
    cfg, cpi = synthesize_target(df, group, "target")
    assert cfg.strategy == SynthesisStrategy.PCA
    assert cpi is not None
    assert np.allclose(cpi.values, 0.0)


def test_pca_synthesis_negative_correlation_flip():
    """11. PCA negative correlation flip: CPI is flipped when corr < 0"""
    # We construct aux columns negatively correlated with target
    target = np.array([0, 1] * 50)
    # aux1 and aux2 are highly negatively correlated with target
    aux1 = -2.0 * target + np.random.normal(0, 0.01, 100)
    aux2 = -1.5 * target + np.random.normal(0, 0.01, 100)
    df = pd.DataFrame({
        "target": target,
        "aux1": aux1,
        "aux2": aux2
    })
    group = [
        ChurnColumnGroupItem(name="target", correlation_with_target=1.0, group_role=GroupRole.PRIMARY),
        ChurnColumnGroupItem(name="aux1", correlation_with_target=-0.8, group_role=GroupRole.AUXILIARY),
        ChurnColumnGroupItem(name="aux2", correlation_with_target=-0.7, group_role=GroupRole.AUXILIARY)
    ]
    cfg, cpi = synthesize_target(df, group, "target")
    assert cfg.strategy == SynthesisStrategy.PCA
    assert cpi is not None
    # Correlation with target should be positive after flip
    corr = np.corrcoef(cpi.values, target)[0, 1]
    assert corr > 0


def test_pca_synthesis_all_nan_column():
    """12. PCA all-NaN column: verify column with only NaNs is filled with 0.0 gracefully"""
    df = pd.DataFrame({
        "target": [0, 1] * 10,
        "aux1": [np.nan] * 20,
        "aux2": [1.0, 2.0] * 10
    })
    group = [
        ChurnColumnGroupItem(name="target", correlation_with_target=1.0, group_role=GroupRole.PRIMARY),
        ChurnColumnGroupItem(name="aux1", correlation_with_target=0.8, group_role=GroupRole.AUXILIARY),
        ChurnColumnGroupItem(name="aux2", correlation_with_target=0.7, group_role=GroupRole.AUXILIARY)
    ]
    cfg, cpi = synthesize_target(df, group, "target")
    assert cfg.strategy == SynthesisStrategy.PCA
    assert cpi is not None
    assert float(cpi.min()) >= 0.0
    assert float(cpi.max()) <= 1.0


def test_weighted_synthesis_zscore_method():
    """13. Weighted synthesis with z-score normalization method"""
    df = pd.DataFrame({
        "target": [0, 1] * 10,
        "aux1": [1.0, 5.0, 10.0, 20.0] * 5
    })
    # Force WEIGHTED strategy by having only 1 aux col
    group = [
        ChurnColumnGroupItem(name="target", correlation_with_target=1.0, group_role=GroupRole.PRIMARY),
        ChurnColumnGroupItem(name="aux1", correlation_with_target=0.8, group_role=GroupRole.AUXILIARY)
    ]
    cfg, cpi = synthesize_target(df, group, "target")
    # Manually configure zscore normalize method
    cfg.weights[0].normalize_method = "zscore"
    
    from backend.app.core.profiler.target_synthesizer import _weighted_synthesis
    cpi_z = _weighted_synthesis(df, cfg.weights)
    assert cpi_z is not None
    assert len(cpi_z) == 20
    assert float(cpi_z.min()) >= 0.0
    assert float(cpi_z.max()) <= 1.0


def test_weighted_synthesis_zscore_constant():
    """14. Weighted synthesis z-score constant column: std is 0"""
    df = pd.DataFrame({
        "target": [0, 1] * 10,
        "aux1": [10.0] * 20
    })
    # Force WEIGHTED
    group = [
        ChurnColumnGroupItem(name="target", correlation_with_target=1.0, group_role=GroupRole.PRIMARY),
        ChurnColumnGroupItem(name="aux1", correlation_with_target=0.8, group_role=GroupRole.AUXILIARY)
    ]
    cfg, _ = synthesize_target(df, group, "target")
    cfg.weights[0].normalize_method = "zscore"
    
    from backend.app.core.profiler.target_synthesizer import _weighted_synthesis
    cpi_z = _weighted_synthesis(df, cfg.weights)
    assert cpi_z is not None
    assert np.allclose(cpi_z.values, 0.0)


def test_weighted_synthesis_unknown_normalize_method():
    """15. Weighted synthesis unknown normalization method: fallback to raw fillna(0.0)"""
    df = pd.DataFrame({
        "target": [0, 1] * 10,
        "aux1": [1.0, 2.0] * 10
    })
    group = [
        ChurnColumnGroupItem(name="target", correlation_with_target=1.0, group_role=GroupRole.PRIMARY),
        ChurnColumnGroupItem(name="aux1", correlation_with_target=0.8, group_role=GroupRole.AUXILIARY)
    ]
    cfg, _ = synthesize_target(df, group, "target")
    cfg.weights[0].normalize_method = "unknown_method_abc"
    
    from backend.app.core.profiler.target_synthesizer import _weighted_synthesis
    cpi_res = _weighted_synthesis(df, cfg.weights)
    assert cpi_res is not None
    assert len(cpi_res) == 20
    # Values should be normalized from fillna(0) raw inputs
    assert float(cpi_res.min()) >= 0.0
    assert float(cpi_res.max()) <= 1.0


def test_weighted_synthesis_zero_total_weight():
    """16. Weighted synthesis with zero total weight: weights sum to 0"""
    df = pd.DataFrame({
        "target": [0, 1] * 10,
        "aux1": [1.0, 2.0] * 10
    })
    group = [
        ChurnColumnGroupItem(name="target", correlation_with_target=1.0, group_role=GroupRole.PRIMARY),
        ChurnColumnGroupItem(name="aux1", correlation_with_target=0.8, group_role=GroupRole.AUXILIARY)
    ]
    cfg, _ = synthesize_target(df, group, "target")
    cfg.weights[0].weight = 0.0
    
    from backend.app.core.profiler.target_synthesizer import _weighted_synthesis
    cpi_res = _weighted_synthesis(df, cfg.weights)
    assert cpi_res is not None
    assert np.allclose(cpi_res.values, 0.0)


def test_weighted_synthesis_missing_column():
    """17. Weighted synthesis with column missing from DataFrame"""
    df = pd.DataFrame({
        "target": [0, 1] * 10,
        # aux1 is missing!
        "aux2": [1.0, 2.0] * 10
    })
    group = [
        ChurnColumnGroupItem(name="target", correlation_with_target=1.0, group_role=GroupRole.PRIMARY),
        ChurnColumnGroupItem(name="aux1", correlation_with_target=0.8, group_role=GroupRole.AUXILIARY),
        ChurnColumnGroupItem(name="aux2", correlation_with_target=0.6, group_role=GroupRole.AUXILIARY)
    ]
    cfg, cpi = synthesize_target(df, group, "target")
    # Will fallback to weighted because aux1 is missing, only aux2 remains
    assert cfg.strategy == SynthesisStrategy.WEIGHTED
    assert cpi is not None
    assert len(cpi) == 20


def test_pca_synthesis_one_row():
    """18. PCA synthesis with single row DataFrame: handles PCA gracefully"""
    df = pd.DataFrame({
        "target": [1],
        "aux1": [5.0],
        "aux2": [10.0]
    })
    group = [
        ChurnColumnGroupItem(name="target", correlation_with_target=1.0, group_role=GroupRole.PRIMARY),
        ChurnColumnGroupItem(name="aux1", correlation_with_target=0.8, group_role=GroupRole.AUXILIARY),
        ChurnColumnGroupItem(name="aux2", correlation_with_target=0.7, group_role=GroupRole.AUXILIARY)
    ]
    # Check if we handle single row correctly without ValueError
    try:
        cfg, cpi = synthesize_target(df, group, "target")
        if cpi is not None:
            assert len(cpi) == 1
    except ValueError as e:
        pytest.fail(f"PCA synthesis failed on 1 row: {e}")


def test_pca_synthesis_non_numeric_target():
    """19. PCA synthesis with non-numeric target: factorize path"""
    df = pd.DataFrame({
        "target": ["Yes", "No", "Yes", "No"] * 5,
        "aux1": [1.0, 2.0] * 10,
        "aux2": [5.0, 10.0] * 10
    })
    group = [
        ChurnColumnGroupItem(name="target", correlation_with_target=1.0, group_role=GroupRole.PRIMARY),
        ChurnColumnGroupItem(name="aux1", correlation_with_target=0.8, group_role=GroupRole.AUXILIARY),
        ChurnColumnGroupItem(name="aux2", correlation_with_target=0.7, group_role=GroupRole.AUXILIARY)
    ]
    cfg, cpi = synthesize_target(df, group, "target")
    assert cfg.strategy == SynthesisStrategy.PCA
    assert cpi is not None
    assert len(cpi) == 20


def test_weighted_synthesis_binary_encode_mode_fallback(monkeypatch):
    """20. Weighted synthesis binary encode: mock stats.mode failure to trigger fallback to mode_val=0"""
    df = pd.DataFrame({
        "target": [0, 1] * 10,
        "aux1": ["A", "B", "A", "C"] * 5
    })
    group = [
        ChurnColumnGroupItem(name="target", correlation_with_target=1.0, group_role=GroupRole.PRIMARY),
        ChurnColumnGroupItem(name="aux1", correlation_with_target=0.8, group_role=GroupRole.AUXILIARY)
    ]
    cfg, _ = synthesize_target(df, group, "target")
    assert cfg.weights[0].normalize_method == "binary_encode"
    
    # Mock stats.mode to raise an Exception
    import scipy.stats as stats
    def mock_mode(*args, **kwargs):
        raise ValueError("Simulated mode failure")
    
    monkeypatch.setattr(stats, "mode", mock_mode)
    
    from backend.app.core.profiler.target_synthesizer import _weighted_synthesis
    cpi_res = _weighted_synthesis(df, cfg.weights)
    assert cpi_res is not None
    assert len(cpi_res) == 20
