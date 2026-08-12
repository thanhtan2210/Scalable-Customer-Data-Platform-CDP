import os
# Set STORAGE_MODE to local before importing modules that configure storage
os.environ["STORAGE_MODE"] = "local"

import pytest
import pandas as pd
import numpy as np
import inspect
from unittest.mock import MagicMock, patch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import pandera as pa

from backend.app.core.profiler.column_profile import ColumnProfile, DataRole
from backend.app.core.profiler.orchestrator import run_profiling
from backend.app.core.pipeline.builder import build_pipeline
from backend.app.core.pipeline.transforms.registry import get_transformer, WinsorizerTransformer
from backend.app.core.pipeline.schema_gen import generate_schema
from backend.app.core.training.model_router import route_models
from backend.app.core.training import model_router
from backend.app.core.training.automl import run_automl
from backend.app.core.exceptions import PipelineBuilderError

# 1. Transform registry đầy đủ
def test_transform_registry():
    """
    Transform registry đầy đủ:
    - Tất cả keys có trong registry: standard, log, power, ohe, ordinal, tfidf, cyclical, passthrough, drop
    - "standard" là Pipeline có WinsorizerTransformer
    - "ohe" có max_categories=10
    """
    keys = ["standard", "log", "power", "ohe", "ordinal", "tfidf", "cyclical", "passthrough", "drop"]
    for k in keys:
        t = get_transformer(k)
        assert t is not None, f"Key '{k}' không tồn tại hoặc trả về None trong registry!"
        
    # "standard" là Pipeline có WinsorizerTransformer
    standard_trans = get_transformer("standard")
    assert isinstance(standard_trans, Pipeline), "standard transform phải là một Pipeline!"
    assert any(isinstance(step[1], WinsorizerTransformer) for step in standard_trans.steps), \
        "standard transform phải chứa WinsorizerTransformer!"
        
    # "ohe" có max_categories=10
    ohe_trans = get_transformer("ohe")
    assert isinstance(ohe_trans, OneHotEncoder), "ohe transform phải là OneHotEncoder!"
    assert ohe_trans.max_categories == 10, f"ohe max_categories phải bằng 10, nhận được {ohe_trans.max_categories}!"


# 2. Pipeline builder — confirmed_profiles
def test_pipeline_builder():
    """
    Pipeline builder — confirmed_profiles:
    - Cột role ID/IGNORE/TARGET bị loại khỏi pipeline
    - Cột potential_leakage=True KHÔNG bị tự động loại
    - Nếu không còn cột nào → raise PipelineBuilderError
    """
    profiles = [
        ColumnProfile(name="id_col", inferred_dtype="int64", inferred_role=DataRole.ID, confidence_score=0.9, null_pct=0.0, unique_count=100, entropy=2.0),
        ColumnProfile(name="ignore_col", inferred_dtype="int64", inferred_role=DataRole.IGNORE, confidence_score=0.9, null_pct=0.0, unique_count=100, entropy=2.0),
        ColumnProfile(name="target_col", inferred_dtype="int64", inferred_role=DataRole.TARGET, confidence_score=0.9, null_pct=0.0, unique_count=2, entropy=0.5),
        ColumnProfile(name="leak_col", inferred_dtype="float64", inferred_role=DataRole.NUMERIC, confidence_score=0.9, null_pct=0.0, unique_count=100, entropy=2.0, potential_leakage=True, impute_strategy="median", transform_strategy="standard"),
        ColumnProfile(name="feature_col", inferred_dtype="float64", inferred_role=DataRole.NUMERIC, confidence_score=0.9, null_pct=0.0, unique_count=100, entropy=2.0, impute_strategy="median", transform_strategy="standard"),
    ]
    
    pipeline = build_pipeline(profiles, target_col="target_col")
    preprocessor = pipeline.named_steps["preprocessor"]
    
    included_cols = []
    for transformer_step in preprocessor.transformers:
        # Each step is (name, transformer, columns)
        included_cols.extend(transformer_step[2])
        
    # Cột ID, IGNORE, TARGET phải bị loại
    assert "id_col" not in included_cols, "Cột id_col (role ID) không được xuất hiện trong pipeline features!"
    assert "ignore_col" not in included_cols, "Cột ignore_col (role IGNORE) không được xuất hiện trong pipeline features!"
    assert "target_col" not in included_cols, "Cột target_col (role TARGET) không được xuất hiện trong pipeline features!"
    
    # Cột potential_leakage=True KHÔNG bị tự động loại
    assert "leak_col" in included_cols, "Cột leak_col (potential_leakage=True) không được tự động loại khỏi builder!"
    assert "feature_col" in included_cols, "Cột feature_col phải nằm trong pipeline features!"
    
    # Nếu không còn cột nào → raise PipelineBuilderError
    invalid_profiles = [
        ColumnProfile(name="id_col", inferred_dtype="int64", inferred_role=DataRole.ID, confidence_score=0.9, null_pct=0.0, unique_count=100, entropy=2.0),
        ColumnProfile(name="target_col", inferred_dtype="int64", inferred_role=DataRole.TARGET, confidence_score=0.9, null_pct=0.0, unique_count=2, entropy=0.5),
    ]
    with pytest.raises(PipelineBuilderError):
        build_pipeline(invalid_profiles, target_col="target_col")


# 3. Schema gen output
def test_schema_gen_output():
    """
    Schema gen output:
    - generate_schema() trả về tuple (schema, metadata)
    - NUMERIC → pa.Float (không phải pa.Int)
    - metadata có đủ keys: inferred_role, transform_strategy, impute_strategy cho mỗi cột
    """
    profiles = [
        ColumnProfile(name="num_col", inferred_dtype="int64", inferred_role=DataRole.NUMERIC, confidence_score=0.9, null_pct=0.0, unique_count=100, entropy=2.0, impute_strategy="median", transform_strategy="standard"),
        ColumnProfile(name="cat_col", inferred_dtype="object", inferred_role=DataRole.CATEGORICAL, confidence_score=0.9, null_pct=0.0, unique_count=5, entropy=1.0, impute_strategy="mode", transform_strategy="ohe"),
    ]
    
    res = generate_schema(profiles, dataset_id="test_dataset", target_col="target_col")
    assert isinstance(res, tuple) and len(res) == 2, "generate_schema phải trả về một tuple gồm 2 phần tử!"
    
    schema, metadata = res
    assert isinstance(schema, pa.DataFrameSchema), "Phần tử đầu tiên trả về phải là pa.DataFrameSchema!"
    assert isinstance(metadata, dict), "Phần tử thứ hai trả về phải là dict!"
    
    # NUMERIC -> pa.Float
    dtype_str = str(schema.columns["num_col"].dtype).lower()
    assert "float" in dtype_str, f"Cột NUMERIC phải được ánh xạ sang kiểu Float, nhận được {dtype_str}!"
    
    # metadata checks
    assert "columns" in metadata, "metadata phải chứa trường 'columns'!"
    for col_name in ["num_col", "cat_col"]:
        assert col_name in metadata["columns"], f"Thiếu cột {col_name} trong metadata columns!"
        col_meta = metadata["columns"][col_name]
        assert "inferred_role" in col_meta, f"Thiếu 'inferred_role' trong metadata của {col_name}!"
        assert "transform_strategy" in col_meta, f"Thiếu 'transform_strategy' trong metadata của {col_name}!"
        assert "impute_strategy" in col_meta, f"Thiếu 'impute_strategy' trong metadata của {col_name}!"


# 4. Model router — domain agnostic
def test_model_router_domain_agnostic():
    """
    Model router — domain agnostic:
    - Không có string tên domain (như churn, telco, customer, v.v.) trong logic source code
    """
    source_code = inspect.getsource(model_router)
    domain_words = ["churn", "telco", "marketing", "retention", "customer"]
    for word in domain_words:
        assert word not in source_code.lower(), f"Phát hiện từ khóa domain cụ thể '{word}' trong source code model_router.py!"

def test_model_router_routing():
    """
    Model router routing rules:
    - n_rows < 1000 → LogisticRegression
    - has_text = True → LogisticRegression
    """
    # Test case has_text = True
    df_text = pd.DataFrame({"feat": ["free text data"] * 1000})
    profiles_text = [
        ColumnProfile(name="feat", inferred_dtype="object", inferred_role=DataRole.TEXT, confidence_score=0.9, null_pct=0.0, unique_count=100, entropy=2.0)
    ]
    models_text = route_models(df_text, profiles_text)
    assert any(m["class"] == LogisticRegression for m in models_text), \
        "has_text=True phải route tới LogisticRegression!"

    # Test case n_rows < 1000
    df_small = pd.DataFrame({"feat": [1.0] * 600})
    profiles_small = [
        ColumnProfile(name="feat", inferred_dtype="float64", inferred_role=DataRole.NUMERIC, confidence_score=0.9, null_pct=0.0, unique_count=100, entropy=2.0)
    ]
    models_small = route_models(df_small, profiles_small)
    assert any(m["class"] == LogisticRegression for m in models_small), \
        "n_rows < 1000 phải route tới LogisticRegression!"


# 5. AutoML integration
@patch("backend.app.core.training.automl.mlflow")
def test_automl_integration(mock_mlflow):
    """
    AutoML integration:
    - run_automl() nhận df + profiles + target_col
    - Dùng StratifiedKFold(n_splits=5, shuffle=True)
    - Đọc n_trials từ OPTUNA_N_TRIALS env var
    - Sau khi chạy: optimal_threshold nằm trong (0, 1)
    """
    # Chuẩn bị dữ liệu
    df = pd.DataFrame({
        "feature_num": np.random.randn(100),
        "feature_cat": np.random.choice(["A", "B"], size=100),
        "target": np.random.choice([0, 1], size=100)
    })
    
    profiles = [
        ColumnProfile(name="feature_num", inferred_dtype="float64", inferred_role=DataRole.NUMERIC, confidence_score=0.9, null_pct=0.0, unique_count=100, entropy=2.0, impute_strategy="median", transform_strategy="standard"),
        ColumnProfile(name="feature_cat", inferred_dtype="object", inferred_role=DataRole.CATEGORICAL, confidence_score=0.9, null_pct=0.0, unique_count=2, entropy=1.0, impute_strategy="mode", transform_strategy="ohe"),
        ColumnProfile(name="target", inferred_dtype="int64", inferred_role=DataRole.TARGET, confidence_score=1.0, null_pct=0.0, unique_count=2, entropy=1.0)
    ]
    
    # Thiết lập env var cho OPTUNA_N_TRIALS
    os.environ["OPTUNA_N_TRIALS"] = "2"
    
    import optuna
    from sklearn.model_selection import StratifiedKFold
    
    original_optimize = optuna.Study.optimize
    
    # Spy optuna Study.optimize và StratifiedKFold
    with patch.object(optuna.Study, "optimize", autospec=True) as mock_optimize, \
         patch("backend.app.core.training.automl.StratifiedKFold", wraps=StratifiedKFold) as mock_skf:
         
        # Ủy quyền cho optimize thực tế chạy để tính được optimal_threshold thật
        def spy_optimize(self, func, n_trials, timeout, **kwargs):
            assert n_trials == 2, f"n_trials nhận được là {n_trials}, kỳ vọng đọc từ env var là 2!"
            return original_optimize(self, func, n_trials=n_trials, timeout=timeout, **kwargs)
            
        mock_optimize.side_effect = spy_optimize
        
        # Chạy AutoML
        model_uri, schema_res = run_automl(df, profiles, target_col="target", dataset_id="test_ds")
        
        # Kiểm tra StratifiedKFold được khởi tạo đúng tham số
        mock_skf.assert_called()
        skf_kwargs = mock_skf.call_args[1]
        assert skf_kwargs.get("n_splits") == 5, f"Kỳ vọng n_splits=5, nhận được {skf_kwargs.get('n_splits')}"
        assert skf_kwargs.get("shuffle") is True, f"Kỳ vọng shuffle=True, nhận được {skf_kwargs.get('shuffle')}"
        
        # Kiểm tra optimal_threshold được ghi nhận và nằm trong khoảng (0, 1)
        # Tìm trong mock_mlflow.log_metric cuộc gọi cho "optimal_threshold"
        threshold_calls = [
            call for call in mock_mlflow.log_metric.call_args_list 
            if call[0][0] == "optimal_threshold"
        ]
        assert len(threshold_calls) > 0, "Không tìm thấy cuộc gọi log_metric('optimal_threshold')!"
        optimal_threshold = threshold_calls[0][0][1]
        assert 0.0 < optimal_threshold < 1.0, f"optimal_threshold ({optimal_threshold}) phải nằm trong khoảng (0, 1)!"


# 6. Contract test Phase 1 → Phase 2
def test_contract_phase1_phase2():
    """
    Contract test Phase 1 → Phase 2:
    - run_profiling(df) → build_pipeline(profiles, target)
    - pipeline.fit(X_train, y_train) không raise exception
    """
    df = pd.DataFrame({
        "CustomerID": [f"ID_{i}" for i in range(100)],
        "Age": np.random.randint(18, 80, size=100).astype(float),
        "Gender": np.random.choice(["Male", "Female"], size=100),
        "Tenure": np.random.randint(0, 72, size=100).astype(float),
        "Churn Value": np.random.choice([0, 1], size=100)
    })
    
    profiles, suggested_target = run_profiling(df)
    target_name = suggested_target.recommended_target
    assert target_name in ["Churn Value", "Gender"], f"Gợi ý target không hợp lý: {target_name}"
    
    pipeline = build_pipeline(profiles, target_col=target_name)
    
    feature_cols = [
        p.name for p in profiles 
        if p.name != target_name and p.inferred_role not in [DataRole.ID, DataRole.IGNORE, DataRole.TARGET]
    ]
    
    X = df[feature_cols]
    y = df[target_name]
    
    try:
        pipeline.fit(X, y)
    except Exception as e:
        pytest.fail(f"Contract test thất bại! pipeline.fit ném ra exception: {e}")


# 7. Auxiliary columns inclusion in pipeline
def test_auxiliary_cols_included():
    """
    Verify that auxiliary_cols are added as a passthrough transformer step in build_pipeline.
    """
    profiles = [
        ColumnProfile(name="feature_col", inferred_dtype="float64", inferred_role=DataRole.NUMERIC, confidence_score=0.9, null_pct=0.0, unique_count=100, entropy=2.0, impute_strategy="median", transform_strategy="standard"),
        ColumnProfile(name="target_col", inferred_dtype="int64", inferred_role=DataRole.TARGET, confidence_score=0.9, null_pct=0.0, unique_count=2, entropy=0.5),
    ]
    
    pipeline = build_pipeline(profiles, target_col="target_col", auxiliary_cols=["cpi_score"])
    preprocessor = pipeline.named_steps["preprocessor"]
    
    # Check that auxiliary_passthrough step is in transformers list
    step_names = [step[0] for step in preprocessor.transformers]
    assert "auxiliary_passthrough" in step_names, "Thiếu bước auxiliary_passthrough trong preprocessor!"
    
    aux_step = next(step for step in preprocessor.transformers if step[0] == "auxiliary_passthrough")
    assert aux_step[1] == "passthrough"
    assert aux_step[2] == ["cpi_score"]

