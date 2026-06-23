import os
import pandas as pd
import pytest
import pandera as pa

from backend.app.core.profiler.orchestrator import run_profiling
from backend.app.core.profiler.column_profile import DataRole
from backend.app.core.pipeline.builder import build_pipeline
from backend.app.core.pipeline.schema_gen import generate_schema, save_schema
from backend.app.core.training.model_router import route_models
from backend.app.core.training.automl import run_automl
from backend.app.core.exceptions import PipelineBuilderError
from backend.app.core.storage import storage

# Set low trials for testing
os.environ["OPTUNA_N_TRIALS"] = "2"

def test_phase2_contract_leakage_fixture_rejected():
    # Load fixture and duplicate to ensure CV doesn't fail due to small size
    df = pd.read_csv("tests/fixtures/target_leakage.csv")
    df = pd.concat([df]*10, ignore_index=True) # 100 rows
    
    # 2. Run Profiling (Phase 1)
    profiles, suggested_target = run_profiling(df)
    assert suggested_target == "churn_target"
    
    # 3. Simulate user confirming DROP of leakage columns
    confirmed_profiles = []
    for p in profiles:
        if p.potential_leakage:
            p.inferred_role = DataRole.IGNORE
        confirmed_profiles.append(p)
        
    for p in confirmed_profiles:
        if p.name == "feature_x":
            p.inferred_role = DataRole.IGNORE

    with pytest.raises(PipelineBuilderError, match="No valid feature columns remaining after filtering"):
        build_pipeline(confirmed_profiles, suggested_target)

def test_phase2_contract_leakage_fixture_accepted():
    df = pd.read_csv("tests/fixtures/target_leakage.csv")
    df = pd.concat([df]*10, ignore_index=True) # 100 rows
    
    profiles, suggested_target = run_profiling(df)
    assert suggested_target == "churn_target"
    
    # Simulate user KEEPING leakage columns (ignoring the flag)
    confirmed_profiles = profiles
    
    # Build Pipeline - Should succeed
    pipeline = build_pipeline(confirmed_profiles, suggested_target)
    
    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps
    
    dataset_id = "test_dataset_leakage_keep"
    model_uri, out_schema_data = run_automl(df, confirmed_profiles, suggested_target, dataset_id)
    
    schema, metadata = out_schema_data
    assert isinstance(schema, pa.DataFrameSchema)
    assert isinstance(metadata, dict)
    
    schema_path, metadata_path = save_schema(schema, metadata, dataset_id, suggested_target)
    
    assert model_uri.startswith("runs:/")
    assert metadata_path.endswith("metadata.json")
    
    # Check NUMERIC is pa.Float
    for p in confirmed_profiles:
        if p.inferred_role == DataRole.NUMERIC and p.name in schema.columns:
            assert schema.columns[p.name].dtype == pa.Float

def test_phase2_contract_constant_fixture():
    df = pd.read_csv("tests/fixtures/constant_columns.csv")
    df = pd.concat([df]*10, ignore_index=True) # 100 rows
    
    import numpy as np
    df["valid_num"] = np.random.randn(len(df))
    
    profiles, suggested_target = run_profiling(df)
    assert suggested_target == "target"
    
    dataset_id = "test_dataset_constant"
    model_uri, out_schema_data = run_automl(df, profiles, suggested_target, dataset_id)
    
    schema, metadata = out_schema_data
    schema_path, metadata_path = save_schema(schema, metadata, dataset_id, suggested_target)
    
    assert model_uri.startswith("runs:/")
    assert suggested_target in schema_path
