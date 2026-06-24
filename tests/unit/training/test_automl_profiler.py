import os
import numpy as np
import pandas as pd
from backend.app.core.profiler.orchestrator import run_profiling
from backend.app.core.profiler.column_profile import DataRole

def test_telco_churn_dataset():
    # Simulate Telco Churn
    df = pd.DataFrame({
        "customerID": [f"ID{i}" for i in range(100)],
        "gender": ["Male", "Female"] * 50,
        "MonthlyCharges": np.random.uniform(20.0, 120.0, 100),
        "Churn": ["Yes", "No", "No", "Yes"] * 25 # Binary, target keyword
    })
    
    profiles, suggested_target = run_profiling(df)
    
    assert len(profiles) == 4
    assert suggested_target.recommended_target == "Churn"
    
    target_prof = next(p for p in profiles if p.inferred_role == DataRole.TARGET)
    assert target_prof.name == "Churn"
    assert target_prof.transform_strategy == "label"
    
    id_prof = next(p for p in profiles if p.name == "customerID")
    assert id_prof.inferred_role == DataRole.ID
    assert id_prof.transform_strategy == "passthrough"

def test_credit_scoring_dataset():
    # Simulate completely different domain (Banking/Credit Scoring)
    df = pd.DataFrame({
        "loan_id": [f"L{i}" for i in range(100)],
        "income": np.random.uniform(30000, 100000, 100),
        "credit_score": np.random.randint(300, 850, 100),
        "default_status": [1, 0] * 50 # Binary, low entropy, position: end
    })
    
    profiles, suggested_target = run_profiling(df)
    
    assert suggested_target.recommended_target == "default_status"
    target_prof = next(p for p in profiles if p.inferred_role == DataRole.TARGET)
    assert target_prof.name == "default_status"
    
    # Test leakage logic
    df["future_leakage"] = df["default_status"] # Perfect correlation
    profiles_with_leakage, _ = run_profiling(df)
    
    leakage_prof = next(p for p in profiles_with_leakage if p.name == "future_leakage")
    assert leakage_prof.potential_leakage is True
    assert leakage_prof.leakage_score > 0.95

def test_edge_case_fixtures():
    # 1. Test Long Text fixture
    df_text = pd.read_csv("tests/fixtures/long_text.csv")
    profiles_text, _ = run_profiling(df_text)
    desc_prof = next(p for p in profiles_text if p.name == "description")
    assert desc_prof.inferred_role == DataRole.TEXT
    assert desc_prof.transform_strategy == "tfidf"
    
    # 2. Test Constant Columns fixture
    df_const = pd.read_csv("tests/fixtures/constant_columns.csv")
    profiles_const, _ = run_profiling(df_const)
    null_prof = next(p for p in profiles_const if p.name == "col_all_null")
    assert null_prof.inferred_role == DataRole.IGNORE
    assert null_prof.transform_strategy == "drop"
    
    # 3. Test Noisy IDs fixture
    df_ids = pd.read_csv("tests/fixtures/noisy_ids.csv")
    profiles_ids, _ = run_profiling(df_ids)
    id_prof = next(p for p in profiles_ids if p.name == "id")
    assert id_prof.inferred_role == DataRole.ID
