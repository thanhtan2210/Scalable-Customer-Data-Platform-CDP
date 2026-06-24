# pyrefly: ignore [missing-import]
import pytest
import pandas as pd
import numpy as np
from hypothesis import given, settings, HealthCheck, strategies as st
from backend.app.core.profiler.target_synthesizer import synthesize_target
from backend.app.core.profiler.target_analysis import (
    SynthesisStrategy, ChurnColumnGroupItem, GroupRole
)
import backend.app.core.config as config

# Define a strategy to generate simple dataframes
# We need dataframes with a target column and several auxiliary columns
@st.composite
def dataframe_with_aux_cols(draw):
    # Number of rows
    n_rows = draw(st.integers(min_value=5, max_value=50))
    # Number of aux columns
    n_aux = draw(st.integers(min_value=0, max_value=5))
    
    data = {}
    # Target column (binary 0 or 1)
    data["target"] = draw(st.lists(st.sampled_from([0, 1]), min_size=n_rows, max_size=n_rows))
    
    # Generate aux columns
    for i in range(n_aux):
        col_type = draw(st.sampled_from(["numeric", "constant", "nan", "categorical"]))
        if col_type == "numeric":
            data[f"aux_{i}"] = draw(st.lists(st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False), min_size=n_rows, max_size=n_rows))
        elif col_type == "constant":
            val = draw(st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False))
            data[f"aux_{i}"] = [val] * n_rows
        elif col_type == "nan":
            data[f"aux_{i}"] = [np.nan] * n_rows
        else: # categorical
            data[f"aux_{i}"] = draw(st.lists(st.sampled_from(["A", "B", "C"]), min_size=n_rows, max_size=n_rows))
            
    df = pd.DataFrame(data)
    
    # Generate corresponding column group items
    group = [
        ChurnColumnGroupItem(name="target", correlation_with_target=1.0, group_role=GroupRole.PRIMARY)
    ]
    for i in range(n_aux):
        corr = draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))
        # Sometimes aux, sometimes duplicate, sometimes other roles
        role = draw(st.sampled_from([GroupRole.AUXILIARY, GroupRole.DUPLICATE]))
        group.append(ChurnColumnGroupItem(name=f"aux_{i}", correlation_with_target=corr, group_role=role))
        
    return df, group

@settings(max_examples=40, deadline=None)
@given(df_and_group=dataframe_with_aux_cols())
def test_cpi_values_always_in_range_0_1(df_and_group):
    df, group = df_and_group
    
    # Run synthesis
    cfg, cpi = synthesize_target(df, group, "target")
    
    if cpi is not None:
        # Check range with float tolerance
        min_val = float(cpi.min())
        max_val = float(cpi.max())
        assert min_val >= -1e-9, f"CPI min value {min_val} is less than 0"
        assert max_val <= 1.0 + 1e-9, f"CPI max value {max_val} is greater than 1"
        assert len(cpi) == len(df)

@settings(max_examples=40, deadline=None)
@given(df_and_group=dataframe_with_aux_cols())
def test_synthesis_strategy_determination(df_and_group):
    df, group = df_and_group
    
    # Run synthesis
    cfg, _ = synthesize_target(df, group, "target")
    
    # Count eligible columns
    eligible = [
        c for c in group
        if c.group_role in (GroupRole.AUXILIARY, GroupRole.DUPLICATE)
        and c.name != "target"
    ]
    
    if not eligible:
        assert cfg.strategy == SynthesisStrategy.NONE
    else:
        # Strategy must be either PCA or WEIGHTED
        assert cfg.strategy in (SynthesisStrategy.PCA, SynthesisStrategy.WEIGHTED)
        
        # Verify that if strategy is PCA, the eligible numeric cols must be >= CPI_MIN_COLUMNS
        if cfg.strategy == SynthesisStrategy.PCA:
            numeric_cols = []
            for c in eligible:
                if c.name in df.columns and pd.api.types.is_numeric_dtype(df[c.name]):
                    numeric_cols.append(c.name)
            assert len(numeric_cols) >= config.CPI_MIN_COLUMNS

@settings(max_examples=40, deadline=None)
@given(df_and_group=dataframe_with_aux_cols())
def test_cpi_confirmation_logic(df_and_group):
    df, group = df_and_group
    
    # Run synthesis
    cfg, cpi = synthesize_target(df, group, "target")
    
    eligible = [
        c for c in group
        if c.group_role in (GroupRole.AUXILIARY, GroupRole.DUPLICATE)
        and c.name != "target"
    ]
    
    if len(eligible) <= config.CPI_AUTO_THRESHOLD:
        assert cfg.requires_confirmation is False
        if len(eligible) > 0:
            assert cpi is not None
    else:
        assert cfg.requires_confirmation is True
        assert cpi is None

@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(df_and_group=dataframe_with_aux_cols())
def test_weighted_sum_always_normalized_properties(monkeypatch, df_and_group):
    df, group = df_and_group
    # Force WEIGHTED strategy by changing CPI_MIN_COLUMNS temporarily in target_synthesizer
    monkeypatch.setattr("backend.app.core.profiler.target_synthesizer.CPI_MIN_COLUMNS", 999)
    
    cfg, cpi = synthesize_target(df, group, "target")
    eligible = [
        c for c in group
        if c.group_role in (GroupRole.AUXILIARY, GroupRole.DUPLICATE)
        and c.name != "target"
    ]
    if eligible:
        assert cfg.strategy == SynthesisStrategy.WEIGHTED
        if len(eligible) <= config.CPI_AUTO_THRESHOLD:
            assert cpi is not None
            assert float(cpi.min()) >= -1e-9
            assert float(cpi.max()) <= 1.0 + 1e-9
