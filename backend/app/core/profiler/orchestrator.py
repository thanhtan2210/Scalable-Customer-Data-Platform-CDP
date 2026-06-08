import pandas as pd
from typing import List
from .column_profile import ColumnProfile
from .layer1_stats import profile_column
from .layer2_semantic import detect_semantic
from .layer3_llm import enrich_with_llm

def run_profiling(df: pd.DataFrame) -> List[ColumnProfile]:
    """Orchestrates the 3-layer profiling process for an entire DataFrame."""
    profiles = []

    for col_name in df.columns:
        series = df[col_name]
        
        # --- Layer 1: Stats (Mandatory) ---
        l1_res = profile_column(series)
        current_profile = {
            "name": col_name,
            "inferred_role": l1_res["role"],
            "transform_strategy": l1_res["strategy"],
            "impute_strategy": l1_res["impute"],
            "confidence": l1_res["confidence"],
            "layer_source": 1,
            "stats": l1_res["stats"]
        }

        # --- Layer 2: Semantic (Always run to refine) ---
        l2_res = detect_semantic(series, current_profile)
        current_profile.update(l2_res)

        # --- Layer 3: LLM (Optional, only for low confidence) ---
        if current_profile["confidence"] < 0.6:
            sample_vals = series.dropna().head(10).tolist()
            llm_res = enrich_with_llm(col_name, sample_vals, current_profile["stats"])
            
            if llm_res:
                current_profile["inferred_role"] = llm_res["role"]
                current_profile["transform_strategy"] = llm_res["transform"]
                current_profile["confidence"] = llm_res["confidence"]
                current_profile["layer_source"] = 3

        # Convert to Pydantic Model
        profiles.append(ColumnProfile(**current_profile))

    return profiles
