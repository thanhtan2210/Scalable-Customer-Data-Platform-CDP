import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import Tuple

from .column_profile import ColumnProfile, DataRole
from .layer1_stats import profile_column
from .layer2_semantic import detect_semantic
from .layer3_llm import refine_with_llm

# Recipe Table Mapping (Phase 1 Goal 2)
ROLE_RECIPES = {
    DataRole.ID: {"impute": "drop", "transform": "passthrough"},
    DataRole.TARGET: {"impute": "drop_row", "transform": "label"},
    DataRole.NUMERIC: {"impute": "median", "transform": "standard"},
    DataRole.CATEGORICAL: {"impute": "mode", "transform": "ohe"},
    DataRole.DATETIME: {"impute": "median", "transform": "date_parts"},
    DataRole.TEXT: {"impute": "constant", "transform": "tfidf"},
    DataRole.IGNORE: {"impute": "drop", "transform": "drop"},
}

def _cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.empty: return 0
    chi2 = stats.chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1)) if n > 1 else 0
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    if min((kcorr-1), (rcorr-1)) <= 0: return 0
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def detect_target(profiles: list[dict], df: pd.DataFrame) -> str:
    best_score = -1.0
    best_col = None
    total_cols = len(df.columns)
    
    for i, p in enumerate(profiles):
        col = p["name"]
        score = 0.0
        
        # Primary Signals
        if p["unique_count"] == 2:
            score += 1.0
        elif 2 < p["unique_count"] <= 5:
            score += 0.5
            
        if 0 < p["entropy"] < 0.8:
            score += 0.5
            
        # Secondary Signals
        if i >= total_cols - 2:
            score += 0.1
            
        name_lower = col.lower()
        if any(kw in name_lower for kw in ['target', 'label', 'churn', 'status', 'attrition']):
            score += 0.1
            
        if score > best_score and score >= 1.0:
            best_score = score
            best_col = col
            
    return best_col

def check_leakage(target_col: str, profiles: list[dict], df: pd.DataFrame):
    if not target_col or target_col not in df.columns:
        return
        
    y = df[target_col].dropna()
    
    for p in profiles:
        col = p["name"]
        if col == target_col or p["inferred_role"] in [DataRole.ID, DataRole.IGNORE]:
            continue
            
        x = df[col].dropna()
        common_idx = x.index.intersection(y.index)
        x_common, y_common = x.loc[common_idx], y.loc[common_idx]
        
        if len(x_common) < 2 or x_common.nunique() <= 1:
            continue
            
        correlation = 0.0
        try:
            if p["inferred_dtype"] in ['float64', 'int64'] and p["unique_count"] > 10:
                y_encoded = pd.factorize(y_common)[0]
                corr, _ = stats.pointbiserialr(y_encoded, x_common)
                correlation = abs(corr) if not np.isnan(corr) else 0.0
            else:
                correlation = _cramers_v(x_common, y_common)
        except Exception:
            pass
            
        if correlation > 0.95:
            p["potential_leakage"] = True
            p["leakage_score"] = float(correlation)

def run_profiling(df: pd.DataFrame) -> Tuple[list[ColumnProfile], str]:
    # Layer 1
    profiles_dict = [profile_column(df[col]) for col in df.columns]
    
    # Target Detection
    suggested_target = detect_target(profiles_dict, df)
    if suggested_target:
        for p in profiles_dict:
            if p["name"] == suggested_target:
                p["inferred_role"] = DataRole.TARGET
                p["confidence_score"] = 1.0
                break
                
    # Leakage Check
    check_leakage(suggested_target, profiles_dict, df)
    
    final_profiles = []
    for p in profiles_dict:
        if p["inferred_role"] != DataRole.TARGET:
            # Layer 2
            p = detect_semantic(df[p["name"]], p)
            # Layer 3
            sample_vals = df[p["name"]].dropna().sample(min(5, df[p["name"]].count())).tolist() if df[p["name"]].count() > 0 else []
            p = refine_with_llm(sample_vals, p)
            
        # Assign Recipes (Phase 1 Goal 2)
        recipe = ROLE_RECIPES.get(p["inferred_role"], ROLE_RECIPES[DataRole.IGNORE])
        p["impute_strategy"] = recipe["impute"]
        p["transform_strategy"] = recipe["transform"]
            
        # Pydantic safety
        p.setdefault("regex_pattern", None)
        p.setdefault("mean_length", None)
        p.setdefault("leakage_score", None)
        p.setdefault("potential_leakage", False)
        
        final_profiles.append(ColumnProfile(**p))
        
    return final_profiles, suggested_target