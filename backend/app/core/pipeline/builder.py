from typing import List
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from .transforms.registry import TRANSFORM_REGISTRY
from ..profiler.column_profile import ColumnProfile

def build_pipeline(profiles: List[ColumnProfile], target_col: str) -> Pipeline:
    """Builds a dynamic scikit-learn Pipeline based on ColumnProfiles."""
    
    transformers = []
    
    # Group profiles by transform strategy
    strategy_groups = {}
    for p in profiles:
        if p.name == target_col or p.transform_strategy == "drop":
            continue
        
        if p.transform_strategy not in strategy_groups:
            strategy_groups[p.transform_strategy] = []
        strategy_groups[p.transform_strategy].append(p)

    for strategy, group_profiles in strategy_groups.items():
        col_names = [p.name for p in group_profiles]
        
        # 1. Imputation Step
        # For simplicity in this builder, we assume same strategy within a transform group or use most frequent
        impute_strategy = group_profiles[0].impute_strategy
        if impute_strategy == "none":
             impute_step = TRANSFORM_REGISTRY[strategy]
        else:
            impute_step = Pipeline([
                ("imputer", SimpleImputer(strategy=impute_strategy)),
                ("transform", TRANSFORM_REGISTRY[strategy])
            ])
            
        transformers.append((f"{strategy}_branch", impute_step, col_names))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    
    return Pipeline([
        ("preprocessor", preprocessor)
    ])
