from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from typing import List, Tuple, Optional

from ..profiler.column_profile import ColumnProfile, DataRole
from .transforms.registry import get_imputer, get_transformer

from ..exceptions import PipelineBuilderError

def build_pipeline(
    confirmed_profiles: List[ColumnProfile],
    target_col: str,
    auxiliary_cols: Optional[List[str]] = None
) -> Pipeline:
    """Builds a scikit-learn Pipeline based on user-confirmed column profiles."""
    if not confirmed_profiles:
        raise ValueError("Cannot build pipeline: Input profiles list is empty.")
    if not target_col:
        raise ValueError("Cannot build pipeline: target_col is missing.")
        
    features = []
    
    # Filter columns
    for p in confirmed_profiles:
        if p.inferred_role in [DataRole.ID, DataRole.IGNORE, DataRole.TARGET]:
            continue
        if p.name == target_col:
            continue
        # Also exclude auxiliary columns from standard pipeline preprocessing if they are handled separately
        if auxiliary_cols and p.name in auxiliary_cols:
            continue
        features.append(p)
        
    if not features and not auxiliary_cols:
        raise PipelineBuilderError("No valid feature columns remaining after filtering. Check leakage flags and column roles before training.")
        
    # Group by (impute_strategy, transform_strategy)
    groups = {}
    for p in features:
        # tfidf needs to be per-column because TfidfVectorizer expects 1D input
        key = (p.impute_strategy, p.transform_strategy, p.name if p.transform_strategy == "tfidf" else None)
        if key not in groups:
            groups[key] = []
        groups[key].append(p.name)
        
    transformers = []
    for (impute_strat, transform_strat, _), cols in groups.items():
        if impute_strat == "drop" and transform_strat == "drop":
            continue
            
        steps = []
        imputer = get_imputer(impute_strat)
        if imputer != "drop":
            steps.append(("imputer", imputer))
            
        transformer = get_transformer(transform_strat)
        if transformer != "drop":
            steps.append(("transformer", transformer))
            
        if steps:
            # Create a sub-pipeline for this group
            pipe = Pipeline(steps)
            step_name = f"pipe_{transform_strat}_{cols[0]}" if transform_strat == "tfidf" else f"pipe_{impute_strat}_{transform_strat}"
            transformers.append((step_name, pipe, cols))
            
    if auxiliary_cols:
        transformers.append(("auxiliary_passthrough", "passthrough", auxiliary_cols))
        
    if not transformers:
        raise ValueError("Cannot build pipeline: No valid transformers resolved.")
        
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    
    # Final pipeline with a placeholder classifier
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', DummyClassifier(strategy="prior"))
    ])

