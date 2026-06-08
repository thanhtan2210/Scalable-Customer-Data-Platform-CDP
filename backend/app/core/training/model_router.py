from typing import List
from ..profiler.column_profile import ColumnProfile

def select_model(profiles: List[ColumnProfile], n_rows: int) -> str:
    """Intelligently selects a model class based on dataset characteristics."""
    
    has_text = any(p.transform_strategy == "tfidf" for p in profiles)
    n_numeric = sum(1 for p in profiles if p.inferred_role == "numeric")
    n_categorical = sum(1 for p in profiles if p.inferred_role == "categorical")
    
    if n_rows < 1000:
        return "LogisticRegression"
    
    if has_text:
        return "LogisticRegression" # Faster and better baseline for high-dim sparse text
    
    if n_numeric > n_categorical * 2:
        return "XGBClassifier"
    
    return "RandomForestClassifier"
