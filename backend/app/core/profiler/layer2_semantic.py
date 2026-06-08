import re
import pandas as pd
from typing import Dict, Any

# Common Regex Patterns
PATTERNS = {
    "email": r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$',
    "url": r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',
    "ip": r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
    "datetime": r'^\d{4}-\d{2}-\d{2}|^\d{2}/\d{2}/\d{4}'
}

def detect_semantic(series: pd.Series, l1_result: Dict[str, Any]) -> Dict[str, Any]:
    """Layer 2: Semantic pattern matching to override or confirm Layer 1."""
    res = l1_result.copy()
    stats = l1_result["stats"]
    
    # Only process objects/strings or potential IDs
    if stats["inferred_type"] == "categorical" or res["inferred_role"] == "id":
        sample = series.dropna().astype(str).head(100)
        
        # 1. Regex Matching
        matches = {k: sample.str.match(v).mean() for k, v in PATTERNS.items()}
        top_match = max(matches, key=matches.get)
        
        if matches[top_match] > 0.8:
            res["layer_source"] = 2
            res["confidence"] = 0.9
            if top_match == "email":
                res["inferred_role"] = "drop" # PII should be dropped by default
            elif top_match == "datetime":
                res["inferred_role"] = "datetime"
                res["transform_strategy"] = "domain_extract"
            return res

        # 2. Text vs Categorical Heuristic
        # If average length is high and cardinality is high -> Text
        if stats["inferred_type"] == "categorical":
            avg_len = series.dropna().astype(str).apply(len).mean()
            if avg_len > 50 and stats["cardinality_ratio"] > 0.3:
                res["inferred_role"] = "text"
                res["transform_strategy"] = "tfidf"
                res["confidence"] = 0.8
                res["layer_source"] = 2

    # 3. Numeric Patterns (Zipcode/Phone)
    if stats["inferred_type"] == "numeric":
        sample_str = series.dropna().astype(str)
        # Zipcode check (all 5 digits)
        if sample_str.str.match(r'^\d{5}$').mean() > 0.9:
            res["inferred_role"] = "categorical"
            res["transform_strategy"] = "ohe"
            res["confidence"] = 0.8
            res["layer_source"] = 2

    return res
