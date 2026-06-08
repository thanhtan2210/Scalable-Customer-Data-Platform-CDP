from pydantic import BaseModel, Field
from typing import Literal, Dict, Any, Optional

class ColumnProfile(BaseModel):
    name: str
    inferred_role: Literal["target", "id", "numeric", "categorical", "text", "datetime", "drop"]
    transform_strategy: Literal["log", "standard", "ohe", "ordinal", "tfidf", "domain_extract", "cyclical", "passthrough", "drop"]
    impute_strategy: Literal["median", "mode", "constant", "none"]
    confidence: float = Field(ge=0.0, le=1.0)
    layer_source: int
    stats: Dict[str, Any]
