from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

class DataRole(str, Enum):
    ID = "ID"
    TARGET = "TARGET"
    NUMERIC = "NUMERIC"
    CATEGORICAL = "CATEGORICAL"
    DATETIME = "DATETIME"
    TEXT = "TEXT"
    IGNORE = "IGNORE"

class ColumnProfile(BaseModel):
    name: str
    inferred_dtype: str
    inferred_role: DataRole
    confidence_score: float = Field(ge=0.0, le=1.0)
    null_pct: float = Field(ge=0.0, le=1.0)
    unique_count: int
    entropy: float
    mean_length: Optional[float] = None
    regex_pattern: Optional[str] = None
    potential_leakage: bool = False
    leakage_score: Optional[float] = None
    
    # New fields for Phase 2 integration
    transform_strategy: Optional[str] = None
    impute_strategy: Optional[str] = None