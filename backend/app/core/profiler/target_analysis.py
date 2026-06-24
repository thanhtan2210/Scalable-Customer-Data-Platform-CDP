from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class TargetRole(str, Enum):
    TARGET = "TARGET"
    AUXILIARY = "AUXILIARY"
    DUPLICATE = "DUPLICATE"
    LEAKAGE = "LEAKAGE"

class GroupRole(str, Enum):
    PRIMARY = "PRIMARY"
    DUPLICATE = "DUPLICATE"
    AUXILIARY = "AUXILIARY"
    LEAKAGE_SUSPECT = "LEAKAGE_SUSPECT"

class TargetSignals(BaseModel):
    is_binary: bool
    entropy: float
    entropy_score: float
    keyword_match: bool
    position_bonus: float

class CandidateTarget(BaseModel):
    name: str
    rank: int
    score: float
    signals: TargetSignals
    suggested_role: TargetRole

class ChurnColumnGroupItem(BaseModel):
    name: str
    correlation_with_target: float
    group_role: GroupRole

class SynthesisStrategy(str, Enum):
    PCA = "PCA"
    WEIGHTED = "WEIGHTED"
    NONE = "NONE"

class ColumnWeight(BaseModel):
    name: str
    weight: float
    normalize_method: str  # "minmax", "zscore", "binary_encode"

class CompositeTargetConfig(BaseModel):
    strategy: SynthesisStrategy
    source_columns: List[str]
    cpi_variance_explained: Optional[float] = None
    weights: Optional[List[ColumnWeight]] = None
    cpi_column_name: str = "cpi_score"
    requires_confirmation: bool = True

class TargetAnalysis(BaseModel):
    recommended_target: str
    candidate_targets: List[CandidateTarget]
    churn_column_group: List[ChurnColumnGroupItem]
    recommended_auxiliary: List[str]
    leakage_suspects: List[str]
    composite_target: Optional[CompositeTargetConfig] = None

