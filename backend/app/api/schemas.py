from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from ..core.profiler.column_profile import ColumnProfile
from ..core.profiler.target_analysis import CompositeTargetConfig, SynthesisStrategy

# --- DATASET SCHEMAS ---
class DatasetResponse(BaseModel):
    dataset_id: str
    row_count: Optional[int] = None
    col_count: Optional[int] = None
    status: str
    detected_format: str
    sheets: Optional[List[str]] = None
    requires_sheet_selection: bool = False

    class Config:
        from_attributes = True

class SelectSheetRequest(BaseModel):
    sheet_name: str

class ProfilingResponse(BaseModel):
    dataset_id: str
    profiles: List[ColumnProfile]
    suggested_target: Optional[str]
    warnings: List[str]
    composite_target: Optional[CompositeTargetConfig] = None  # Bug 2 fix: expose CPI config to client

# --- TRAINING SCHEMAS ---
class TrainingRequest(BaseModel):
    confirmed_target: str
    confirmed_profiles: List[ColumnProfile]
    composite_config: Optional[CompositeTargetConfig] = None  # Bug 1 fix: flow CPI config into run_automl()
    prior_model_uri: Optional[str] = None

class JobResponse(BaseModel):
    job_id: str
    status: str
    estimated_minutes: int

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    roc_auc: Optional[float]
    model_uri: Optional[str]
    finished_at: Optional[datetime]

# --- PREDICTION SCHEMAS ---
class PredictionRequest(BaseModel):
    dataset_id: str
    records: List[Dict[str, Any]]

class PredictionResult(BaseModel):
    record_id: Optional[str]
    churn_probability: float
    risk_level: str

class PredictionResponse(BaseModel):
    predictions: List[PredictionResult]

class ConfirmCompositeRequest(BaseModel):
    confirmed: bool
    selected_strategy: Optional[SynthesisStrategy] = None
    selected_source_columns: Optional[List[str]] = None

class ConfirmCompositeResponse(BaseModel):
    dataset_id: str
    composite_target: Optional[CompositeTargetConfig]
    cpi_attached: bool
