from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from ..core.profiler.column_profile import ColumnProfile

# --- DATASET SCHEMAS ---
class DatasetResponse(BaseModel):
    dataset_id: str
    row_count: int
    col_count: int
    status: str

    class Config:
        from_attributes = True

class ProfilingResponse(BaseModel):
    dataset_id: str
    profiles: List[ColumnProfile]
    suggested_target: Optional[str]
    warnings: List[str]

# --- TRAINING SCHEMAS ---
class TrainingRequest(BaseModel):
    confirmed_target: str
    confirmed_profiles: List[ColumnProfile]

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
