from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from ..core.profiler.column_profile import ColumnProfile
from ..core.profiler.target_analysis import (
    CompositeTargetConfig,
    SynthesisStrategy,
    CandidateTarget,
    ChurnColumnGroupItem,
)


# --- DATASET SCHEMAS ---
class DatasetResponse(BaseModel):
    dataset_id: str
    row_count: Optional[int] = None
    col_count: Optional[int] = None
    status: str
    detected_format: str
    sheets: Optional[List[str]] = None
    requires_sheet_selection: bool = False
    r2_path: Optional[str] = None

    class Config:
        from_attributes = True


class SelectSheetRequest(BaseModel):
    sheet_name: str


class ProfilingResponse(BaseModel):
    dataset_id: str
    profiles: List[ColumnProfile]
    suggested_target: Optional[str]
    candidate_targets: List[CandidateTarget]
    composite_target: Optional[CompositeTargetConfig] = None
    churn_column_group: List[ChurnColumnGroupItem]
    leakage_suspects: List[str]
    warnings: List[str]


# --- TRAINING SCHEMAS ---
class TrainingRequest(BaseModel):
    confirmed_target: str
    confirmed_profiles: List[ColumnProfile]
    composite_config: Optional[CompositeTargetConfig] = (
        None  # Bug 1 fix: flow CPI config into run_automl()
    )
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
    optimal_threshold: Optional[float] = None
    finished_at: Optional[datetime]
    error_message: Optional[str] = None


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


class ReEvaluateLeakageRequest(BaseModel):
    confirmed_target: str


class ReEvaluateLeakageResponse(BaseModel):
    profiles_updated_in_db: bool
    updated_profiles: List[ColumnProfile]
    dataset_id: str
    confirmed_target: str
    leakage_suspects: List[str] = []


class BatchPredictionRequest(BaseModel):
    dataset_id: str
    file_path: str


class BatchPredictionResult(BaseModel):
    record_id: str
    probability: float
    risk_level: str


class BatchPredictionResponse(BaseModel):
    total_records: int
    high_risk: int
    medium_risk: int
    low_risk: int
    predictions: List[BatchPredictionResult]
    threshold_used: float
    threshold_source: str
    model_type: Optional[str] = None


class DriftRequest(BaseModel):
    target_file_path: Optional[str] = None
    date: Optional[str] = None


class FeatureDriftResult(BaseModel):
    type: str
    ks_statistic: Optional[float] = None
    ks_p_value: Optional[float] = None
    chi2_p_value: Optional[float] = None
    psi: float
    drift_level: str
    is_drifted: bool


class DriftResponse(BaseModel):
    dataset_id: str
    reference_rows: int
    target_rows: int
    drift_detected: bool
    metrics: Dict[str, FeatureDriftResult]


class ModelVersionInfo(BaseModel):
    job_id: str
    model_uri: str
    model_class: Optional[str] = None
    target_col: str
    roc_auc: Optional[float] = None
    optimal_threshold: Optional[float] = None
    created_at: datetime
    status: str
    is_active: bool
    tags: dict = {}


class ModelListResponse(BaseModel):
    dataset_id: str
    total: int
    models: List[ModelVersionInfo]


class PromoteModelRequest(BaseModel):
    job_id: str


class PromoteModelResponse(BaseModel):
    promoted_job_id: str
    previous_active: Optional[str] = None
    dataset_id: str


class ModelCompareResponse(BaseModel):
    model_a: ModelVersionInfo
    model_b: ModelVersionInfo
    winner: Literal["a", "b", "tie"]
    delta_roc_auc: float


# --- JWT AUTH SCHEMAS ---
class UserInfo(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    is_admin: bool
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserInfo


class UserRegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(
        min_length=8,
        description="Minimum 8 characters"
    )
    full_name: Optional[str] = None


class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8)
