from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
import pandas as pd
import io
import uuid
from sqlalchemy.orm import Session
from ...db.models import Dataset, Profile, TrainingJob
from ...db.session import get_db # Assuming session manager exists
from ...api.schemas import DatasetResponse, ProfilingResponse, TrainingRequest, JobResponse, ConfirmCompositeRequest, ConfirmCompositeResponse
from ...core.storage import storage
from ...core.profiler.orchestrator import run_profiling
from ...core.profiler.target_analysis import TargetAnalysis
from ...core.training.automl import run_automl

router = APIRouter(prefix="/datasets", tags=["datasets"])

@router.post("/upload", response_model=DatasetResponse)
async def upload_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # 1. Validation
    if not (file.filename.endswith('.csv') or file.filename.endswith('.parquet')):
        raise HTTPException(status_code=400, detail="Only CSV or Parquet files allowed")
    
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (>50MB)")
    
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content), nrows=100)
        else:
            df = pd.read_parquet(io.BytesIO(content))
    except Exception:
         raise HTTPException(status_code=400, detail="Could not parse file")

    if len(df.columns) < 2:
        raise HTTPException(status_code=400, detail="Dataset must have at least 2 columns")

    # 2. Store to R2
    dataset_id = str(uuid.uuid4())
    user_id = "default_user" # Placeholder for auth
    r2_path = f"raw/{user_id}/{dataset_id}/{file.filename}"
    storage.upload_file(content, r2_path)

    # 3. DB Metadata
    new_dataset = Dataset(
        id=dataset_id,
        user_id=user_id,
        filename=file.filename,
        r2_path=r2_path,
        row_count=0, # Will be updated after full read if needed
        col_count=len(df.columns),
        status="uploaded"
    )
    db.add(new_dataset)
    db.commit()
    
    return {
        "dataset_id": dataset_id,
        "row_count": 0,
        "col_count": len(df.columns),
        "status": "uploaded"
    }

@router.post("/{dataset_id}/profile", response_model=ProfilingResponse)
async def profile_dataset(dataset_id: str, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # 1. Read from R2
    content = storage.download_file(dataset.r2_path)
    df = pd.read_csv(io.BytesIO(content)) if dataset.filename.endswith('.csv') else pd.read_parquet(io.BytesIO(content))
    
    # 2. Run Profiling
    profiles, suggested_target = run_profiling(df)
    
    # 4. Save to DB
    new_profile = Profile(
        dataset_id=dataset_id,
        profiles_json=[p.dict() for p in profiles],
        suggested_target=suggested_target.json() if hasattr(suggested_target, 'json') else str(suggested_target)
    )
    db.merge(new_profile)
    dataset.status = "profiled"
    db.commit()
    
    return {
        "dataset_id": dataset_id,
        "profiles": profiles,
        "suggested_target": suggested_target.recommended_target if hasattr(suggested_target, 'recommended_target') else str(suggested_target),
        "warnings": [p.name for p in profiles if p.inferred_role == "drop"],
        "composite_target": suggested_target.composite_target if hasattr(suggested_target, 'composite_target') else None
    }

@router.post("/{dataset_id}/confirm-composite", response_model=ConfirmCompositeResponse)
async def confirm_composite(
    dataset_id: str,
    req: ConfirmCompositeRequest,
    db: Session = Depends(get_db)
):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    profile = db.query(Profile).filter(Profile.dataset_id == dataset_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    # If confirmed is False, clear composite target in DB and return cpi_attached=False
    if not req.confirmed:
        try:
            target_analysis = TargetAnalysis.parse_raw(profile.suggested_target)
            target_analysis.composite_target = None
            profile.suggested_target = target_analysis.json()
        except Exception:
            profile.suggested_target = None
            
        db.commit()
        return {
            "dataset_id": dataset_id,
            "composite_target": None,
            "cpi_attached": False
        }

    # If confirmed is True:
    # 1. Download file from R2
    content = storage.download_file(dataset.r2_path)
    df = pd.read_csv(io.BytesIO(content)) if dataset.r2_path.endswith('.csv') or dataset.filename.endswith('.csv') else pd.read_parquet(io.BytesIO(content))

    # 2. Get target analysis
    try:
        target_analysis = TargetAnalysis.parse_raw(profile.suggested_target)
    except Exception:
        _, target_analysis = run_profiling(df)

    # 3. Determine strategy & source columns
    strategy = req.selected_strategy
    if not strategy:
        if target_analysis.composite_target:
            strategy = target_analysis.composite_target.strategy
        else:
            strategy = "WEIGHTED"
            
    source_columns = req.selected_source_columns
    if not source_columns:
        if target_analysis.composite_target:
            source_columns = target_analysis.composite_target.source_columns
        else:
            source_columns = [
                c.name for c in target_analysis.churn_column_group
                if c.group_role in ("AUXILIARY", "DUPLICATE") and c.name != target_analysis.recommended_target
            ]

    if strategy == "NONE" or not source_columns:
        target_analysis.composite_target = None
        profile.suggested_target = target_analysis.json()
        db.commit()
        return {
            "dataset_id": dataset_id,
            "composite_target": None,
            "cpi_attached": False
        }

    # 4. Perform target synthesis
    from ...core.profiler.target_synthesizer import _pca_synthesis, _weighted_synthesis, _auto_assign_weights
    from ...core.profiler.target_analysis import CompositeTargetConfig
    
    cpi_variance_explained = None
    weights = None
    cpi_series = None

    if strategy == "PCA":
        numeric_cols = [c for c in source_columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) >= 2:
            cpi_variance_explained, cpi_series = _pca_synthesis(df, numeric_cols, target_analysis.recommended_target)
        else:
            strategy = "WEIGHTED" # fallback

    if strategy == "WEIGHTED":
        eligible_items = [c for c in target_analysis.churn_column_group if c.name in source_columns]
        weights = _auto_assign_weights(eligible_items, df)
        cpi_series = _weighted_synthesis(df, weights)

    # 5. Build CompositeTargetConfig
    composite_config = CompositeTargetConfig(
        strategy=strategy,
        source_columns=source_columns,
        cpi_variance_explained=cpi_variance_explained,
        weights=weights,
        cpi_column_name="cpi_score",
        requires_confirmation=False
    )

    # 6. Attach cpi_score and re-upload to processed/... path as parquet
    df["cpi_score"] = cpi_series
    
    pq_buffer = io.BytesIO()
    df.to_parquet(pq_buffer, index=False)
    pq_content = pq_buffer.getvalue()
    
    new_r2_path = f"processed/{dataset_id}/with_cpi.parquet"
    storage.upload_file(pq_content, new_r2_path)

    # 7. Update Dataset & Profile records
    dataset.r2_path = new_r2_path
    dataset.filename = "with_cpi.parquet" # ensures jobs.py reads it as parquet

    target_analysis.composite_target = composite_config
    profile.suggested_target = target_analysis.json()

    # Append cpi_score to ColumnProfile list
    from ...core.profiler.column_profile import ColumnProfile, DataRole
    cpi_profile = ColumnProfile(
        name="cpi_score",
        inferred_dtype="float64",
        inferred_role=DataRole.NUMERIC,
        confidence_score=1.0,
        null_pct=0.0,
        unique_count=df["cpi_score"].nunique(),
        entropy=0.0,
        potential_leakage=False,
        transform_strategy="standard",
        impute_strategy="median"
    )
    
    existing_profiles = profile.profiles_json or []
    existing_profiles = [p for p in existing_profiles if p.get("name") != "cpi_score"]
    existing_profiles.append(cpi_profile.dict())
    profile.profiles_json = existing_profiles

    db.commit()

    return {
        "dataset_id": dataset_id,
        "composite_target": composite_config,
        "cpi_attached": True
    }
