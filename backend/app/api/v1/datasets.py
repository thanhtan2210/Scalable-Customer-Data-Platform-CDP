from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
import pandas as pd
import io
import uuid
from sqlalchemy.orm import Session
from ...db.models import Dataset, Profile, TrainingJob
from ...db.session import get_db # Assuming session manager exists
from ...api.schemas import DatasetResponse, ProfilingResponse, TrainingRequest, JobResponse
from ...core.storage import storage
from ...core.profiler.orchestrator import run_profiling
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
        suggested_target=suggested_target
    )
    db.merge(new_profile)
    dataset.status = "profiled"
    db.commit()
    
    return {
        "dataset_id": dataset_id,
        "profiles": profiles,
        "suggested_target": suggested_target,
        "warnings": [p.name for p in profiles if p.inferred_role == "drop"]
    }
