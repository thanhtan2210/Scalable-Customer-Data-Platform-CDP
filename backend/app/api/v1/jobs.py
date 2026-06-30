from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from datetime import datetime
import uuid
import io
import pandas as pd
from ...db.models import Dataset, TrainingJob, Profile
from ...db.session import get_db
from ...api.schemas import TrainingRequest, JobResponse, JobStatusResponse
from ...core.storage import storage
from ...core.training.automl import run_automl
from ...core.profiler.column_profile import ColumnProfile
from ...core.profiler.target_analysis import CompositeTargetConfig
from typing import Optional, List

router = APIRouter(prefix="/jobs", tags=["jobs"])

async def training_task(
    job_id: str,
    dataset_id: str,
    target: str,
    profiles_dict: list,
    db_session: Session,
    composite_config: Optional[CompositeTargetConfig] = None,  # Bug 1 fix
):
    try:
        # 1. Load Data
        dataset = db_session.query(Dataset).filter(Dataset.id == dataset_id).first()
        content = storage.download_file(dataset.r2_path)
        df = pd.read_csv(io.BytesIO(content)) if dataset.filename.endswith('.csv') else pd.read_parquet(io.BytesIO(content))

        # 2. Reconstruct ColumnProfile objects from dict
        confirmed_profiles = [ColumnProfile(**p) for p in profiles_dict]

        # 3. Run AutoML — now with composite_config to enable MTL path
        model_uri, _schema_path = run_automl(
            df,
            confirmed_profiles,
            target,
            dataset_id=dataset_id,
            composite_config=composite_config,
        )

        # Extract best_roc_auc from MLflow
        import re
        import mlflow
        best_roc_auc = None
        match = re.search(r"runs:/+([^/]+)", model_uri)
        if match:
            run_id = match.group(1)
            try:
                client = mlflow.tracking.MlflowClient()
                run_data = client.get_run(run_id)
                best_roc_auc = run_data.data.metrics.get("best_roc_auc")
            except Exception as ex:
                print(f"Failed to fetch metric from MLflow: {ex}")

        # 4. Update Job
        job = db_session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        job.status = "completed"
        job.model_uri = model_uri
        if best_roc_auc is not None:
            job.roc_auc = float(best_roc_auc)
        job.finished_at = datetime.utcnow()

        dataset.status = "completed"
        db_session.commit()
    except Exception as e:
        job = db_session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if job:
            job.status = "failed"
            db_session.commit()
        print(f"Training Failed for {job_id}: {str(e)}")

@router.post("/datasets/{dataset_id}/train", response_model=JobResponse)
async def start_training(
    dataset_id: str, 
    req: TrainingRequest, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Idempotency check
    existing_job = db.query(TrainingJob).filter(
        TrainingJob.dataset_id == dataset_id,
        TrainingJob.target_column == req.confirmed_target,
        TrainingJob.status == "completed"
    ).first()
    
    if existing_job:
        return {
            "job_id": existing_job.id,
            "status": "completed",
            "estimated_minutes": 0
        }

    # Create Job
    job_id = str(uuid.uuid4())
    new_job = TrainingJob(
        id=job_id,
        dataset_id=dataset_id,
        status="training",
        target_column=req.confirmed_target
    )
    db.add(new_job)
    dataset.status = "training"
    db.commit()

    # Trigger Background Task
    background_tasks.add_task(
        training_task,
        job_id,
        dataset_id,
        req.confirmed_target,
        [p.dict() for p in req.confirmed_profiles],
        db,
        req.composite_config,  # Bug 1 fix: pass CPI config through
    )

    return {
        "job_id": job_id,
        "status": "training",
        "estimated_minutes": 5 # Static estimate for MVP
    }

@router.get("/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job.id,
        "status": job.status,
        "roc_auc": job.roc_auc,
        "model_uri": job.model_uri,
        "finished_at": job.finished_at
    }
