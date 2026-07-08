from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from sqlalchemy.orm import Session
from datetime import datetime
import uuid
import logging

logger = logging.getLogger("cdp.jobs")
import io
import pandas as pd
from ...db.models import Dataset, TrainingJob, Profile
from ...db.session import get_db, SessionLocal
from ...api.schemas import TrainingRequest, JobResponse, JobStatusResponse
from ...core.storage import storage
from ...core.training.automl import run_automl
from ...core.profiler.column_profile import ColumnProfile
from ...core.profiler.target_analysis import CompositeTargetConfig
from ...core.ingestion.parsers import parse_file
from typing import Optional, List
from ...core.limiter import limiter

router = APIRouter(prefix="/jobs", tags=["jobs"])

async def training_task(
    job_id: str,
    dataset_id: str,
    target: str,
    profiles_dict: list,
    composite_config: Optional[CompositeTargetConfig] = None,
    prior_model_uri: Optional[str] = None,
):
    db_session = SessionLocal()
    try:
        # 1. Update status to training
        job = db_session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if job:
            job.status = "training"
        dataset = db_session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if dataset:
            dataset.status = "training"
        db_session.commit()

        # 2. Load Data
        content = storage.download_file(dataset.r2_path)
        result = parse_file(content=content, filename=dataset.filename)
        df = result.df

        # 3. Reconstruct ColumnProfile objects from dict
        confirmed_profiles = [ColumnProfile(**p) for p in profiles_dict]

        # 4. Run AutoML
        model_uri, _schema_path = run_automl(
            df,
            confirmed_profiles,
            target,
            dataset_id=dataset_id,
            composite_config=composite_config,
            prior_model_uri=prior_model_uri,
        )

        # 5. Generate and Save Schema (FIX)
        try:
            from ...core.pipeline.schema_gen import generate_schema, save_schema
            schema, metadata = generate_schema(confirmed_profiles, dataset_id, target)
            save_schema(schema, metadata, dataset_id, target)
        except Exception as schema_ex:
            job = db_session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if job:
                job.status = "failed"
                job.error_message = f"Schema save failed: {str(schema_ex)}"
            dataset = db_session.query(Dataset).filter(Dataset.id == dataset_id).first()
            if dataset:
                dataset.status = "failed"
            db_session.commit()
            raise schema_ex

        # Extract metrics from MLflow
        import re
        import mlflow
        best_roc_auc = None
        optimal_threshold = None
        match = re.search(r"runs:/+([^/]+)", model_uri)
        if match:
            run_id = match.group(1)
            try:
                client = mlflow.tracking.MlflowClient()
                run_data = client.get_run(run_id)
                best_roc_auc = run_data.data.metrics.get("best_roc_auc")
                optimal_threshold = run_data.data.metrics.get("optimal_threshold")
            except Exception as ex:
                logger.error(f"Failed to fetch metric from MLflow: {ex}")

        # 6. Update Job
        job = db_session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        dataset = db_session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if job:
            job.status = "completed"
            job.model_uri = model_uri
            if best_roc_auc is not None:
                job.roc_auc = float(best_roc_auc)
            if optimal_threshold is not None:
                job.optimal_threshold = float(optimal_threshold)
            job.finished_at = datetime.utcnow()

        if dataset:
            dataset.status = "completed"
        db_session.commit()

        # Sau khi update job status:
        try:
            from app.core.serving.model_loader import model_cache
        except ImportError:
            from backend.app.core.serving.model_loader import model_cache
            
        invalidated = model_cache.invalidate(dataset_id=dataset_id)
        logger.info(f"Cache invalidated {invalidated} entries for dataset {dataset_id}")
    except Exception as e:
        try:
            job = db_session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if job and job.status != "failed":
                job.status = "failed"
                job.error_message = str(e)
            dataset = db_session.query(Dataset).filter(Dataset.id == dataset_id).first()
            if dataset and dataset.status != "failed":
                dataset.status = "failed"
            db_session.commit()
        except Exception as commit_ex:
            logger.error(f"Failed to save failure status to DB: {commit_ex}")
        logger.error(f"Training Failed for {job_id}: {str(e)}")
    finally:
        db_session.close()


@router.post("/datasets/{dataset_id}/train", response_model=JobResponse)
@limiter.limit("2/minute")
async def start_training(
    request: Request,
    dataset_id: str, 
    req: TrainingRequest, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    # Validate confirmed_target exists in dataset
    try:
        content = storage.download_file(dataset.r2_path)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"File not found in storage: {str(e)}")
        
    try:
        result = parse_file(content=content, filename=dataset.filename)
        df = result.df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse dataset file: {str(e)}")
        
    if df is None:
        raise HTTPException(status_code=400, detail="Failed to parse dataset dataframe")
        
    if req.confirmed_target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Confirmed target '{req.confirmed_target}' not found in dataset columns")
    
    # Idempotency check
    existing_job = db.query(TrainingJob).filter(
        TrainingJob.dataset_id == dataset_id,
        TrainingJob.target_column == req.confirmed_target,
        TrainingJob.status == "completed",
        TrainingJob.roc_auc > 0.65,
        TrainingJob.prior_model_uri == req.prior_model_uri
    ).first()
    
    if existing_job:
        return {
            "job_id": existing_job.id,
            "status": "completed",
            "estimated_minutes": 0
        }

    # Create Job with status="queued"
    job_id = str(uuid.uuid4())
    new_job = TrainingJob(
        id=job_id,
        dataset_id=dataset_id,
        status="queued",
        target_column=req.confirmed_target,
        prior_model_uri=req.prior_model_uri
    )
    db.add(new_job)
    dataset.status = "queued"
    db.commit()

    # Trigger Background Task
    background_tasks.add_task(
        training_task,
        job_id,
        dataset_id,
        req.confirmed_target,
        [p.dict() for p in req.confirmed_profiles],
        req.composite_config,
        req.prior_model_uri,
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "estimated_minutes": 5
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
        "optimal_threshold": job.optimal_threshold,
        "finished_at": job.finished_at
    }
