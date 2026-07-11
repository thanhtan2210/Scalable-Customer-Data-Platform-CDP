from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional, List, Literal

from ...db.session import get_db
from ...db.models import TrainingJob
from ...api.schemas import (
    ModelListResponse,
    ModelVersionInfo,
    PromoteModelRequest,
    PromoteModelResponse,
    ModelCompareResponse,
)
from ...core.serving.model_loader import model_cache

router = APIRouter(prefix="/models", tags=["model-versioning"])


## GET /models/{dataset_id}
@router.get("/{dataset_id}", response_model=ModelListResponse)
def list_models(dataset_id: str, db: Session = Depends(get_db)):
    jobs = (
        db.query(TrainingJob)
        .filter(TrainingJob.dataset_id == dataset_id, TrainingJob.status == "completed")
        .order_by(TrainingJob.started_at.desc())
        .all()
    )

    if not jobs:
        raise HTTPException(
            status_code=404, detail=f"No completed models for dataset {dataset_id}"
        )

    models = [
        ModelVersionInfo(
            job_id=str(j.id),
            model_uri=j.model_uri or "",
            model_class=getattr(j, "model_class", None),
            target_col=j.target_column,
            roc_auc=j.roc_auc,
            optimal_threshold=getattr(j, "optimal_threshold", None),
            created_at=j.started_at,
            status=j.status,
            is_active=j.is_active or False,
            tags=j.tags or {},
        )
        for j in jobs
    ]

    return ModelListResponse(dataset_id=dataset_id, total=len(models), models=models)


## POST /models/{dataset_id}/promote
@router.post("/{dataset_id}/promote", response_model=PromoteModelResponse)
def promote_model(
    dataset_id: str, request: PromoteModelRequest, db: Session = Depends(get_db)
):
    # Tìm job cần promote
    job = (
        db.query(TrainingJob)
        .filter(
            TrainingJob.id == request.job_id,
            TrainingJob.dataset_id == dataset_id,
            TrainingJob.status == "completed",
        )
        .first()
    )

    if not job:
        raise HTTPException(
            status_code=404, detail=f"Job {request.job_id} not found or not completed"
        )

    # Tìm model đang active (nếu có)
    current_active = (
        db.query(TrainingJob)
        .filter(TrainingJob.dataset_id == dataset_id, TrainingJob.is_active == True)
        .first()
    )

    previous_active_id = str(current_active.id) if current_active else None

    # Deactivate tất cả models của dataset
    db.query(TrainingJob).filter(TrainingJob.dataset_id == dataset_id).update(
        {"is_active": False}
    )

    # Activate model được chọn
    job.is_active = True
    db.commit()

    # Invalidate model cache
    model_cache.invalidate(dataset_id=dataset_id)

    return PromoteModelResponse(
        promoted_job_id=str(job.id),
        previous_active=previous_active_id,
        dataset_id=dataset_id,
    )


## GET /models/{dataset_id}/compare
@router.get("/{dataset_id}/compare", response_model=ModelCompareResponse)
def compare_models(
    dataset_id: str, job_id_a: str, job_id_b: str, db: Session = Depends(get_db)
):
    def get_job(job_id):
        job = (
            db.query(TrainingJob)
            .filter(
                TrainingJob.id == job_id,
                TrainingJob.dataset_id == dataset_id,
                TrainingJob.status == "completed",
            )
            .first()
        )
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        return job

    job_a = get_job(job_id_a)
    job_b = get_job(job_id_b)

    auc_a = job_a.roc_auc or 0.0
    auc_b = job_b.roc_auc or 0.0
    delta = round(auc_a - auc_b, 4)

    if abs(delta) < 0.005:
        winner = "tie"
    elif auc_a > auc_b:
        winner = "a"
    else:
        winner = "b"

    def to_info(j):
        return ModelVersionInfo(
            job_id=str(j.id),
            model_uri=j.model_uri or "",
            model_class=getattr(j, "model_class", None),
            target_col=j.target_column,
            roc_auc=j.roc_auc,
            optimal_threshold=getattr(j, "optimal_threshold", None),
            created_at=j.started_at,
            status=j.status,
            is_active=j.is_active or False,
            tags=j.tags or {},
        )

    return ModelCompareResponse(
        model_a=to_info(job_a),
        model_b=to_info(job_b),
        winner=winner,
        delta_roc_auc=delta,
    )
