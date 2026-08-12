from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional, List, Literal
import json

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


## GET /models/{dataset_id}/feature-importance
@router.get("/{dataset_id}/feature-importance")
def get_feature_importance(
    dataset_id: str,
    job_id: Optional[str] = None,
    method: Optional[str] = None,
    db: Session = Depends(get_db),
):
    # 1. Get job
    if job_id:
        job = (
            db.query(TrainingJob)
            .filter(
                TrainingJob.id == job_id,
                TrainingJob.dataset_id == dataset_id,
            )
            .first()
        )
    else:
        job = (
            db.query(TrainingJob)
            .filter(
                TrainingJob.dataset_id == dataset_id,
                TrainingJob.is_active == True,
            )
            .first()
            or db.query(TrainingJob)
            .filter(
                TrainingJob.dataset_id == dataset_id,
                TrainingJob.status == "completed",
            )
            .order_by(TrainingJob.roc_auc.desc())
            .first()
        )

    if not job:
        raise HTTPException(
            status_code=404, detail="No completed model found"
        )

    # 2. Load model + feature names
    model, model_type = model_cache.get_or_load(
        str(job.id), job.model_uri
    )

    feature_names = None
    try:
        import mlflow
        client = mlflow.tracking.MlflowClient()
        uri_parts = (job.model_uri or "").split("/")
        if len(uri_parts) >= 2:
            run_id = uri_parts[1]
            local_path = client.download_artifacts(run_id, "feature_names.json")
            with open(local_path, "r") as f:
                feature_names = json.load(f).get("feature_names")
    except Exception:
        pass

    if not feature_names and hasattr(model, "steps") and len(model.steps) > 1:
        try:
            feature_names = list(model[:-1].get_feature_names_out())
        except Exception:
            feature_names = None

    if hasattr(model, "steps"):
        estimator = model.steps[-1][1]
    else:
        estimator = model

    # 3. Select method
    use_shap = (
        method == "shap" or
        (method is None and model_type == "mtl_sklearn")
    )

    importance_data = []
    method_used = ""

    if use_shap:
        try:
            import shap
            import numpy as np
            from ...core.storage import storage

            sample_df = None
            try:
                sample_path = f"ml_artifacts/{dataset_id}/inference/"
                files = storage.list_files(sample_path)
                if files:
                    latest = sorted(files)[-1]
                    sample_df = storage.download_dataframe(latest)
                    if sample_df is not None and not sample_df.empty:
                        sample_df = sample_df.head(100)
            except Exception:
                sample_df = None

            if sample_df is None or sample_df.empty:
                raise ValueError("No inference data for SHAP")

            X_sample = model[:-1].transform(sample_df)

            if model_type == "mtl_sklearn":
                explainer = shap.KernelExplainer(
                    lambda x: estimator.predict_proba(x)[:, 1],
                    shap.sample(X_sample, min(50, len(X_sample))),
                )
                shap_vals = explainer.shap_values(X_sample[:50])
            else:
                explainer = shap.TreeExplainer(estimator)
                shap_vals = explainer.shap_values(X_sample)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]

            mean_abs = np.abs(shap_vals).mean(axis=0)

            if feature_names and len(feature_names) == len(mean_abs):
                importance_data = sorted(
                    [
                        {
                            "feature": str(name),
                            "importance": float(val),
                            "method": "shap",
                        }
                        for name, val in zip(feature_names, mean_abs)
                    ],
                    key=lambda x: x["importance"],
                    reverse=True,
                )[:20]
            else:
                importance_data = sorted(
                    [
                        {
                            "feature": f"feature_{i}",
                            "importance": float(val),
                            "method": "shap",
                        }
                        for i, val in enumerate(mean_abs)
                    ],
                    key=lambda x: x["importance"],
                    reverse=True,
                )[:20]
            method_used = "shap"
        except Exception as e:
            import logging
            logging.getLogger("cdp.models").warning(
                f"SHAP failed: {e}. Falling back to sklearn."
            )
            use_shap = False

    if not use_shap or not importance_data:
        if hasattr(estimator, "feature_importances_"):
            names = feature_names or [
                f"feature_{i}"
                for i in range(len(estimator.feature_importances_))
            ]
            importance_data = sorted(
                [
                    {
                        "feature": str(name),
                        "importance": float(imp),
                        "method": "feature_importances",
                    }
                    for name, imp in zip(names, estimator.feature_importances_)
                ],
                key=lambda x: x["importance"],
                reverse=True,
            )[:20]
            method_used = "feature_importances"
        else:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Model type '{model_type}' does not support feature "
                    f"importance. No inference data available for SHAP."
                ),
            )

    return {
        "dataset_id": dataset_id,
        "job_id": str(job.id),
        "model_type": model_type,
        "method": method_used,
        "feature_importance": importance_data,
        "top_n": len(importance_data),
        "note": (
            "SHAP values computed on sample inference data"
            if method_used == "shap"
            else "sklearn built-in importance"
        ),
    }
