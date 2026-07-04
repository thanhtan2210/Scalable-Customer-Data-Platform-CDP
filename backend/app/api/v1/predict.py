from fastapi import APIRouter, Depends, HTTPException
from typing import List
import pandas as pd
import io
import logging
from sqlalchemy.orm import Session

logger = logging.getLogger("cdp.predict")
from ...db.models import Dataset, TrainingJob, Profile
from ...db.session import get_db
from ...api.schemas import PredictionRequest, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse, DriftRequest, DriftResponse
from ...core.storage import storage
from ...core.serving.model_loader import model_cache

router = APIRouter(prefix="/predict", tags=["prediction"])

@router.post("", response_model=PredictionResponse)
async def predict(req: PredictionRequest, db: Session = Depends(get_db)):
    # 1. Get Best Model for Dataset
    job = db.query(TrainingJob).filter(
        TrainingJob.dataset_id == req.dataset_id, 
        TrainingJob.status == "completed"
    ).order_by(TrainingJob.roc_auc.desc()).first()
    
    if not job or not job.model_uri:
        raise HTTPException(status_code=404, detail="No completed training job found for this dataset")

    # 2. Load Model (cached)
    model = model_cache.get_model(job.model_uri)
    
    # 3. Prepare Input
    input_df = pd.DataFrame(req.records)
    
    # 4. Get optimal threshold
    optimal_threshold, _ = get_optimal_threshold(job)
    
    try:
        # 5. Predict
        probabilities = model.predict_proba(input_df)[:, 1]
        
        results = []
        for i, prob in enumerate(probabilities):
            risk = classify_risk(float(prob), optimal_threshold)
            results.append({
                "record_id": req.records[i].get("id") or str(i),
                "churn_probability": float(prob),
                "risk_level": risk
            })
            
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

def get_optimal_threshold(job: TrainingJob) -> tuple[float, str]:
    import re
    import json
    import mlflow
    import os
    from urllib.parse import urlparse
    optimal_threshold = None
    threshold_source = "optimal"
    
    match = re.search(r"runs:/+([^/]+)", job.model_uri)
    if match:
        run_id = match.group(1)
        try:
            client = mlflow.tracking.MlflowClient()
            run_data = client.get_run(run_id)
            optimal_threshold = run_data.data.metrics.get("optimal_threshold")
        except Exception:
            pass

    if optimal_threshold is None:
        if job.optimal_threshold is not None:
            optimal_threshold = job.optimal_threshold

    if optimal_threshold is None and match:
        try:
            client = mlflow.tracking.MlflowClient()
            local_path = client.download_artifacts(run_id, "threshold.json")
            with open(local_path, "r") as f:
                threshold_data = json.load(f)
                optimal_threshold = float(threshold_data.get("optimal_threshold"))
        except Exception as ex:
            logger.error(f"Failed to load optimal_threshold from MLflow artifact: {ex}")

    # Fallback to parse local file path directly if local URI (Hướng 1)
    if optimal_threshold is None and (job.model_uri.startswith("file://") or os.path.exists(job.model_uri)):
        try:
            path = urlparse(job.model_uri).path
            if os.name == "nt" and len(path) > 3 and path[0] == "/" and path[2] == ":":
                path = path[1:]
            threshold_path = os.path.join(os.path.dirname(path), "threshold.json")
            if os.path.exists(threshold_path):
                with open(threshold_path, "r") as f:
                    threshold_data = json.load(f)
                    optimal_threshold = float(threshold_data.get("optimal_threshold"))
        except Exception as ex:
            logger.error(f"Failed to load optimal_threshold from local file path: {ex}")

    if optimal_threshold is None:
        optimal_threshold = 0.5
        threshold_source = "fallback_default"
        logger.warning("Warning: optimal_threshold not found. Using default fallback threshold of 0.5.")
        
    return optimal_threshold, threshold_source

def classify_risk(prob: float, threshold: float) -> str:
    if prob >= min(threshold * 1.5, 0.85):
        return "High"
    elif prob >= threshold:
        return "Medium"
    else:
        return "Low"

@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(req: BatchPredictionRequest, db: Session = Depends(get_db)):
    # 1. Get Best Model for Dataset
    job = db.query(TrainingJob).filter(
        TrainingJob.dataset_id == req.dataset_id, 
        TrainingJob.status == "completed"
    ).order_by(TrainingJob.roc_auc.desc()).first()
    
    if not job or not job.model_uri:
        raise HTTPException(status_code=404, detail="No completed training job found for this dataset")

    # 2. Load Model (cached)
    try:
        model = model_cache.get_model(job.model_uri)
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    # 3. Load Data from R2
    try:
        content = storage.download_file(req.file_path)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"File not found in storage: {str(e)}")

    from ...core.ingestion.parsers import parse_file
    try:
        result = parse_file(content=content, filename=req.file_path)
        df = result.df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse dataset file: {str(e)}")

    if df is None or df.empty:
         raise HTTPException(status_code=400, detail="Empty or invalid dataset dataframe")

    # 4. Load optimal_threshold using helper (Fix 2)
    optimal_threshold, threshold_source = get_optimal_threshold(job)

    # 5. Get identifier column if exists
    id_col = None
    for c in df.columns:
        if c.lower() in ["customerid", "customer_id", "id", "record_id", "uuid"]:
            id_col = c
            break

    # Determine feature columns used by the model
    feature_cols = []
    if hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)
    elif hasattr(model, "steps") and hasattr(model.steps[0][1], "feature_names_in_"):
        feature_cols = list(model.steps[0][1].feature_names_in_)
    
    if not feature_cols:
        feature_cols = [
            c for c in df.columns 
            if c != job.target_column and c.lower() not in ["customerid", "customer_id", "id", "record_id", "uuid", "churn"]
        ]
        
    try:
        # 6. Predict
        probabilities = model.predict_proba(df[feature_cols])[:, 1]
    except Exception as e:
        try:
            probabilities = model.predict_proba(df)[:, 1]
        except Exception as fallback_e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(fallback_e)}")

    results = []
    high_count = 0
    medium_count = 0
    low_count = 0

    for i, prob in enumerate(probabilities):
        risk = classify_risk(float(prob), optimal_threshold)
        if risk == "High":
            high_count += 1
        elif risk == "Medium":
            medium_count += 1
        else:
            low_count += 1
            
        record_id = str(df.iloc[i][id_col]) if id_col else str(i)
        results.append({
            "record_id": record_id,
            "probability": float(prob),
            "risk_level": risk
        })

    return {
        "total_records": len(df),
        "high_risk": high_count,
        "medium_risk": medium_count,
        "low_risk": low_count,
        "predictions": results,
        "threshold_used": optimal_threshold,
        "threshold_source": threshold_source
    }

@router.get("/datasets/{dataset_id}/feature-importance")
async def get_feature_importance(dataset_id: str, db: Session = Depends(get_db)):
    # 1. Get Best Model for Dataset
    job = db.query(TrainingJob).filter(
        TrainingJob.dataset_id == dataset_id, 
        TrainingJob.status == "completed"
    ).order_by(TrainingJob.roc_auc.desc()).first()
    
    if not job or not job.model_uri:
        raise HTTPException(status_code=404, detail="No completed training job found for this dataset")

    # 2. Load Model
    try:
        model = model_cache.get_model(job.model_uri)
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    # 3. Try to extract feature importances
    estimator = None
    if hasattr(model, "steps"):
        estimator = model.steps[-1][1]
    else:
        estimator = model
        
    importances = None
    feature_names = []
    
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_.tolist()
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
        elif hasattr(estimator, "feature_names_in_"):
            feature_names = list(estimator.feature_names_in_)
        else:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
            
    elif hasattr(estimator, "coef_"):
        import numpy as np
        coef = estimator.coef_
        if len(coef.shape) > 1:
            coef = coef[0]
        importances = np.abs(coef).tolist()
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
        else:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
            
    if importances is None:
        return {"feature_importances": []}
        
    sorted_importances = sorted(
        [{"feature": f, "importance": imp} for f, imp in zip(feature_names, importances)],
        key=lambda x: x["importance"],
        reverse=True
    )
    
    return {"feature_importances": sorted_importances}

@router.post("/datasets/{dataset_id}/drift", response_model=DriftResponse)
async def detect_dataset_drift(dataset_id: str, req: DriftRequest, db: Session = Depends(get_db)):
    # 1. Get Dataset and its profile
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    profile_record = db.query(Profile).filter(Profile.dataset_id == dataset_id).first()
    if not profile_record or not profile_record.profiles_json:
        raise HTTPException(status_code=404, detail="Dataset profile not found")

    # 2. Get Best Completed Training Job to identify feature columns
    job = db.query(TrainingJob).filter(
        TrainingJob.dataset_id == dataset_id,
        TrainingJob.status == "completed"
    ).order_by(TrainingJob.roc_auc.desc()).first()
    
    if not job or not job.model_uri:
        raise HTTPException(status_code=404, detail="No completed training job found for this dataset")

    # 3. Load Model to extract feature names
    try:
        model = model_cache.get_model(job.model_uri)
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    # 4. Determine feature columns and their types from profiles
    profiles_list = profile_record.profiles_json
    numerical_cols = [p["name"] for p in profiles_list if p["inferred_role"] == "NUMERIC"]
    categorical_cols = [p["name"] for p in profiles_list if p["inferred_role"] == "CATEGORICAL"]

    feature_cols = []
    if hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)
    elif hasattr(model, "steps") and hasattr(model.steps[0][1], "feature_names_in_"):
        feature_cols = list(model.steps[0][1].feature_names_in_)
    
    if not feature_cols:
        feature_cols = [
            c for c in numerical_cols + categorical_cols
            if c != job.target_column and c.lower() not in ["customerid", "customer_id", "id", "record_id", "uuid", "churn"]
        ]

    # 5. Download original training dataset (reference)
    try:
        ref_content = storage.download_file(dataset.r2_path)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Reference training dataset not found in storage: {str(e)}")

    # 6. Download target inference dataset (target)
    try:
        target_content = storage.download_file(req.target_file_path)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Target inference dataset not found in storage: {str(e)}")

    # Parse dataframes
    from ...core.ingestion.parsers import parse_file
    try:
        ref_df = parse_file(content=ref_content, filename=dataset.filename).df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse reference dataset: {str(e)}")

    try:
        target_df = parse_file(content=target_content, filename=req.target_file_path).df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse target dataset: {str(e)}")

    if ref_df is None or ref_df.empty or target_df is None or target_df.empty:
         raise HTTPException(status_code=400, detail="Empty reference or target dataframe")

    # 7. Run drift detector
    from ...core.serving.drift_detector import calculate_drift_report
    report = calculate_drift_report(
        reference_df=ref_df,
        target_df=target_df,
        feature_cols=feature_cols,
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols
    )

    return {
        "dataset_id": dataset_id,
        "reference_rows": len(ref_df),
        "target_rows": len(target_df),
        "drift_detected": report["drift_detected"],
        "metrics": report["metrics"]
    }
