from fastapi import APIRouter, Depends, HTTPException
from typing import List
import pandas as pd
import io
from sqlalchemy.orm import Session
from ...db.models import Dataset, TrainingJob
from ...db.session import get_db
from ...api.schemas import PredictionRequest, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse
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
    
    try:
        # 4. Predict
        probabilities = model.predict_proba(input_df)[:, 1]
        
        results = []
        for i, prob in enumerate(probabilities):
            risk = "High" if prob > 0.7 else ("Medium" if prob > 0.4 else "Low")
            results.append({
                "record_id": req.records[i].get("id") or str(i),
                "churn_probability": float(prob),
                "risk_level": risk
            })
            
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

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

    # 4. Load optimal_threshold from MLflow artifact or DB fallback
    import re
    import json
    import mlflow
    optimal_threshold = 0.5
    match = re.search(r"runs:/+([^/]+)", job.model_uri)
    if match:
        run_id = match.group(1)
        try:
            client = mlflow.tracking.MlflowClient()
            local_path = client.download_artifacts(run_id, "threshold.json")
            with open(local_path, "r") as f:
                threshold_data = json.load(f)
                optimal_threshold = float(threshold_data.get("optimal_threshold", 0.5))
        except Exception as ex:
            print(f"Failed to load optimal_threshold from MLflow artifact: {ex}")
            if job.optimal_threshold is not None:
                optimal_threshold = job.optimal_threshold

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
        if prob >= optimal_threshold:
            risk = "High"
            high_count += 1
        elif prob >= optimal_threshold * 0.5:
            risk = "Medium"
            medium_count += 1
        else:
            risk = "Low"
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
        "threshold_used": optimal_threshold
    }
