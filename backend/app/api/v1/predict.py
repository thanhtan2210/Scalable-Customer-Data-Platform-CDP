from fastapi import APIRouter, Depends, HTTPException
from typing import List
import pandas as pd
import io
from sqlalchemy.orm import Session
from ...db.models import Dataset, TrainingJob
from ...db.session import get_db
from ...api.schemas import PredictionRequest, PredictionResponse
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
