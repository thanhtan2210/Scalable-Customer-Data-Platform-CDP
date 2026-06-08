from fastapi import FastAPI, Security, HTTPException, status, Depends
from fastapi.security.api_key import APIKeyHeader
import os
from datetime import datetime
from .api.v1 import datasets, predict, jobs

app = FastAPI(title="Churn Prediction Platform API")

API_KEY = os.getenv("API_KEY", "test-api-key")
api_key_header = APIKeyHeader(name="X-API-Key")

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN, detail="Could not validate credentials"
    )

# Apply Auth globally for MVP
app.include_router(datasets.router, prefix="/api/v1", dependencies=[Depends(get_api_key)])
app.include_router(jobs.router, prefix="/api/v1", dependencies=[Depends(get_api_key)])
app.include_router(predict.router, prefix="/api/v1", dependencies=[Depends(get_api_key)])

@app.get("/")
def home():
    return {"status": "alive", "version": "1.0.0"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
