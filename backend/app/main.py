from fastapi import FastAPI, Security, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from datetime import datetime
from .api.v1 import datasets, predict, jobs
from .core.config import ENVIRONMENT, API_KEY, ALLOWED_ORIGINS
from .core.logging_config import setup_logging

# Initialize Logging
setup_logging()

# Enforce security check: prevent starting in production with the default API key
if ENVIRONMENT == "production" and API_KEY == "test-api-key":
    raise RuntimeError(
        "CRITICAL SECURITY ERROR: API_KEY cannot be the default 'test-api-key' in production environment!"
    )

app = FastAPI(title="Churn Prediction Platform API")

@app.on_event("startup")
def startup_db():
    from .db.session import engine
    from .db.models import Base
    Base.metadata.create_all(bind=engine)

# Configure CORS Middleware
origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

