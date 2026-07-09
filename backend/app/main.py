from fastapi import FastAPI, Security, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from datetime import datetime
from .api.v1 import datasets, predict, jobs, monitoring, models
from .core.serving.ab_service import router as ab_router
from .core.config import ENVIRONMENT, API_KEY, ALLOWED_ORIGINS
from .core.logging_config import setup_logging

# Initialize Logging
setup_logging()

import logging
logger = logging.getLogger("cdp.main")

# Enforce security check: prevent starting in production with the default API key
if ENVIRONMENT == "production" and API_KEY == "test-api-key":
    raise RuntimeError(
        "CRITICAL SECURITY ERROR: API_KEY cannot be the default 'test-api-key' in production environment!"
    )

import asyncio
import os
from contextlib import asynccontextmanager
from .core import config
from .core.serving.retrain_loop import run_drift_check_loop

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auto-run migrations on startup
    try:
        from alembic.config import Config
        from alembic import command
        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")
        logger.info("Database migrations: up to date")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        # Không raise — app vẫn start nhưng log warning rõ ràng

    task = None
    if config.DRIFT_AUTO_RETRAIN and config.ENABLE_DRIFT_SCHEDULER:
        task = asyncio.create_task(run_drift_check_loop())
        logger.info("Drift scheduler started on this instance")
    elif config.DRIFT_AUTO_RETRAIN:
        logger.warning(
            "DRIFT_AUTO_RETRAIN=true but ENABLE_DRIFT_SCHEDULER=false — "
            "drift checks disabled on this instance (multi-instance mode)"
        )
    yield
    # shutdown
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

from .core.config import IS_PRODUCTION

app = FastAPI(
    title="Churn Prediction Platform API",
    lifespan=lifespan,
    docs_url="/docs" if not IS_PRODUCTION else None,
    redoc_url="/redoc" if not IS_PRODUCTION else None,
    openapi_url="/openapi.json" if not IS_PRODUCTION else None
)

# Register request logging middleware
from .middleware.logging_middleware import logging_middleware
app.middleware("http")(logging_middleware)

from .core.limiter import limiter
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Configure CORS Middleware
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
allowed_origins = [o.strip() for o in allowed_origins_str.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
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
app.include_router(monitoring.router, prefix="/api/v1", dependencies=[Depends(get_api_key)])
app.include_router(ab_router, prefix="/api/v1", dependencies=[Depends(get_api_key)])
app.include_router(models.router, prefix="/api/v1", dependencies=[Depends(get_api_key)])

@app.get("/")
def home():
    return {"status": "alive", "version": "1.0.0"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

