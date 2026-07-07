import time
import threading
from datetime import datetime, timedelta
import mlflow
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from ...db.session import get_db
from ...db.models import TrainingJob
from ...core.storage import StorageClient

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

def ping_with_timeout(check_fn, timeout_sec=3.0) -> dict:
    result = {"status": "down", "error": "timeout"}
    start = time.monotonic()

    def run():
        try:
            check_fn()
            ms = (time.monotonic() - start) * 1000
            result.update({
                "status": "up",
                "latency_ms": round(ms, 2)
            })
            result.pop("error", None)
        except Exception as e:
            result.update({
                "status": "down",
                "error": str(e)[:100]
            })

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)
    return result

@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    services = {
        "database": ping_with_timeout(
            lambda: db.execute(text("SELECT 1"))
        ),
        "storage": ping_with_timeout(
            lambda: StorageClient().ping()
        ),
        "mlflow": ping_with_timeout(
            lambda: mlflow.tracking.MlflowClient().search_experiments(max_results=1)
        )
    }
    statuses = [v["status"] for v in services.values()]
    overall = (
        "healthy" if all(s == "up" for s in statuses)
        else "degraded" if any(s == "up" for s in statuses)
        else "unhealthy"
    )
    return {
        "status": overall,
        "timestamp": datetime.utcnow().isoformat(),
        "services": services
    }

@router.get("/metrics")
def get_metrics(db: Session = Depends(get_db)):
    now = datetime.utcnow()
    last_24h = now - timedelta(hours=24)

    # Job counts từ DB
    def count_jobs(status, since=None):
        q = db.query(TrainingJob).filter(TrainingJob.status == status)
        if since:
            q = q.filter(TrainingJob.finished_at >= since)
        return q.count()

    # Storage stats — graceful degrade
    try:
        files = StorageClient().list_files("")
        total_files = len(files)
    except Exception:
        total_files = -1

    return {
        "timestamp": now.isoformat(),
        "jobs": {
            "queued": count_jobs("queued"),
            "training": count_jobs("training"),
            "completed_24h": count_jobs("completed", last_24h),
            "failed_24h": count_jobs("failed", last_24h)
        },
        "storage": {
            "total_files": total_files
        },
        "predict_endpoint": {
            "note": "metrics not yet tracked"
        }
    }

@router.get("/jobs/summary")
def jobs_summary(db: Session = Depends(get_db)):
    last_7d = datetime.utcnow() - timedelta(days=7)
    jobs = db.query(TrainingJob).filter(
        TrainingJob.started_at >= last_7d
    ).all()

    by_status = {}
    for job in jobs:
        by_status[job.status] = by_status.get(job.status, 0) + 1

    completed = [j for j in jobs if j.roc_auc is not None]
    best = max(completed, key=lambda j: j.roc_auc, default=None)

    return {
        "period": "last_7_days",
        "total_jobs": len(jobs),
        "by_status": by_status,
        "best_model": {
            "job_id": str(best.id),
            "roc_auc": best.roc_auc,
            "dataset_id": best.dataset_id
        } if best else None
    }
