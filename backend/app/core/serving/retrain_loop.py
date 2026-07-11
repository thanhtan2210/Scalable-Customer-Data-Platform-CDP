import asyncio
import logging
import httpx
from datetime import datetime, timedelta
import os

from ...core import config
from ...db.session import SessionLocal
from ...db.models import TrainingJob

logger = logging.getLogger("cdp.retrain_loop")


async def cleanup_old_inference_files():
    """Xóa inference parquet files cũ hơn INFERENCE_RETENTION_DAYS ngày."""
    try:
        from ...core.storage import storage

        INFERENCE_RETENTION_DAYS = int(os.getenv("INFERENCE_RETENTION_DAYS", "30"))
        cutoff = datetime.utcnow() - timedelta(days=INFERENCE_RETENTION_DAYS)

        # List tất cả inference files
        all_files = storage.list_files("ml_artifacts/")
        inference_files = [f for f in all_files if "/inference/" in f]

        deleted = 0
        for file_path in inference_files:
            try:
                # Parse date từ path:
                # ml_artifacts/{id}/inference/{YYYY-MM-DD}/{batch_id}.parquet
                date_str = file_path.split("/inference/")[1].split("/")[0]
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if file_date < cutoff:
                    storage.delete_file(file_path)
                    deleted += 1
            except (ValueError, IndexError):
                continue  # skip malformed paths

        if deleted > 0:
            logger.info(
                f"Cleaned up {deleted} inference files older than {INFERENCE_RETENTION_DAYS} days"
            )
    except Exception as e:
        logger.error(f"Inference cleanup failed: {e}")


async def run_drift_check_loop():
    logger.info("Drift auto-retrain loop started.")

    interval = config.DRIFT_CHECK_INTERVAL_SEC
    api_key = config.API_KEY
    headers = {"X-API-Key": api_key}

    # Wait a short delay on startup so the Uvicorn server is up and listening
    await asyncio.sleep(10)

    while True:
        try:
            logger.info("Scanning for dataset drift...")
            db = SessionLocal()
            try:
                active_dataset_ids = [
                    r[0]
                    for r in db.query(TrainingJob.dataset_id)
                    .filter(TrainingJob.status == "completed")
                    .distinct()
                    .all()
                ]
            finally:
                db.close()

            async with httpx.AsyncClient(timeout=60.0) as client:
                for ds_id in active_dataset_ids:
                    logger.info(f"Checking drift for dataset {ds_id}...")
                    drift_url = f"http://localhost:8000/api/v1/predict/{ds_id}/drift"
                    try:
                        resp = await client.post(drift_url, json={}, headers=headers)
                        if resp.status_code == 200:
                            drift_data = resp.json()
                            if drift_data.get("drift_detected", False):
                                logger.warning(
                                    f"Drift detected for dataset {ds_id}! Triggering auto-retrain..."
                                )
                                train_url = "http://localhost:8000/api/v1/jobs/train"
                                train_payload = {
                                    "dataset_id": ds_id,
                                    "target_column": "churn_label",
                                }

                                db = SessionLocal()
                                try:
                                    best_job = (
                                        db.query(TrainingJob)
                                        .filter(
                                            TrainingJob.dataset_id == ds_id,
                                            TrainingJob.status == "completed",
                                        )
                                        .order_by(TrainingJob.roc_auc.desc())
                                        .first()
                                    )
                                    if best_job:
                                        train_payload["target_column"] = (
                                            best_job.target_column
                                        )
                                finally:
                                    db.close()

                                train_resp = await client.post(
                                    train_url, json=train_payload, headers=headers
                                )
                                if train_resp.status_code == 200:
                                    logger.info(
                                        f"Auto-retrain job triggered successfully for dataset {ds_id}: {train_resp.json()}"
                                    )
                                else:
                                    logger.error(
                                        f"Failed to trigger auto-retrain for {ds_id}: {train_resp.status_code} - {train_resp.text}"
                                    )
                        elif resp.status_code == 404:
                            logger.info(
                                f"No inference data found for dataset {ds_id} today, skipping."
                            )
                        else:
                            logger.error(
                                f"Drift check API error for {ds_id}: {resp.status_code} - {resp.text}"
                            )
                    except Exception as api_err:
                        logger.error(
                            f"Error calling drift API for dataset {ds_id}: {api_err}"
                        )

            # Run old inference files cleanup
            await cleanup_old_inference_files()

        except Exception as loop_err:
            logger.error(f"Error in drift check loop iteration: {loop_err}")

        logger.info(f"Drift loop sleeping for {interval} seconds.")
        await asyncio.sleep(interval)
