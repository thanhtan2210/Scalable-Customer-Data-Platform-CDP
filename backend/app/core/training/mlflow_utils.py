import os
import mlflow


def setup_mlflow() -> str:
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI", "http://localhost:5000"  # fallback local
    )
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-prediction")
    mlflow.set_experiment(experiment_name)

    return tracking_uri


def cleanup_old_runs(experiment_name: str, keep_last_n: int = 5):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        return

    try:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attribute.start_time DESC"],
        )
    except Exception as e:
        # Fallback if attribute.start_time DESC is not supported by backend
        try:
            runs = client.search_runs(experiment_ids=[experiment.experiment_id])
            runs = sorted(runs, key=lambda r: r.info.start_time, reverse=True)
        except Exception:
            return

    # Keep only the last N runs, delete older runs
    for run in runs[keep_last_n:]:
        try:
            client.delete_run(run.info.run_id)
        except Exception:
            pass  # Ignore if run cannot be deleted (e.g. active run)
