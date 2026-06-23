import os
import mlflow

def setup_mlflow() -> str:
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000"  # fallback local
    )
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "churn-prediction"
    )
    mlflow.set_experiment(experiment_name)

    return tracking_uri
