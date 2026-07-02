import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path to enable module imports
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))


def setup_spark_env():
    """Configure Spark environment if running on Windows."""
    if platform.system() == "Windows":
        # Look for Hadoop (Winutils) directory
        hadoop_dir = BASE_DIR / "bin" / "hadoop"
        if hadoop_dir.exists():
            os.environ["HADOOP_HOME"] = str(hadoop_dir)
            os.environ["PATH"] += os.pathsep + str(hadoop_dir / "bin")
            print("🪟 Windows: Hadoop environment configured.")


def run_spark_clean():
    """Execute Spark data cleaning process."""
    print("🚀 Running Spark Data Cleaning...")
    setup_spark_env()
    from backend.jobs.jobs.clean_data_spark import run as spark_run

    spark_run()


def run_ab_service():
    """Start the A/B Testing API service."""
    print("🛰️ Starting A/B Testing Service (FastAPI)...")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "backend.app.core.serving.ab_service:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8081",
                "--reload",
            ]
        )
    except KeyboardInterrupt:
        print("\n🛑 Service stopped.")


def run_churn_api():
    """Start the Churn Prediction API service."""
    print("🔮 Starting Churn Prediction API (FastAPI)...")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "backend.app.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8001",
                "--reload",
            ]
        )
    except KeyboardInterrupt:
        print("\n🛑 Service stopped.")


def run_dashboard():
    """Launch the Streamlit dashboard."""
    print("📊 Starting Sales Dashboard (Streamlit)...")
    dashboard_path = BASE_DIR / "analytics" / "streamlit_app.py"
    try:
        subprocess.run(["streamlit", "run", str(dashboard_path)])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped.")


def run_pipeline():
    """Execute the full data pipeline."""
    print("⚙️ Running Entire Data Pipeline (Pandas fallback)...")
    from backend.jobs.jobs.clean_data import run_job
    run_job()


def run_train():
    """Train the churn prediction model (with MLflow tracking)."""
    print("🧠 Local legacy training script has been removed for production hardening.")
    print("💡 Please trigger training via the REST API endpoint: POST /api/v1/jobs/datasets/{dataset_id}/train")


def main():
    parser = argparse.ArgumentParser(description="CDP Project Launcher CLI")
    parser.add_argument(
        "command",
        choices=[
            "spark-clean",
            "ab-service",
            "churn-api",
            "dashboard",
            "pipeline",
            "train",
        ],
        help="Command to execute",
    )

    args = parser.parse_args()

    commands = {
        "spark-clean": run_spark_clean,
        "ab-service": run_ab_service,
        "churn-api": run_churn_api,
        "dashboard": run_dashboard,
        "pipeline": run_pipeline,
        "train": run_train,
    }

    if args.command in commands:
        commands[args.command]()


if __name__ == "__main__":
    main()
