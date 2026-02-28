#!/usr/bin/env python3
"""API container entrypoint.

Responsibilities:
- Load environment variables from /app/.env if present (optional)
- Wait for dependent services (MinIO, MLflow)
- Run optional migration hooks (placeholder)
- Exec Uvicorn to start the FastAPI app
"""
import os
import sys
import subprocess
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

ROOT = Path(__file__).resolve().parents[1]


def load_env_file(env_path: Path):
    if not env_path.exists():
        return
    if load_dotenv:
        load_dotenv(dotenv_path=str(env_path))
        print(f"Loaded env from {env_path}")
    else:
        # Basic parser fallback
        print("python-dotenv not installed, parsing .env manually")
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def load_docker_secrets(secrets_dir: Path = Path("/run/secrets")):
    """Load Docker secrets from files in /run/secrets into environment.

    Each filename becomes the environment variable name. Files without readable
    content are skipped.
    """
    if not secrets_dir.exists() or not secrets_dir.is_dir():
        return
    for p in secrets_dir.iterdir():
        if not p.is_file():
            continue
        try:
            val = p.read_text().strip()
            if val:
                os.environ[p.name] = val
                print(f"Loaded secret into env: {p.name}")
        except Exception as e:
            print(f"Could not read secret {p}: {e}")


def run_wait_for(urls):
    if not urls:
        return
    wait_script = ROOT / 'scripts' / 'wait_for.py'
    if not wait_script.exists():
        print(f"wait_for script not found at {wait_script}; skipping wait")
        return
    cmd = [sys.executable, str(wait_script)] + urls
    print("Running wait-for:", cmd)
    rc = subprocess.call(cmd)
    if rc != 0:
        print("wait_for failed; exiting", file=sys.stderr)
        sys.exit(rc)


def run_migrations():
    # Run DB migrations (if configured) and preload model from MLflow (if configured).
    # 1) DB migrations via Alembic if available and DATABASE_URL set
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        try:
            import alembic.config
            import alembic.command

            print("Running Alembic migrations...")
            cfg = alembic.config.Config("alembic.ini")
            # Override sqlalchemy.url from env
            cfg.set_main_option("sqlalchemy.url", db_url)
            alembic.command.upgrade(cfg, "head")
            print("Alembic migrations applied.")
        except Exception as e:
            print(f"Alembic migrations skipped/failed: {e}")

    # 2) Model preloading from MLflow model registry
    model_name = os.environ.get("MODEL_NAME")
    model_version = os.environ.get("MODEL_VERSION", "1")
    if model_name:
        try:
            import mlflow
            import joblib

            mlflow.set_tracking_uri(os.environ.get(
                "MLFLOW_TRACKING_URI", "http://mlflow:5000"))
            model_uri = f"models:/{model_name}/{model_version}"
            print(f"Preloading model from MLflow: {model_uri}")
            # Try sklearn flavor first (if available)
            try:
                sk_model = mlflow.sklearn.load_model(model_uri)
                out_dir = Path("/app/models")
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{model_name}.joblib"
                joblib.dump(sk_model, out_path)
                print(f"Saved sklearn model to {out_path}")
            except Exception:
                # Fallback to pyfunc to ensure model artifacts are downloaded/cached
                try:
                    pyf = mlflow.pyfunc.load_model(model_uri)
                    # Saving pyfunc is non-trivial; instead, ensure it's cached by calling predict on empty input if supported
                    try:
                        import pandas as pd

                        dummy = pd.DataFrame()
                        # Some pyfunc models may error on empty input; ignore
                        _ = pyf.predict(dummy)
                    except Exception:
                        pass
                    print("MLflow pyfunc model loaded (cached).")
                except Exception as e:
                    print(f"Failed to preload MLflow model: {e}")
        except Exception as e:
            print(f"Model preload failed: {e}")


def exec_uvicorn(app_module: str = "src.api.main:app"):
    cmd = ["uvicorn", app_module, "--host", "0.0.0.0",
           "--port", os.environ.get("PORT", "8000")]
    print("Starting server:", cmd)
    os.execvp(cmd[0], cmd)


def main():
    # 1. Load Docker secrets first (production-friendly)
    load_docker_secrets(Path("/run/secrets"))

    # 2. Load .env if present in image root (fallback)
    env_path = ROOT / '.env'
    load_env_file(env_path)

    # 2. Wait for dependencies (override via ENV WAIT_URLS as space-separated)
    wait_urls = os.environ.get('WAIT_URLS')
    if wait_urls:
        urls = wait_urls.split()
    else:
        urls = [
            os.environ.get('MLFLOW_HEALTH_URL', 'http://mlflow:5000/'),
            os.environ.get('MINIO_HEALTH_URL',
                           'http://minio:9000/minio/health/live'),
        ]
    run_wait_for(urls)

    # 3. Run migrations/hooks
    run_migrations()

    # 4. Exec server
    exec_uvicorn()


if __name__ == '__main__':
    main()
