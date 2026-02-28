# Deployment

## API Container
- Dockerfile: deploy/api/Dockerfile
- Minimal requirements: deploy/api/requirements.txt
- Environment variables at runtime:
  - MLFLOW_S3_ENDPOINT_URL
  - AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
  - MLFLOW_TRACKING_URI
  - MODEL_NAME / MODEL_VERSION

## Build & Run (example)
- Build: docker build -t cdp-api:latest -f deploy/api/Dockerfile .
- Run: docker run -p 8000:8000 --env-file .env cdp-api:latest

## Compose with MinIO + MLflow
- The API service is declared in docker-compose.yml (service `api`).
- Start all services:
  - docker compose up -d
- Access:
  - API: http://localhost:8000
  - MLflow UI: http://localhost:5000
  - MinIO Console: http://localhost:9001 (user: admin, pwd: password)

### Rebuild API after code changes
- docker compose build api && docker compose up -d api

## CI/CD (GitHub Container Registry)
- Workflow: .github/workflows/api-image.yml
- Pushes to main will build and push to GHCR at `ghcr.io/<OWNER>/cdp-api`.
- Permissions: workflow sets `packages: write` and logs in with `GITHUB_TOKEN`.

### Pull & Run
- Pull: `docker pull ghcr.io/<OWNER>/cdp-api:main`
- Run: `docker run -p 8000:8000 --env-file .env ghcr.io/<OWNER>/cdp-api:main`
