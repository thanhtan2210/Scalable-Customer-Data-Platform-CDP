# scripts/run_e2e.ps1
# Script to run Phase 3 E2E test with proper Docker container cleanup and sequencing to bypass Windows race conditions.

Write-Host "Stopping and cleaning up any running Docker containers..." -ForegroundColor Cyan
docker compose down -v

Write-Host "Force removing conflict containers if any exist..." -ForegroundColor Cyan
docker rm -f churn_platform_api churn_platform_mlflow churn_platform_db churn_platform_s3 2>$null
docker container prune -f

Write-Host "Waiting 10 seconds for Docker daemon to settle..." -ForegroundColor Cyan
Start-Sleep 10

Write-Host "Starting minimal backend stack (db, minio, mlflow, backend)..." -ForegroundColor Cyan
docker compose up db minio mlflow backend -d

Write-Host "Waiting 30 seconds for services to fully initialize..." -ForegroundColor Cyan
Start-Sleep 30

Write-Host "Verifying running container status..." -ForegroundColor Cyan
docker compose ps

Write-Host "Running Phase 3 E2E tests..." -ForegroundColor Green
.venv\Scripts\pytest tests/e2e/test_phase3_e2e.py -v --tb=short -s

Write-Host "Cleaning up services..." -ForegroundColor Cyan
docker compose down -v
Write-Host "E2E Run completed." -ForegroundColor Green
