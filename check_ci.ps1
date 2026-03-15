# check_ci.ps1 - Local CI/CD Runner for Windows
# Usage: .\check_ci.ps1

Write-Host " Starting Local CI/CD Checks..." -ForegroundColor Cyan

# 1. Create necessary directories
Write-Host "`n Step 1: Preparing data directories..." -ForegroundColor Yellow
$dirs = "data/raw", "data/parquet", "data/processed", "reports"
foreach ($dir in $dirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "   + Created $dir"
    }
}

# 2. Check and Install Dependencies
Write-Host "`n Step 2: Checking dependencies..." -ForegroundColor Yellow
python -m pip install --quiet ruff black pytest pytest-cov sqlalchemy psycopg2-binary
Write-Host "`n   + Dependencies verified."

# 3. Linting Check (Ruff)
Write-Host "`n Step 3: Running Linting (Ruff)..." -ForegroundColor Yellow
ruff check .
if ($LASTEXITCODE -ne 0) {
    Write-Host " Linting failed! Please fix errors above." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "   Linting passed." -ForegroundColor Green

# 4. Formatting Check (Black)
Write-Host "`n Step 4: Checking Format (Black)..." -ForegroundColor Yellow
black --check .
if ($LASTEXITCODE -ne 0) {
    Write-Host " Formatting issues found! Run 'black .' to fix them automatically." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "  Formatting passed." -ForegroundColor Green

# 5. Unit Tests (Pytest)
Write-Host "`n Step 5: Running Unit Tests (Pytest)..." -ForegroundColor Yellow
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:DATABASE_URL = "postgresql://user:pass@localhost:5432/db"
$env:PYTHONPATH = "."

pytest tests/ -v --maxfail=3
if ($LASTEXITCODE -ne 0) {
    Write-Host " Tests failed! Check the output above." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "`n All local CI checks passed! You are ready to push. ✨" -ForegroundColor Green
