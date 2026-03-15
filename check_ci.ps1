# check_ci.ps1 - Comprehensive Local CI/CD Runner for Windows
# Usage: .\check_ci.ps1

Write-Host " Starting COMPREHENSIVE Local CI/CD Checks..." -ForegroundColor Cyan

# --- STEP 1: PREPARATION ---
Write-Host "`n Step 1: Preparing directories and mock data..." -ForegroundColor Yellow
$dirs = "data/raw", "data/parquet", "data/processed", "reports", "ci_outputs"
foreach ($dir in $dirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
# Create dummy data for A/B tests if missing
$dummy_csv = "data/raw/cleaned_telco.csv"
if (!(Test-Path $dummy_csv)) {
    "CustomerID,Gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,InternetService,MonthlyCharges,TotalCharges,Churn" | Out-File -FilePath $dummy_csv -Encoding utf8
    "1,Male,0,Yes,No,1,Yes,DSL,50,100,0" | Out-File -Append -FilePath $dummy_csv -Encoding utf8
    Write-Host "   + Created dummy data for A/B tests."
}

# --- STEP 2: DEPENDENCIES ---
Write-Host "`n Step 2: Verifying dependencies..." -ForegroundColor Yellow
python -m pip install --quiet ruff black pytest pytest-cov sqlalchemy psycopg2-binary nbconvert jupyter pandas
Write-Host "   Dependencies verified."

# --- STEP 3: CDP PIPELINE (LINT/FORMAT/TEST) ---
Write-Host "`n🔍 Step 3: Running CDP Pipeline Checks (Lint/Format/Unit Tests)..." -ForegroundColor Yellow
$env:PYTHONPATH = "."

Write-Host "   - Ruff (Linting)..."
ruff check .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "   - Black (Formatting)..."
black --check .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "   - Pytest (Logic)..."
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:DATABASE_URL = "postgresql://user:pass@localhost:5432/db"
pytest tests/ -v --maxfail=3
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# --- STEP 4: A/B NOTEBOOK WORKFLOW ---
Write-Host "`n Step 4: Simulating A/B Notebook CI..." -ForegroundColor Yellow
try {
    Write-Host "   - Running A/B Assignment..."
    python scripts/ab_assign.py --input data/raw/cleaned_telco.csv --out reports/ab_assignment.csv --ratio 0.5
    
    Write-Host "   - Generating outcomes..."
    python scripts/generate_outcomes_from_exposures_csv.py
    
    Write-Host "   - Testing Notebook Execution (nbconvert)..."
    jupyter nbconvert --to notebook --execute notebooks/ab_analysis.ipynb --ExecutePreprocessor.timeout=60 --output ci_outputs/test_out.ipynb
    Write-Host "    A/B Workflow simulation passed." -ForegroundColor Green
}
catch {
    Write-Host "    A/B Workflow failed!" -ForegroundColor Red
    exit 1
}

# --- STEP 5: DOCKER IMAGE BUILD (Optional) ---
Write-Host "`n Step 5: Checking Docker API Image Build (Dry-run)..." -ForegroundColor Yellow
$docker_check = Get-Command docker -ErrorAction SilentlyContinue
if ($docker_check) {
    Write-Host "   - Building API Image locally..."
    docker build -f deploy/api/Dockerfile . -t api-test-local --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Host "    Docker build successful." -ForegroundColor Green
    }
    else {
        Write-Host "    Docker build failed!" -ForegroundColor Red
        exit 1
    }
}
else {
    Write-Host "     Docker not found. Skipping image build check." -ForegroundColor Gray
}

Write-Host "`n ALL SYSTEM CHECKS PASSED! YOU ARE READY TO PUSH. ✨✨" -ForegroundColor Green
