install:
	pip install -r requirements.txt
	pip install ruff black pytest pytest-cov

lint:
	ruff check .

format-check:
	black --check .

format-fix:
	black .
	ruff check . --fix

test:
	@mkdir -p data/raw data/parquet data/processed
	export MLFLOW_TRACKING_URI=http://localhost:5000; \
	export DATABASE_URL=postgresql://user:pass@localhost:5432/db; \
	pytest tests/ -v --maxfail=3

ci: lint format-check test
	@echo "✅ All local CI checks passed!"

etl:
	python launcher.py