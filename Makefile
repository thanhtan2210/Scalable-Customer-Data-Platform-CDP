.PHONY: dev test lint migrate shell

# Environment variables
PYTHONPATH := $(shell pwd)

# Install local dependencies
install-local:
	pip install -r requirements-local.txt

# Local Development
dev:
	docker-compose up --build

# Backend tasks
test:
	python -m pytest backend/tests -v

lint:
	ruff check backend/app
	black backend/app --check
	mypy backend/app

# Database migrations
migrate-init:
	cd backend/app/db && alembic init migrations

migrate:
	cd backend/app/db && alembic upgrade head

# Clean up
clean:
	docker-compose down -v
	find . -type d -name "__pycache__" -exec rm -rf {} +
