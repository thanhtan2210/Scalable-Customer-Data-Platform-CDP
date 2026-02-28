FROM python:3.11-slim AS builder

WORKDIR /wheels

# Build wheels for all requirements to avoid installing build deps in runtime
COPY deploy/api/requirements.txt /app/requirements-api.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev && \ 
    python -m pip wheel -r /app/requirements-api.txt -w /wheels && rm -rf /var/lib/apt/lists/*

FROM python:3.11-slim

WORKDIR /app

# Copy wheels from builder and install them without network
COPY --from=builder /wheels /wheels
COPY deploy/api/requirements.txt /app/requirements-api.txt
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r /app/requirements-api.txt && rm -rf /wheels

# Copy runtime files
COPY src /app/src
COPY scripts/wait_for.py /app/scripts/wait_for.py
COPY deploy/api/entrypoint.py /app/entrypoint.py
COPY .env.example /app/.env.example

ENV PYTHONPATH=/app

EXPOSE 8000

# Use entrypoint which loads env, waits for services and starts Uvicorn
ENTRYPOINT ["python", "/app/entrypoint.py"]
