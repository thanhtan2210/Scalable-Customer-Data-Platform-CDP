# Stage 1: Base dependencies (không có torch)
FROM python:3.11-slim AS base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --default-timeout=1000 --no-cache-dir \
    $(grep -v torch requirements.txt | grep -v pyspark | grep -v "^#" | tr '\n' ' ')

# Stage 2: Full (có torch — chỉ dùng production)
FROM base AS full
RUN pip install --no-cache-dir \
    torch --index-url \
    https://download.pytorch.org/whl/cpu

# Copy app code
COPY backend/app ./app
RUN touch app/__init__.py

# Security: chạy với non-root user
RUN addgroup --system --gid 1001 appgroup \
 && adduser --system --uid 1001 \
            --ingroup appgroup appuser

# Đảm bảo appuser có quyền đọc code
RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["gunicorn", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "app.main:app"]

# Stage 3: Dev (không có torch — dùng local .venv)
FROM base AS dev
ENV ENABLE_LLM_LAYER=false
ENV TORCH_AVAILABLE=false

# Copy app code
COPY backend/app ./app
RUN touch app/__init__.py

# Security: chạy với non-root user
RUN addgroup --system --gid 1001 appgroup \
 && adduser --system --uid 1001 \
            --ingroup appgroup appuser

# Đảm bảo appuser có quyền đọc code
RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
