FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

COPY analytics/ ./analytics
COPY backend/app/api/schemas.py ./backend/app/api/schemas.py
# Copy other necessary core modules if needed for analytics

EXPOSE 8501

CMD ["streamlit", "run", "analytics/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
