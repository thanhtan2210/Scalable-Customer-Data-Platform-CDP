FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy minimal requirements for the dashboard
COPY requirements.txt /app/requirements-streamlit.txt

RUN pip install --no-cache-dir -r requirements-streamlit.txt

# Copy the dashboard
COPY Sales_Dashboard /app/Sales_Dashboard

EXPOSE 8501

CMD ["streamlit", "run", "Sales_Dashboard/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
