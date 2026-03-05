FROM python:3.10-slim
WORKDIR /app
COPY src/ src/
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080
CMD ["uvicorn", "src.api.ab_service:app", "--host", "0.0.0.0", "--port", "8080"]
