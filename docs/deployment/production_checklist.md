# Production Deployment Checklist

## Environment Variables (bắt buộc)
- [ ] ENV=production
- [ ] DATABASE_URL=postgresql://...
      (Supabase connection string)
- [ ] S3_ENDPOINT_URL=https://...r2...
- [ ] S3_ACCESS_KEY_ID=...
- [ ] S3_SECRET_ACCESS_KEY=...
- [ ] S3_BUCKET_NAME=cdp-datalake-assets
- [ ] MLFLOW_TRACKING_URI=https://your-dagshub-tracking-uri
- [ ] MLFLOW_TRACKING_USERNAME=...
- [ ] MLFLOW_TRACKING_PASSWORD=...
- [ ] API_KEY=... (strong random key)
- [ ] ALLOWED_ORIGINS=https://your-app.vercel.app
- [ ] SECRET_KEY=... (for JWT if added)

## Environment Variables (optional)
- [ ] MAX_UPLOAD_SIZE_MB=50
- [ ] MAX_TRAINING_MINUTES=30
- [ ] OPTUNA_N_TRIALS=15
- [ ] OPTUNA_TIMEOUT_SECONDS=600
- [ ] LOG_LEVEL=INFO
- [ ] ENABLE_DRIFT_SCHEDULER=true
      (chỉ true trên 1 instance)
- [ ] DRIFT_AUTO_RETRAIN=false
      (enable sau khi stable)
- [ ] INFERENCE_RETENTION_DAYS=30
- [ ] MLFLOW_KEEP_LAST_N_RUNS=5

## Pre-deploy checklist
- [ ] alembic upgrade head đã chạy
      (tự động qua lifespan)
- [ ] MinIO buckets đã tạo
      (tự động qua minio-init service)
- [ ] DagsHub experiment đã tạo
- [ ] R2 bucket đã tạo và có CORS policy
- [ ] Supabase project đã tạo

## Post-deploy verification
- [ ] GET /health → status: healthy
- [ ] GET /api/v1/monitoring/health
      → tất cả services: up
- [ ] POST /api/v1/datasets/upload
      → upload 1 file nhỏ thành công
- [ ] /docs không accessible
      (trả về 404)
- [ ] Rate limiting hoạt động
      (gọi nhiều lần → 429)

## Render.com specific
- [ ] Set health check path: /health
- [ ] Set health check timeout: 30s
- [ ] Set instance type: Starter ($7/mo)
      (free tier sleep sau 15 phút — không phù hợp production)
- [ ] Set auto-deploy: on push to main

## Vercel specific (Frontend)
- [ ] NEXT_PUBLIC_API_URL=https://...
- [ ] NEXT_PUBLIC_API_KEY=...
- [ ] Framework Preset: Next.js
- [ ] Root Directory: frontend/
