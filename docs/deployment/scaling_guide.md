# Scaling Guide

## Single Instance (default)
ENABLE_DRIFT_SCHEDULER=true
→ Drift check chạy bình thường

## Multiple Instances
Instance 1: ENABLE_DRIFT_SCHEDULER=true
Instance 2: ENABLE_DRIFT_SCHEDULER=false
Instance 3: ENABLE_DRIFT_SCHEDULER=false

Lý do: Drift scheduler dùng FastAPI BackgroundTasks — không có distributed coordination. Chỉ 1 instance được phép chạy scheduler để tránh:
  - Double auto-retrain jobs
  - Race condition khi update drift_reports

## Future improvement
Khi cần scale thật sự:
- Dùng Celery + Redis cho distributed jobs
- Hoặc dùng APScheduler với database lock
