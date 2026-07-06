# Monitoring Design — Phase 3

> Cập nhật: 2026-07-06
> Trạng thái: Design — chưa implement

---

## 1. Mục tiêu

API monitoring cung cấp visibility về:

- **Health** của từng service (backend, database, storage, MLflow)
- **Performance metrics** của predict endpoint
- **Job queue status** (queued / training / completed / failed)
- **Storage usage** (Cloudflare R2 hoặc MinIO local)

Monitoring endpoints **không yêu cầu external tools** (Prometheus, Grafana) — chỉ dùng SQLAlchemy queries đơn giản và boto3/HTTP ping cho MVP.

---

## 2. Endpoints cần implement

### `GET /api/v1/monitoring/health`

Kiểm tra connectivity đến từng dependency, trả về overall health status.

**Response schema:**

```json
{
  "status": "healthy | degraded | unhealthy",
  "timestamp": "2026-07-06T09:30:00.000Z",
  "services": {
    "database": {
      "status": "up | down",
      "latency_ms": 12.5
    },
    "storage": {
      "status": "up | down",
      "latency_ms": 45.2
    },
    "mlflow": {
      "status": "up | down",
      "latency_ms": 120.8
    }
  }
}
```

**Logic xác định overall status:**

| Điều kiện | Overall status |
|-----------|---------------|
| Tất cả services `up` | `healthy` |
| Có service `down` nhưng `database` vẫn `up` | `degraded` |
| `database` bị `down` | `unhealthy` |

**Implementation notes:**
- Ping database: `SELECT 1` qua SQLAlchemy
- Ping storage: `s3.head_bucket(Bucket=S3_BUCKET_NAME)` qua boto3
- Ping MLflow: `GET {MLFLOW_TRACKING_URI}/health` qua `urllib.request`
- Mỗi service ping có **timeout cứng 3 giây** — nếu vượt timeout → `down`
- Dùng `asyncio.gather()` hoặc `concurrent.futures.ThreadPoolExecutor` để ping song song

---

### `GET /api/v1/monitoring/metrics`

Trả về performance metrics tổng hợp trong 24 giờ gần nhất.

**Response schema:**

```json
{
  "predict_endpoint": {
    "total_calls_24h": 1523,
    "avg_latency_ms": 45.2,
    "p95_latency_ms": 120.8,
    "error_rate_pct": 0.5
  },
  "jobs": {
    "queued": 2,
    "training": 1,
    "completed_24h": 15,
    "failed_24h": 1
  },
  "storage": {
    "total_files": 342,
    "total_size_mb": 1024.5
  }
}
```

**Data sources:**

| Metric group | Source | Query method |
|-------------|--------|-------------|
| `predict_endpoint` | `predict_log` table (nếu tồn tại) | SQLAlchemy COUNT/AVG |
| `jobs` | `training_jobs` table | SQLAlchemy COUNT + GROUP BY status |
| `storage` | boto3 `list_objects_v2` | Paginate và sum `ContentLength` |

> [!IMPORTANT]
> Hiện tại **chưa có `predict_log` table** trong DB schema (`backend/app/db/models.py` chỉ có 3 tables: `datasets`, `profiles`, `training_jobs`).
> Khi implement, nếu table chưa tồn tại thì trả về `predict_endpoint: null` kèm note `"partial_metrics": true`.
> Việc tạo `predict_log` table nên được xử lý trong một task riêng trước khi implement metrics endpoint.

**Job status query:**

```sql
-- queued + training: không filter thời gian
SELECT status, COUNT(*) FROM training_jobs
WHERE status IN ('queued', 'training')
GROUP BY status;

-- completed + failed: chỉ 24h gần nhất
SELECT status, COUNT(*) FROM training_jobs
WHERE status IN ('completed', 'failed')
  AND finished_at >= NOW() - INTERVAL '24 hours'
GROUP BY status;
```

---

### `GET /api/v1/monitoring/jobs/summary`

Tóm tắt tất cả jobs trong **7 ngày** gần nhất, group by status.

**Response schema:**

```json
{
  "period_days": 7,
  "total_jobs": 42,
  "by_status": {
    "completed": 35,
    "failed": 4,
    "training": 2,
    "queued": 1
  },
  "avg_training_duration_minutes": 3.2,
  "jobs": [
    {
      "job_id": "abc-123",
      "dataset_id": "def-456",
      "status": "completed",
      "target_column": "Churn",
      "roc_auc": 0.87,
      "started_at": "2026-07-01T10:00:00Z",
      "finished_at": "2026-07-01T10:03:12Z",
      "duration_minutes": 3.2
    }
  ]
}
```

**Query:**

```sql
SELECT * FROM training_jobs
WHERE started_at >= NOW() - INTERVAL '7 days'
ORDER BY started_at DESC;
```

`duration_minutes` tính bằng `(finished_at - started_at).total_seconds() / 60` — `null` nếu job chưa finish.

---

## 3. Implementation notes

### Timeout và performance
- Health check **phải có timeout cứng 3s/service** để không block response
- Nếu 1 service timeout → trả về `"status": "down", "latency_ms": 3000`
- Health endpoint tổng thể phải respond trong **< 5 giây** worst case

### Metrics tính từ DB
- Metrics tính từ `training_jobs` table (đã có) và `predict_log` table (chưa có)
- Nếu `predict_log` chưa tồn tại → ghi rõ `"partial_metrics": true` trong response
- **Không dùng external monitoring tool** — chỉ dùng SQLAlchemy queries đơn giản

### Router registration
- Tạo file `backend/app/api/v1/monitoring.py`
- Mount vào `main.py` với prefix `/api/v1` và dependency `[Depends(get_api_key)]`
- Monitoring endpoints **yêu cầu auth** giống các endpoint khác

### Dependency map

```
monitoring.py
├── db/session.py      (SQLAlchemy session)
├── db/models.py       (TrainingJob model)
├── core/storage.py    (StorageClient — list_files)
└── core/config.py     (MLFLOW_TRACKING_URI)
```

---

## 4. Tác động đến Phase 4

Frontend sẽ dùng 3 endpoints này để hiển thị:

| Endpoint | UI Component |
|----------|-------------|
| `/monitoring/health` | Status bar trên navigation — xanh/vàng/đỏ |
| `/monitoring/metrics` | Metrics dashboard — cards và charts |
| `/monitoring/jobs/summary` | Jobs timeline — bảng + filter by status |

> [!NOTE]
> Frontend cần poll `/monitoring/health` mỗi 30 giây để cập nhật status bar.
> `/monitoring/metrics` và `/monitoring/jobs/summary` chỉ cần fetch khi user navigate đến trang Monitoring.
