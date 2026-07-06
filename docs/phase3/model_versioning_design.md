# Model Versioning Design — Phase 3

> Cập nhật: 2026-07-06
> Trạng thái: Design — chưa implement

---

## 1. Mục tiêu

Quản lý nhiều version model cho cùng 1 dataset, cho phép:

- **List** tất cả completed models của một dataset
- **Compare** 2 models side-by-side (metrics, threshold, thời gian train)
- **Promote** model tốt nhất lên trạng thái active — predict endpoint tự động dùng model đó

Hiện tại predict endpoint load model có `roc_auc` cao nhất (`ORDER BY roc_auc DESC LIMIT 1`). Sau khi có versioning, logic sẽ ưu tiên model có `is_active=True`.

---

## 2. Endpoints cần implement

### `GET /api/v1/models/{dataset_id}`

Trả về tất cả completed jobs cho `dataset_id`, sort by `created_at DESC`.

**Response schema:**

```json
{
  "dataset_id": "abc-123",
  "models": [
    {
      "job_id": "job-001",
      "model_uri": "runs:/abc/model",
      "target_col": "Churn",
      "roc_auc": 0.892,
      "optimal_threshold": 0.42,
      "created_at": "2026-07-05T14:30:00Z",
      "finished_at": "2026-07-05T14:33:12Z",
      "status": "completed",
      "is_active": true,
      "tags": {"experiment": "baseline"}
    },
    {
      "job_id": "job-002",
      "model_uri": "runs:/def/model",
      "target_col": "Churn",
      "roc_auc": 0.871,
      "optimal_threshold": 0.45,
      "created_at": "2026-07-04T10:00:00Z",
      "finished_at": "2026-07-04T10:02:45Z",
      "status": "completed",
      "is_active": false,
      "tags": {}
    }
  ]
}
```

**Query:**

```sql
SELECT * FROM training_jobs
WHERE dataset_id = :dataset_id
  AND status = 'completed'
ORDER BY started_at DESC;
```

> [!NOTE]
> Field `model_class` chưa có trong `training_jobs` table hiện tại.
> Nếu cần group by model class (XGBoost vs RF vs MTL), cần thêm field `model_class` vào schema.
> Cho MVP: bỏ qua field này, chỉ hiển thị tất cả models trong danh sách phẳng.

---

### `POST /api/v1/models/{dataset_id}/promote`

Set một model cụ thể làm active model cho dataset.

**Request body:**

```json
{
  "job_id": "job-001"
}
```

**Logic:**

1. Verify `job_id` thuộc về `dataset_id` và có `status = 'completed'`
2. Set `is_active = False` cho **tất cả** jobs khác của cùng `dataset_id`
3. Set `is_active = True` cho `job_id` được chỉ định
4. Trả về kết quả promotion

**Response schema:**

```json
{
  "promoted_job_id": "job-001",
  "previous_active": "job-003",
  "roc_auc": 0.892,
  "optimal_threshold": 0.42
}
```

`previous_active` = `null` nếu trước đó chưa có model nào được promote.

**Error cases:**

| Điều kiện | HTTP Status | Detail |
|-----------|-------------|--------|
| `job_id` không tồn tại | 404 | `"Job not found"` |
| `job_id` không thuộc `dataset_id` | 400 | `"Job does not belong to this dataset"` |
| `job.status != 'completed'` | 400 | `"Cannot promote a non-completed job"` |

**SQL (transaction):**

```sql
BEGIN;
  -- Tìm active model hiện tại
  SELECT id FROM training_jobs
  WHERE dataset_id = :dataset_id AND is_active = TRUE;

  -- Deactivate tất cả
  UPDATE training_jobs
  SET is_active = FALSE
  WHERE dataset_id = :dataset_id;

  -- Activate model được chọn
  UPDATE training_jobs
  SET is_active = TRUE
  WHERE id = :job_id;
COMMIT;
```

---

### `GET /api/v1/models/{dataset_id}/compare`

So sánh side-by-side 2 models.

**Query params:** `?job_id_a=X&job_id_b=Y`

**Response schema:**

```json
{
  "model_a": {
    "job_id": "job-001",
    "roc_auc": 0.892,
    "optimal_threshold": 0.42,
    "target_col": "Churn",
    "created_at": "2026-07-05T14:30:00Z",
    "is_active": true
  },
  "model_b": {
    "job_id": "job-002",
    "roc_auc": 0.871,
    "optimal_threshold": 0.45,
    "target_col": "Churn",
    "created_at": "2026-07-04T10:00:00Z",
    "is_active": false
  },
  "winner": "a",
  "delta_roc_auc": 0.021
}
```

**Logic xác định `winner`:**

| Điều kiện | Winner |
|-----------|--------|
| `abs(roc_auc_a - roc_auc_b) < 0.005` | `"tie"` |
| `roc_auc_a > roc_auc_b` | `"a"` |
| `roc_auc_b > roc_auc_a` | `"b"` |

`delta_roc_auc = abs(roc_auc_a - roc_auc_b)`, luôn dương.

**Error cases:**

| Điều kiện | HTTP Status | Detail |
|-----------|-------------|--------|
| Thiếu `job_id_a` hoặc `job_id_b` | 422 | `"Both job_id_a and job_id_b are required"` |
| `job_id_a == job_id_b` | 400 | `"Cannot compare a model with itself"` |
| Job không thuộc `dataset_id` | 400 | `"Job does not belong to this dataset"` |

---

## 3. Thay đổi DB schema

### Thêm fields vào `training_jobs` table

```python
# backend/app/db/models.py — class TrainingJob

# Thêm 2 columns:
is_active = Column(Boolean, default=False, nullable=False)
tags = Column(JSON, default=dict, nullable=True)
```

| Field | Type | Default | Mô tả |
|-------|------|---------|-------|
| `is_active` | `Boolean` | `False` | Chỉ 1 job per dataset có thể `True` |
| `tags` | `JSON` | `{}` | Key-value metadata tự do (experiment name, notes) |

### Alembic migration

Cần tạo migration cho 2 fields mới:

```
alembic revision --autogenerate -m "add is_active and tags to training_jobs"
alembic upgrade head
```

> [!WARNING]
> Hiện tại project dùng `Base.metadata.create_all()` trong `main.py` startup thay vì Alembic migrations.
> Nếu chưa setup Alembic, cần đánh giá xem có nên thêm Alembic hay chỉ dùng `create_all()` (sẽ tự thêm column mới nếu table chưa tồn tại, nhưng **không ALTER existing tables**).
> Giải pháp tạm: chạy raw SQL `ALTER TABLE training_jobs ADD COLUMN is_active BOOLEAN DEFAULT FALSE` trên production DB.

---

## 4. Tác động đến predict endpoint

### Thay đổi logic load model

**File:** `backend/app/api/v1/predict.py`

**Hiện tại** (sort by roc_auc):

```python
job = db.query(TrainingJob)\
    .filter_by(dataset_id=dataset_id, status="completed")\
    .order_by(TrainingJob.roc_auc.desc())\
    .first()
```

**Sau khi có versioning** (ưu tiên is_active):

```python
# 1. Ưu tiên: model có is_active=True
job = db.query(TrainingJob)\
    .filter_by(dataset_id=dataset_id, status="completed", is_active=True)\
    .first()

# 2. Fallback: nếu chưa có model nào được promote, dùng roc_auc cao nhất
if not job:
    job = db.query(TrainingJob)\
        .filter_by(dataset_id=dataset_id, status="completed")\
        .order_by(TrainingJob.roc_auc.desc())\
        .first()
```

Logic fallback đảm bảo backward compatibility — hệ thống vẫn hoạt động bình thường khi chưa có model nào được promote.

### Ảnh hưởng đến batch predict

Áp dụng cùng logic cho endpoint `POST /api/v1/predict/batch`:
- `get_optimal_threshold()` trong `predict.py` cũng cần ưu tiên model `is_active=True`

---

## 5. Router registration

- Tạo file `backend/app/api/v1/models.py`
- Mount vào `main.py`:

```python
from .api.v1 import models
app.include_router(models.router, prefix="/api/v1", dependencies=[Depends(get_api_key)])
```

### Dependency map

```
models.py
├── db/session.py      (SQLAlchemy session)
├── db/models.py       (TrainingJob — is_active, tags)
└── api/schemas.py     (ModelListResponse, PromoteRequest, CompareResponse)
```
