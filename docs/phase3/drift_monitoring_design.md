# Drift Monitoring Design — Phase 3

> Cập nhật: 2026-07-06
> Trạng thái: Partially implemented — logic có, automation chưa có

---

## 1. Mục tiêu

Phát hiện data drift giữa training data (reference) và production data (target) để:

- Xác định khi nào model cần retrain
- Cảnh báo khi distribution của features thay đổi đáng kể
- Cung cấp chi tiết feature-level drift metrics cho investigation

### Trạng thái hiện tại

| Component | File | Trạng thái |
|-----------|------|-----------|
| PSI calculation (numerical) | `core/serving/drift_detector.py` L6-45 | ✅ Implemented |
| PSI calculation (categorical) | `core/serving/drift_detector.py` L47-86 | ✅ Implemented |
| KS test (numerical) | `core/serving/drift_detector.py` L110-116 | ✅ Implemented |
| `calculate_drift_report()` | `core/serving/drift_detector.py` L88-165 | ✅ Implemented |
| API endpoint | `api/v1/predict.py` — `POST /datasets/{id}/drift` | ✅ Implemented |
| API schema | `api/schemas.py` — `DriftRequest`, `DriftResponse` | ✅ Implemented |
| Automated scheduling | — | ❌ Chưa có |
| Alerting (email/webhook) | — | ❌ Chưa có |
| Drift history / trend tracking | — | ❌ Chưa có |

---

## 2. Endpoints

### `POST /api/v1/predict/datasets/{dataset_id}/drift` (đã có)

**Request body:**

```json
{
  "target_file_path": "landing_zone/abc-123/new_data.csv"
}
```

**Response schema** (đã implement trong `api/schemas.py`):

```json
{
  "dataset_id": "abc-123",
  "reference_rows": 7032,
  "target_rows": 1500,
  "drift_detected": true,
  "metrics": {
    "tenure": {
      "type": "numeric",
      "ks_statistic": 0.15,
      "ks_p_value": 0.002,
      "psi": 0.23,
      "drift_level": "high",
      "is_drifted": true
    },
    "Contract": {
      "type": "categorical",
      "ks_statistic": null,
      "ks_p_value": null,
      "psi": 0.08,
      "drift_level": "low",
      "is_drifted": false
    }
  }
}
```

### Logic hiện tại trong `drift_detector.py`

Reference data = file gốc đã dùng để train model (lấy từ `datasets.r2_path` trong DB).
Target data = file mới do user upload qua `target_file_path` trong request body.

Chỉ so sánh các feature columns (exclude target column và ID columns) dựa trên `ColumnProfile.inferred_role`.

---

## 3. Thresholds và interpretation

### Per-feature drift classification

| Metric | Điều kiện | Drift Level |
|--------|-----------|-------------|
| PSI | `< 0.1` | `low` — no significant drift |
| PSI | `0.1 – 0.2` | `medium` — moderate drift, monitor |
| PSI | `≥ 0.2` | `high` — significant drift, action needed |
| KS p-value | `< 0.05` | Distribution change detected (numerical only) |

### Per-feature `is_drifted` flag

```python
# Numerical features
is_drifted = (psi >= 0.2) or (ks_p_value < 0.05)

# Categorical features
is_drifted = (psi >= 0.2)
```

> [!NOTE]
> Logic này **đã được implement** trong `drift_detector.py` L122-157.

### Overall drift status (đề xuất bổ sung)

Hiện tại response chỉ có `"drift_detected": true/false` (binary). Đề xuất thêm field `"overall_drift_level"` với logic:

| Điều kiện | Overall Level |
|-----------|--------------|
| Tất cả features PSI < 0.1 | `"no_drift"` |
| Có feature PSI 0.1–0.2, nhưng không có PSI ≥ 0.2 | `"moderate"` |
| Có feature PSI ≥ 0.2 | `"high"` |
| > 30% features bị `is_drifted = True` | `"critical"` |

**Updated response (đề xuất):**

```json
{
  "dataset_id": "abc-123",
  "reference_rows": 7032,
  "target_rows": 1500,
  "drift_detected": true,
  "overall_drift_level": "high",
  "drifted_feature_count": 3,
  "total_feature_count": 18,
  "drifted_feature_pct": 16.7,
  "metrics": { "..." }
}
```

---

## 4. Dependency với Model Versioning

### Vấn đề: Reference data cần match với active model

Khi có nhiều models cho cùng 1 dataset (model versioning), drift report cần biết **model nào đang active** để lấy đúng training dataset tương ứng làm reference distribution.

**Flow:**

```
POST /datasets/{id}/drift
  │
  ├── 1. Query: model nào is_active=True cho dataset_id?
  │       └── Fallback: model có roc_auc cao nhất (như predict endpoint)
  │
  ├── 2. Lấy training data path từ job tương ứng
  │       └── datasets.r2_path (file gốc đã dùng để train)
  │
  ├── 3. Load reference_df từ r2_path
  │
  ├── 4. Load target_df từ request.target_file_path
  │
  └── 5. calculate_drift_report(reference_df, target_df, ...)
```

> [!IMPORTANT]
> **Drift Monitoring phụ thuộc vào Model Versioning đã implement xong** (cụ thể: field `is_active` trên `TrainingJob` model).
> Nếu implement trước khi có model versioning, dùng fallback logic: lấy model có `roc_auc` cao nhất.

### Trường hợp đặc biệt

| Tình huống | Xử lý |
|-----------|-------|
| Dataset chưa có completed model | 400 — `"No trained model found for this dataset"` |
| Reference file bị xóa khỏi R2 | 404 — `"Reference data file not found in storage"` |
| Target file format khác reference | 400 — `"Column mismatch between reference and target data"` |

---

## 5. Tác động đến Phase 4

### Frontend hiển thị

Frontend sẽ hiển thị drift report dưới dạng:

| UI Component | Data source |
|-------------|-------------|
| **Heatmap** — feature nào drift nhiều nhất | `metrics[feature].psi` — color scale: green (< 0.1) → yellow (0.1–0.2) → red (≥ 0.2) |
| **Summary card** — overall drift level | `overall_drift_level` — badge xanh/vàng/đỏ |
| **Feature table** — sortable by PSI | `metrics` object — table với columns: Feature, Type, PSI, KS p-value, Drift Level |
| **Action button** — trigger retrain | Hiển thị khi `overall_drift_level` = `"high"` hoặc `"critical"` |

### User flow

```
ModelHub page
  └── Chọn dataset
      └── Tab "Drift Monitoring"
          ├── Upload new data file
          ├── Click "Run Drift Check"
          ├── Xem heatmap + summary
          └── Nếu drift cao → "Retrain Model" button
              └── POST /jobs/datasets/{id}/train
```

---

## 6. Future: Automated drift monitoring (Phase 5+)

Scope của Phase 3 là **manual drift check** (user chủ động gọi API). Automated monitoring cần:

- **Scheduler** (cron job hoặc Celery beat) chạy drift check định kỳ
- **Alerting** (email, Slack webhook) khi `overall_drift_level >= "high"`
- **Auto-retrain trigger** khi `overall_drift_level == "critical"`
- **Drift history table** lưu kết quả drift check theo thời gian để track trend

> [!NOTE]
> Automated drift monitoring là scope của Phase 5 (post-MVP). Phase 3 chỉ cần manual endpoint hoạt động đúng và có overall_drift_level interpretation rõ ràng.
