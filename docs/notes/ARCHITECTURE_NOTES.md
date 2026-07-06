---

## Luồng dữ liệu và mô hình — Từ code thực tế

### Khi người dùng upload file

```
POST /api/v1/datasets/upload
         ↓
1. Validate: chỉ .csv hoặc .parquet, < 50MB
2. Tạo dataset_id = UUID ngẫu nhiên (vd: "a3f7-...")
         ↓
┌─────────────────────────────────────────────┐
│  FILE GỐC lưu lên Cloudflare R2:            │
│                                             │
│  Đường dẫn: raw/{user_id}/{dataset_id}/     │
│             telco_customers.csv             │
│                                             │
│  VD thực tế:                                │
│  s3://cdp-datalake-assets/                  │
│    raw/default_user/a3f7-.../               │
│    telco_customers.csv                      │
└─────────────────────────────────────────────┘
         ↓
3. Lưu metadata vào Supabase (PostgreSQL):
   bảng "datasets":
   ┌──────────────┬────────────────────┐
   │ id           │ "a3f7-..."         │
   │ user_id      │ "default_user"     │
   │ filename     │ "telco.csv"        │
   │ r2_path      │ "raw/user/.../..." │  ← địa chỉ file trên R2
   │ status       │ "uploaded"         │
   └──────────────┴────────────────────┘
```

---

### Khi profiling xong

```
POST /api/v1/datasets/{dataset_id}/profile
         ↓
1. Tải file về từ R2 (dùng r2_path trong DB)
2. Chạy run_profiling(df)
         ↓
Lưu kết quả vào Supabase:
bảng "profiles":
┌──────────────────┬──────────────────────────────┐
│ dataset_id       │ "a3f7-..."                   │
│ profiles_json    │ [{"name":"tenure","role":...}]│  ← JSON toàn bộ ColumnProfile
│ suggested_target │ "Churn"                      │
└──────────────────┴──────────────────────────────┘
```

---

### Khi training xong — Quan trọng nhất

```
POST /api/v1/jobs/datasets/{dataset_id}/train
         ↓
1. Tải file từ R2
2. Chạy AutoML (Optuna optimization)
         ↓
┌────────────────────────────────────────────────────────────┐
│  MÔ HÌNH được lưu ở 2 nơi:                                │
│                                                            │
│  1. DagsHub MLflow (artifact store):                       │
│     - File .pkl của Pipeline (preprocessor + model)        │
│     - threshold.json (ngưỡng tối ưu)                       │
│     - input_example (5 dòng mẫu để validate)              │
│     URI: runs:/603bbb27.../model                           │
│                                                            │
│  2. MLflow Model Registry (tên model):                     │
│     TelcoChurnModel                                        │
│     ├── Version 1 (AUC: 0.81) ← model cũ                  │
│     ├── Version 2 (AUC: 0.83) ← model mới hơn             │
│     └── Version 3 (AUC: 0.85) ← production hiện tại       │
└────────────────────────────────────────────────────────────┘
         ↓
3. Lưu model_uri vào Supabase:
bảng "training_jobs":
┌───────────┬──────────────────────────────┐
│ job_id    │ "b9c2-..."                   │
│ status    │ "completed"                  │
│ model_uri │ "runs:/603b.../model"        │  ← địa chỉ để load lại
│ roc_auc   │ 0.851                        │
└───────────┴──────────────────────────────┘
```

---

## Sơ đồ tổng thể

```
┌──────────┐   upload    ┌──────────────────────────────────────┐
│  Client  │ ──────────→ │        Cloudflare R2                 │
│  (User)  │             │  s3://cdp-datalake-assets/           │
└──────────┘             │  └── raw/user_id/dataset_id/file.csv │
                         └──────────────────────────────────────┘
     │                                     ▲
     │ profile/train                       │ download khi cần
     ▼                                     │
┌──────────────────────┐                   │
│   FastAPI Backend    │ ──────────────────┘
│  (churn_platform_api)│
└──────────────────────┘
     │                      ┌────────────────────────────────────┐
     │ lưu metadata         │     Supabase (PostgreSQL)          │
     ├────────────────────→ │  - datasets (file info + r2_path)  │
     │                      │  - profiles  (column analysis JSON) │
     │                      │  - training_jobs (model_uri, AUC)  │
     │                      └────────────────────────────────────┘
     │
     │ log model            ┌────────────────────────────────────┐
     └────────────────────→ │     DagsHub MLflow                 │
                            │  - Experiment runs (params, AUC)   │
                            │  - Model artifacts (.pkl files)    │
                            │  - Model Registry (versioning)     │
                            └────────────────────────────────────┘
```

---

## Tóm tắt ngắn gọn

| Loại dữ liệu | Lưu ở đâu | Mục đích |
|---|---|---|
| **File gốc** (CSV, Parquet) | Cloudflare R2 | Không tốn RAM server, tải về khi cần train |
| **Metadata** (tên file, path, status) | Supabase PostgreSQL | Tra cứu nhanh, quản lý nhiều user |
| **Kết quả profiling** (ColumnProfile JSON) | Supabase PostgreSQL | Hiển thị lại cho user mà không cần đọc lại file |
| **Model file** (.pkl, preprocessor) | DagsHub MLflow Artifacts | Versioning, load về để serving |
| **Model version + URI** | Supabase PostgreSQL | API biết model nào đang dùng cho dataset nào |
