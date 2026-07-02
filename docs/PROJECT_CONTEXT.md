# PROJECT_CONTEXT.md

> **Mục đích**: Onboard AI assistant mới vào đúng context của dự án.
> **Cập nhật lần cuối**: 2026-06-27 — sau khi fix Bug 1 & Bug 2.

---

## Mục 1: Tổng quan dự án

**Churn Prediction Platform** là hệ thống ML end-to-end cho phép người dùng upload dataset dạng bảng bất kỳ (CSV, Parquet), nhận phân tích tự động về cột dữ liệu, huấn luyện mô hình churn prediction, và gọi API để dự đoán realtime. Target user là Data Scientist hoặc Data Analyst muốn deploy mô hình churn mà không cần viết ML pipeline từ đầu — hệ thống tự động xác định target column, xử lý feature, và chọn thuật toán tốt nhất. Tech stack: **FastAPI + PostgreSQL (Supabase) + Cloudflare R2 + MLflow (DagsHub) + Streamlit + Docker**.

---

## Mục 2: Kiến trúc thực tế hiện tại

### `backend/app/api/v1/` — REST API Layer

| File          | Endpoints                                                 | Trạng thái       |
| ------------- | --------------------------------------------------------- | ---------------- |
| `datasets.py` | `POST /datasets/upload`, `POST /datasets/{id}/profile`    | Production-ready |
| `jobs.py`     | `POST /jobs/datasets/{id}/train`, `GET /jobs/{id}/status` | Production-ready |
| `predict.py`  | `POST /predict`                                           | Production-ready |

Auth toàn cục qua `X-API-Key` header, cấu hình trong `main.py`.

### `backend/app/core/profiler/` — Profiling Engine

| File                    | Chức năng                                                  | Trạng thái                                     |
| ----------------------- | ---------------------------------------------------------- | ---------------------------------------------- |
| `orchestrator.py`       | Điều phối toàn bộ flow — gọi 3 layer + synthesizer         | Production-ready                               |
| `layer1_stats.py`       | Tính dtype, null_pct, unique_count, entropy                | Production-ready                               |
| `layer2_semantic.py`    | Phát hiện ID/EMAIL/PHONE/DATE qua regex                    | Production-ready                               |
| `layer3_llm.py`         | Refine role qua LLM (Groq API)                             | ⚠️ Optional — skip nếu không có `GROQ_API_KEY` |
| `target_analysis.py`    | Pydantic models: `TargetAnalysis`, `CompositeTargetConfig` | Production-ready                               |
| `target_synthesizer.py` | Tổng hợp CPI từ auxiliary columns (PCA hoặc WEIGHTED)      | Production-ready                               |
| `column_profile.py`     | Pydantic model `ColumnProfile`                             | Production-ready                               |

### `backend/app/core/ingestion/` — Ingestion Layer

| File         | Chức năng                                                        | Trạng thái       |
| ------------ | ---------------------------------------------------------------- | ---------------- |
| `parsers.py` | Hàm `parse_file()` phân tích CSV, TSV, Parquet, JSON, ODS, Excel | Production-ready |

### `backend/app/core/training/` — Training Engine

| File                   | Chức năng                                                  | Trạng thái                        |
| ---------------------- | ---------------------------------------------------------- | --------------------------------- |
| `automl.py`            | Entry point training — route MTL hoặc Standard AutoML      | Production-ready                  |
| `model_router.py`      | Chọn model candidates dựa trên dataset characteristics     | Production-ready                  |
| `mtl_trainer.py`       | PyTorch MTL model (2 heads: churn binary + CPI regression) | Production-ready (torch optional) |
| `train.py`             | Legacy script — hardcode Telco features                    | Stub/Legacy                       |
| `train_mlflow.py`      | Legacy — MLflow logging script riêng                       | Stub/Legacy                       |
| `train_using_spark.py` | Legacy — Spark training                                    | Stub/Legacy                       |
| `mlflow_utils.py`      | Setup MLflow tracking URI                                  | Production-ready                  |

### `backend/app/core/pipeline/` — Feature Engineering

| File            | Chức năng                                                 | Trạng thái       |
| --------------- | --------------------------------------------------------- | ---------------- |
| `builder.py`    | `build_pipeline(profiles, target_col)` → sklearn Pipeline | Production-ready |
| `transforms/`   | Registry imputers và transformers theo strategy name      | Production-ready |
| `schema_gen.py` | Tự sinh Pandera schema từ confirmed profiles              | Production-ready |

### `backend/app/core/serving/` — Model Serving

| File              | Chức năng                                               | Trạng thái       |
| ----------------- | ------------------------------------------------------- | ---------------- |
| `model_loader.py` | `ModelCache` — load từ MLflow, TTL 10 phút, thread-safe | Production-ready |
| `ab_service.py`   | A/B testing service độc lập (FastAPI app riêng)         | In progress      |

### `backend/app/db/` — Database Layer

| File                                                            | Trạng thái       |
| --------------------------------------------------------------- | ---------------- |
| `models.py` — 3 tables: `datasets`, `profiles`, `training_jobs` | Production-ready |
| `session.py` — SQLAlchemy session, đọc `DATABASE_URL` từ env    | Production-ready |

### `analytics/streamlit_app.py` — Dashboard

Data Profiling đọc từ API thực tế. Model Explainability và Cohort Analysis là placeholder. `row_count` hardcode `"10,240"`. Trạng thái: **In progress**.

### `scripts/` — Utilities

`validation/validate_database.py`, `validation/validate_storage.py`, `validation/validate_mlflow.py` — kiểm tra remote connections. `validation/init_db.py` — khởi tạo DB tables. `ab_testing/ab_assign.py`, `ab_testing/simulate_campaign.py` — A/B testing utilities.

### `frontend/` — React Frontend

**Stub** — chỉ có `package.json`, `src/` rỗng. Chưa có code.

### `backend/app/core/etl/` — ETL Pipeline

7 files (cleaning, incremental, lineage, observability, partitioning...). ⚠️ Chưa verify — không tích hợp vào main API flow.

---

## Mục 3: Các object/contract trung tâm

### `ColumnProfile` — `column_profile.py`

```python
class DataRole(str, Enum):
    ID | TARGET | NUMERIC | CATEGORICAL | DATETIME | TEXT | IGNORE

class ColumnProfile(BaseModel):
    name: str
    inferred_dtype: str
    inferred_role: DataRole
    confidence_score: float          # [0.0, 1.0]
    null_pct: float                  # [0.0, 1.0]
    unique_count: int
    entropy: float
    mean_length: Optional[float]     # chỉ TEXT
    regex_pattern: Optional[str]     # email/phone/date
    potential_leakage: bool
    leakage_score: Optional[float]
    transform_strategy: Optional[str]  # "standard"|"ohe"|"tfidf"|"label"|"date_parts"
    impute_strategy: Optional[str]     # "median"|"mode"|"constant"|"drop"|"drop_row"
```

### `TargetAnalysis` — `target_analysis.py`

```python
class TargetAnalysis(BaseModel):
    recommended_target: str
    candidate_targets: List[CandidateTarget]       # top 3, ranked by score
    churn_column_group: List[ChurnColumnGroupItem] # PRIMARY/AUXILIARY/DUPLICATE/LEAKAGE_SUSPECT
    recommended_auxiliary: List[str]               # convenience list
    leakage_suspects: List[str]                    # convenience list
    composite_target: Optional[CompositeTargetConfig]  # None nếu strategy=NONE
```

### `CompositeTargetConfig` — `target_analysis.py`

```python
class SynthesisStrategy(str, Enum):
    PCA | WEIGHTED | NONE

class CompositeTargetConfig(BaseModel):
    strategy: SynthesisStrategy
    source_columns: List[str]
    cpi_variance_explained: Optional[float]   # chỉ khi strategy=PCA
    weights: Optional[List[ColumnWeight]]      # chỉ khi strategy=WEIGHTED
    cpi_column_name: str = "cpi_score"
    requires_confirmation: bool
```

### API Schemas — `api/schemas.py`

```python
# Requests
TrainingRequest:
    confirmed_target: str
    confirmed_profiles: List[ColumnProfile]
    composite_config: Optional[CompositeTargetConfig] = None  # [Bug 1 fixed]

# Responses
DatasetResponse:    dataset_id, row_count, col_count, status
ProfilingResponse:  dataset_id, profiles, suggested_target, warnings,
                    composite_target: Optional[CompositeTargetConfig]  # [Bug 2 fixed]
JobResponse:        job_id, status, estimated_minutes
JobStatusResponse:  job_id, status, roc_auc, model_uri, finished_at
PredictionResult:   record_id, churn_probability, risk_level  # "High"/"Medium"/"Low"
```

---

## Mục 4: Luồng dữ liệu thực tế

```text
[Client] POST /api/v1/datasets/upload
    → validate .csv / .parquet, < 50MB
    → upload raw file lên Cloudflare R2: raw/{user_id}/{dataset_id}/{filename}
    → ghi metadata vào DB: bảng datasets
    → trả về: dataset_id

[Client] POST /api/v1/datasets/{id}/profile
    → load file từ R2
    → run_profiling(df):
        Layer 1: dtype/entropy/null stats
        detect_target(): entropy scoring → recommended_target
        check_leakage(): đánh dấu corr > 0.95
        synthesize_target(): CPI nếu có auxiliary cols
        Layer 2: semantic detection (regex)
        Layer 3: LLM refinement (optional)
        assign ROLE_RECIPES (impute/transform strategy)
    → ghi profiles JSON vào DB
    → trả về: List[ColumnProfile] + suggested_target + composite_target  ✅ [Bug 2 fixed]

[Client] POST /api/v1/jobs/datasets/{id}/train
    → req.composite_config được nhận từ client  ✅ [Bug 1 fixed]
    → tạo TrainingJob record (status="training")
    → background_task(composite_config=req.composite_config):
        load file từ R2
        reconstruct ColumnProfile objects từ dict  ✅ [Bug 1 fixed]
        build_pipeline(profiles, target_col)
        IF composite_config.strategy != NONE AND torch available:
            → MTL path: MTLChurnModel.fit(X, y_binary, y_cpi)  ✅ [Bug 1 fixed]
        ELSE:
            → Standard AutoML: Optuna (XGBoost/RF/LogReg)
        → model_uri, schema_path = run_automl(...)
        → mlflow.sklearn.log_model() → DagsHub
        → update job: status="completed", model_uri  ✅ [Bug 1 fixed - roc_auc từ MLflow]
    → trả về: job_id

[Client] POST /api/v1/predict
    → query DB: lấy job với roc_auc cao nhất cho dataset_id
    → model_cache.get_model(model_uri) — load từ MLflow, cache 10 phút
    → model.predict_proba(input_df)[:, 1]
    → trả về: [{record_id, churn_probability, risk_level}]
```

> ✅ **GAP đã sửa**: `roc_auc` trong `training_jobs` table hiện tại được cập nhật tự động bằng cách truy vấn trực tiếp từ run của MLflow sau khi training hoàn tất.
>
> ✅ **GAP đã xác minh**: `model_loader` tương thích hoàn toàn nhờ lớp `MTLChurnModel` định nghĩa `__getstate__`/`__setstate__` chuyển đổi PyTorch state_dict thành bytes khi pickle.
>
> ✅ **GAP đã sửa**: Endpoint `POST /datasets/{id}/confirm-composite` đã được thêm để người dùng xác nhận cấu hình CPI và đính kèm cột chỉ số vào tập dữ liệu Parquet trên R2.
---

## Mục 5: Những gì được thêm ngoài roadmap gốc

| File/Module                  | Lý do thêm                                                                                 |
| ---------------------------- | ------------------------------------------------------------------------------------------ |
| `target_synthesizer.py`      | Tổng hợp CPI từ auxiliary columns (PCA/WEIGHTED) để cung cấp continuous label cho MTL      |
| `target_analysis.py`         | Pydantic models mở rộng: `TargetAnalysis`, `CompositeTargetConfig`, `ColumnWeight`         |
| `mtl_trainer.py`             | PyTorch Multi-Task Learning — train 2 objectives đồng thời (churn binary + CPI regression) |
| `model_router.py`            | Routing logic chọn model candidates dựa trên dataset sparsity và size                      |
| `serving/ab_service.py`      | A/B testing service độc lập — deterministic hash assignment                                |
| `services/exposure_store.py` | Log A/B exposure events vào DB hoặc JSONL                                                  |
| `scripts/validation/validate_*.py`      | Validation scripts cho DB/Storage/MLflow remote connections                                |
| `etl/` (7 files)             | ETL pipeline với Spark — cleaning, lineage, observability                                  |
| `pipeline/schema_gen.py`     | Tự sinh Pandera schema (thay Great Expectations)                                           |

---

## Mục 6: Trạng thái từng Phase

### Phase 0 — Foundation & Infrastructure

- [x] Docker Compose — `docker-compose.yml` — PostgreSQL + MinIO + MLflow + FastAPI + Streamlit
- [x] StorageClient — `backend/app/core/storage.py` — unified S3/local
- [x] MLflow setup — `backend/app/core/training/mlflow_utils.py`
- [x] DB models — `backend/app/db/models.py` — 3 tables
- [x] Supabase connection — Đã verify thành công kết nối Supabase, Cloudflare R2 và DagsHub

### Phase 1 — Data Contract & Profiling Engine

- [x] `ColumnProfile` model — `column_profile.py`
- [x] Layer 1 stats — `layer1_stats.py`
- [x] Layer 2 semantic — `layer2_semantic.py`
- [x] Layer 3 LLM — `layer3_llm.py` (optional)
- [x] `detect_target()` — `orchestrator.py`
- [x] `group_churn_cols()` — inline trong `orchestrator.py`
- [x] `TargetAnalysis` schema — `target_analysis.py`
- [x] `synthesize_target()` — `target_synthesizer.py`
- [x] `CompositeTargetConfig` schema — `target_analysis.py`
- [x] `build_pipeline()` — `pipeline/builder.py`
- [x] `schema_gen.py` — Pandera schema generation
- [x] `/profile` API trả về `composite_target` — `datasets.py` — **[Bug 2 fixed]**

### Phase 2 — Training Pipeline

- [x] `run_automl()` — `training/automl.py` — Optuna + StratifiedKFold
- [x] `model_router.py` — routing XGBoost/RF/LogReg
- [x] MTL path trong `automl.py`
- [x] `MTLChurnModel` — `mtl_trainer.py` — pickle-safe
- [x] MLflow logging — params, metrics, model registry
- [x] Optimal threshold — Precision-Recall curve
- [x] `composite_config` flow qua REST API — `jobs.py` — **[Bug 1 fixed]**
- [x] `ColumnProfile` reconstruct từ dict — `jobs.py` — **[Bug 1 fixed]**
- [x] job.roc_auc được cập nhật từ MLflow sau training — **[GAP fixed]**
- [x] Ablation Study — reproducible SEED=42, citation verified, committed

### Phase 3 — MLOps / Serving & BI

- [x] `POST /predict` (Single record inference) — `api/v1/predict.py`
- [x] `ModelCache` — `serving/model_loader.py`
- [x] A/B testing service — `serving/ab_service.py`
- [x] Streamlit dashboard — `analytics/streamlit_app.py` (partial)
- [x] Model drift monitoring — `drift_detector.py`
- [x] Automated retraining trigger / endpoint — `POST /jobs/datasets/{id}/train` (queued state & background worker)
- [x] `POST /datasets/{id}/confirm-composite` — **[GAP fixed]**
- [x] Universal file ingestion (Excel/JSON/TSV) — `parsers.py` **[Phase 3 check]**
- [x] `POST /datasets/{id}/select-sheet` — Xử lý chọn sheet cho Excel đa sheet
- [x] `POST /datasets/{id}/re-evaluate-leakage` — Re-evaluate target leakage
- [x] `POST /predict/batch` — Batch inference utilizing MLflow optimal threshold & dynamic risk levels

### Phase 4 — UI / Frontend

- [ ] React frontend — `frontend/src/` rỗng

---

## Mục 7: Các quyết định thiết kế đã chốt

- **Không dùng Great Expectations** — dùng Pandera tự sinh từ `confirmed_profiles` trong `schema_gen.py`
- **TEXT column**: impute `"constant"` (empty string), transform `"tfidf"`
- **Target detection**: entropy scoring liên tục (4 điểm), không threshold cứng từng feature
- **CPI auto-attach ngưỡng = 2**: `eligible_cols ≤ 2` → tự động, `≥ 3` → `requires_confirmation=True` (env: `CPI_AUTO_THRESHOLD`)
- **PCA variance threshold = 0.80**: thấp hơn → fallback WEIGHTED (env: `CPI_VARIANCE_THRESHOLD`)
- **Loss weight MTL = 0.7 BCE + 0.3 MSE**: churn binary là objective chính
- **PyTorch là optional dependency**: fallback về Standard AutoML khi không có torch
- **Auth qua API Key** (`X-API-Key`), không JWT cho MVP
- **Model cache TTL = 10 phút** trong `ModelCache`
- **Leakage threshold = 0.95**, **DUPLICATE threshold = 0.98**
- **`composite_config` là optional trong `TrainingRequest`**: nếu `None` → Standard AutoML luôn được dùng

---

## Mục 8: Những gì chưa có doc

| File/Module                                 | Chức năng                                        |
| ------------------------------------------- | ------------------------------------------------ |
| `backend/app/core/pipeline/builder.py`      | Auto-build sklearn Pipeline từ ColumnProfile     |
| `backend/app/core/pipeline/transforms/`     | Registry imputers/transformers theo strategy     |
| `backend/app/core/training/model_router.py` | Routing logic chọn model candidates              |
| `backend/app/core/training/automl.py`       | AutoML entry point — MTL vs Standard path        |
| `backend/app/core/serving/ab_service.py`    | A/B testing — hash assignment + exposure logging |
| `backend/app/core/etl/` (7 files)           | Toàn bộ ETL pipeline — không có doc nào          |
| `backend/app/core/storage.py`               | Unified StorageClient S3/local                   |
| `scripts/validation/validate_*.py`                     | Validation scripts — chưa có hướng dẫn           |
| `analytics/streamlit_app.py`                | Streamlit dashboard                              |

---

## Mục 9: Task tiếp theo (theo thứ tự ưu tiên)

### 1. Fix Column Review UI TypeScript errors — Phase 4 Frontend
*   **File**: `frontend/src/pages/ColumnReview.jsx`
*   **Dependency**: API profile & leakage endpoints done.

### 2. Training status UI — Phase 4 Frontend
*   **File**: `frontend/src/pages/Jobs.jsx`
*   **Dependency**: Jobs trigger & status APIs done.

### 3. Streamlit dashboard completion — Phase 3 BI
*   **File**: `analytics/streamlit_app.py`
*   **Dependency**: Model explainability & cohort analysis implementation.

### 4. Model drift monitoring — Phase 3 MLOps
*   **File**: `backend/app/core/serving/drift_detector.py`
*   **Dependency**: None.
