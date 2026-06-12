# Phase 0: Cloud Storage Foundation & Infrastructure

**Date:** June 2026
**Scope:** Foundation & Infrastructure

This document defines the foundational architecture for Cloud Storage within the Scalable Customer Data Platform (CDP). Because the local environment is resource-constrained (making local MinIO via Docker unviable for heavy workloads), we treat **Cloudflare R2** as our primary S3-compatible Data Lake right from the foundational phase.

## 1. Core Objectives of Storage Infrastructure

In Phase 0, the Cloud Storage layer must provide a robust, scalable, and cost-effective foundation that serves all downstream consumers (Data Engineering pipelines, ML Training, and Model Serving).

- **Unified Protocol:** All internal services communicate with storage strictly via the **S3 API standard** (using `boto3`, `s3fs`, or PySpark's `hadoop-aws`). This ensures complete abstraction; the codebase does not know if it is talking to MinIO, AWS S3, or Cloudflare R2.
- **Cost-Efficiency:** Utilizing Cloudflare R2 eliminates egress fees, which is critical for Data Engineering pipelines that pull and write large datasets iteratively.
- **Resilience:** Serves as the single source of truth (SSOT) for data state, preventing data loss if local bare-metal processes crash.

## 2. Data Lake Architectural Zones (Bucket Layout)

The storage infrastructure is structured into logical zones within a single bucket (e.g., `cdp-datalake-assets`). This strict partitioning prevents data contamination.

```text
cdp-datalake-assets/
├── landing_zone/          # RAW DATA
│   └── {dataset_id}/      # Incoming raw CSV/Excel files (immutable)
│       └── {original_filename}
│
├── processing_zone/       # INTERMEDIATE
│   ├── {dataset_id}/      # Temporary outputs during ETL
│   └── checkpoints/       # Pandera schema JSON generated from profiler/schema_gen.py
│
├── feature_store/         # PARQUET FEATURE TABLES
│   └── {dataset_id}/      # Partitioned by Year -> Month -> Day
│       ├── year=2026/     
│       │   └── month=06/
│       │       └── day=11/
│       │           └── part-0000.parquet
│
└── ml_artifacts/          # MLOPS
    └── {dataset_id}/      # MLflow serialized models, profiling schemas and JSON reports
```

## 3. Infrastructure Configuration & Security

As part of the foundation, the following infrastructure rules must be established:

### A. Access Management (IAM/Tokens)
- **Granular Tokens:** Generate an R2 API token with specific access limited **only** to the `cdp-datalake-assets` bucket.
- **No Public Read:** The bucket must remain private. All read/write operations must be signed via HMAC (boto3 standard).

### B. CORS Configuration
If the Vercel Frontend needs to upload raw files directly to R2 (to bypass Render.com's memory/timeout limits on heavy files), CORS must be configured on the R2 bucket to allow `PUT` requests from the Vercel domain.

### C. Secret Management
Infrastructure credentials must never be hardcoded. They are injected at runtime via environment variables:
- `S3_ENDPOINT_URL`
- `S3_ACCESS_KEY_ID`
- `S3_SECRET_ACCESS_KEY`
- `S3_REGION` (often 'auto' or 'us-east-1' for R2)
- `S3_BUCKET_NAME`

## 4. Storage Integration Strategy (Python/Spark)

To establish the infrastructure foundation in the codebase, we will implement a centralized `StorageClient` singleton (e.g., in `backend/app/core/storage.py`).

**Key Features of the Foundation Code:**
- **S3fs Integration:** For Pandas to read S3 streams directly (`pd.read_parquet('s3://...')`).
- **Boto3 Client:** For presigned URLs, file listing, and artifact management.
- **Spark Hadoop Configuration:** Setting `fs.s3a.endpoint`, `fs.s3a.access.key`, and `fs.s3a.secret.key` directly in the Spark session builder to allow Spark to write partitioned data directly to the Cloud layer.

## 5. Local Fallback Provision

While Cloudflare R2 is the primary storage, the infrastructure design mandates a "Local Fallback" pattern. If `STORAGE_MODE=local`, the system automatically reroutes all paths to the local `data/` directory, mirroring the Cloud layout. This is crucial for offline development and fast unit testing.
