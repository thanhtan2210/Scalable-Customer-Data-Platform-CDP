# Scalable Customer Data Platform (CDP)

Goal: build a scalable customer data processing platform (50M users target) for Data Engineering (DE), Data Science (DS), and MLOps.

## Overview
- Goal: Ingest & build a clean feature table for downstream modeling and analytics.
- Key constraints: S3 as raw + feature storage (Parquet), Spark for heavy ETL, partitioning strategy, eventual Iceberg/Delta Lake for ACID & schema evolution.
- CV line: "Data Warehousing (BigQuery/Simulating Iceberg architecture) — built S3 Parquet pipeline with Spark for data cleaning & dedup".

## 🚀 Quick Start & Usage
The project includes a unified `launcher.py` to manage all services and jobs.

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- Apache Spark (for local execution)

### Launching Services
```powershell
# 1. Start Infrastructure (MinIO, MLflow, Postgres)
docker-compose up -d

# 2. Run Data Pipeline
python launcher.py pipeline

# 3. Train Model
python launcher.py train

# 4. Start APIs
python launcher.py churn-api
python launcher.py ab-service

# 5. Launch Dashboard
python launcher.py dashboard
```

## Stage 1 — Data Engineering (DE)
- Purpose: Standardize raw data, remove noise and duplicates, and create a well-partitioned feature table for downstream jobs.

1. **Ingestion**:
    - Raw data landing in S3 (MinIO) as Parquet.
    - Script: `scripts/ingest_to_minio.py` & `scripts/csv_to_parquet.py`.

2. **Cleaning (Spark)**:
    - Schema enforcement (explicit schema, types).
    - Null handling and conversions (e.g., Total Charges numeric).
    - Normalize categorical values with canonical mapping.
    - Outlier detection/handling for numeric features (capping / winsorize).
    - Script: `spark_jobs/clean_data_spark.py`.

3. **Duplicate handling**:
    - Dedup based on stable keys (CustomerID + event_timestamp / last_updated).
    - Use windowing to pick latest record:
    - Partition by CustomerID, order by last_updated desc, row_number() == 1.

4. **Partitioning & Storage**:
    - Write Parquet to S3 with partition layout like: `s3://bucket/cdp/customer_features/year=YYYY/month=MM/day=DD/`
    - Partition keys: date (event_date) & maybe region or product line (low-cardinality).
    - Avoid partition by high-cardinality keys (CustomerID).

5. **Metadata & Format**:
    - Start on Parquet; evaluate Delta Lake or Apache Iceberg for ACID/merge/schema evolution.
    - For local testing, simulate Iceberg/Delta with Docker + Spark (set up catalog).

6. **Orchestration**:
    - Use Airflow to schedule jobs (daily incremental, hourly micro-batches).
    - Jobs: ingest -> validate (Great Expectations) -> transform (Spark) -> write feature table -> register metadata (Glue/Metastore or Iceberg catalog).

- Practical checks:
    - Validate row counts, null rates, detect sudden changes (data drift).
    - Compact small Parquet files after many small writes.

### Best practices for partitioning + file sizing:
- Aim for file sizes between 128MB - 512MB.
- Use `coalesce()` or `repartition()` before writing to control file count.
- Implement a "Compaction Job" to merge small files in older partitions.

### Delta Lake / Apache Iceberg notes (local with Docker):
- Why: ACID, time travel, schema evolution, MERGE.
- Local testing: run Spark with delta-core or iceberg-spark runtime jar:
	- Example with Delta Lake (Spark + delta-core):
	- Add delta-core jar to Spark session and write as `format("delta")`.
	- Example with Iceberg, configure Spark session:
	- `spark.sql.catalog.local = 'org.apache.iceberg.spark.SparkCatalog'` with `catalog-impl` set to `hadoop`.
- If infra limited: keep Parquet on S3 and simulate Iceberg via partitioned layout + metadata.

### Airflow DAG (skeleton)
- See `dags/telco_pipeline.py` for the implementation.
- Uses `PythonOperator` and `SparkSubmitOperator` to chain ingestion and cleaning.

### Validation & QA:
- **Great Expectations**: Checkpoints defined in `great_expectations/checkpoints/`.
- **Lineage**: Tracked in `src/etl/lineage.py`.

## Stage 2 — Data Science (DS)
- Use the feature table from DE; standard pipeline: train/test split, model baseline (Logistic), model improvements (XGBoost/LightGBM), hyperparameter tune, calibration, SHAP explanations.
- Persist model + preprocessing (joblib / ONNX).
- **MLflow Tracking**: All experiments logged to `http://localhost:5000`.

## Stage 3 — MLOps / Serving & BI
- Export model & feature pipeline; serve via FastAPI or model server (SageMaker / TorchServe).
- Online inference: retrieve latest features (via feature store or read Parquet & filter).
- Batch/Streaming scores: schedule daily batch job to score all customers and write predictions to CDP (S3/warehouse/DB).
- **Dashboard**: Real-time business metrics via Streamlit (`Sales_Dashboard/`).

## Project Structure
```text
├── dags/               # Airflow DAGs
├── data/               # Local data samples (raw/processed)
├── deploy/             # Dockerfiles and deployment configs
├── docs/               # Detailed documentation
├── models/             # Trained model artifacts (.joblib)
├── notebooks/          # EDA and Analysis notebooks
├── scripts/            # Utility and standalone scripts
├── spark_jobs/         # PySpark ETL scripts
├── src/                # Core application logic
│   ├── api/            # FastAPI endpoints (Churn & A/B)
│   ├── etl/            # Transformation and cleaning logic
│   └── models/         # Model training wrappers
└── tests/              # Pytest suite
```

## Tech Stack recommendation
- **Storage**: MinIO (S3 Compatible)
- **Processing**: Apache Spark
- **Database**: PostgreSQL (for A/B Service metadata)
- **ML Tracking**: MLflow
- **Serving**: FastAPI & Streamlit
- **Validation**: Great Expectations

## Deliverables
- [x] Spark ETL jobs (Clean / Dedup / Partition)
- [x] Airflow DAG for pipeline orchestration
- [x] FastAPI for Churn Prediction & A/B Assignment
- [x] Automated ROI Evaluation report
- [x] Executive Dashboard (Streamlit)
- [x] Comprehensive test suite (Pytest)

## Notes / Fallback
- If Iceberg/Delta not feasible in current infra: use Parquet on S3 + strong partitioning + metadata in Glue/Metastore.
- Record steps in CV: highlight S3 Parquet with simulated Iceberg architecture if used BigQuery as warehouse.

# Context
A fictional telco company that provided home phone and Internet services to 7043 customers in California in Q3.

# Data Description
7043 observations with 33 variables
- **CustomerID**: A unique ID that identifies each customer.
- **Tenure Months**: Amount of months customer has been with company.
- **Monthly Charge**: Current total monthly charge.
- **Total Charges**: Cumulative charges.
- **Churn Value**: 1 = Churned, 0 = Remained.
- **CLTV**: Customer Lifetime Value.
- *(See docs/data_quality.md for full schema details)*

# Source
# Data Description
7043 observations with 33 variables

CustomerID: A unique ID that identifies each customer.

Count: A value used in reporting/dashboarding to sum up the number of customers in a filtered set.

Country: The country of the customer’s primary residence.

State: The state of the customer’s primary residence.

City: The city of the customer’s primary residence.

Zip Code: The zip code of the customer’s primary residence.

Lat Long: The combined latitude and longitude of the customer’s primary residence.

Latitude: The latitude of the customer’s primary residence.

Longitude: The longitude of the customer’s primary residence.

Gender: The customer’s gender: Male, Female

Senior Citizen: Indicates if the customer is 65 or older: Yes, No

Partner: Indicate if the customer has a partner: Yes, No

Dependents: Indicates if the customer lives with any dependents: Yes, No. Dependents could be children, parents, grandparents, etc.

Tenure Months: Indicates the total amount of months that the customer has been with the company by the end of the quarter specified above.

Phone Service: Indicates if the customer subscribes to home phone service with the company: Yes, No

Multiple Lines: Indicates if the customer subscribes to multiple telephone lines with the company: Yes, No

Internet Service: Indicates if the customer subscribes to Internet service with the company: No, DSL, Fiber Optic, Cable.

Online Security: Indicates if the customer subscribes to an additional online security service provided by the company: Yes, No

Online Backup: Indicates if the customer subscribes to an additional online backup service provided by the company: Yes, No

Device Protection: Indicates if the customer subscribes to an additional device protection plan for their Internet equipment provided by the company: Yes, No

Tech Support: Indicates if the customer subscribes to an additional technical support plan from the company with reduced wait times: Yes, No

Streaming TV: Indicates if the customer uses their Internet service to stream television programing from a third party provider: Yes, No. The company does not charge an additional fee for this service.

Streaming Movies: Indicates if the customer uses their Internet service to stream movies from a third party provider: Yes, No. The company does not charge an additional fee for this service.

Contract: Indicates the customer’s current contract type: Month-to-Month, One Year, Two Year.

Paperless Billing: Indicates if the customer has chosen paperless billing: Yes, No

Payment Method: Indicates how the customer pays their bill: Bank Withdrawal, Credit Card, Mailed Check

Monthly Charge: Indicates the customer’s current total monthly charge for all their services from the company.

Total Charges: Indicates the customer’s total charges, calculated to the end of the quarter specified above.

Churn Label: Yes = the customer left the company this quarter. No = the customer remained with the company. Directly related to Churn Value.

Churn Value: 1 = the customer left the company this quarter. 0 = the customer remained with the company. Directly related to Churn Label.

Churn Score: A value from 0-100 that is calculated using the predictive tool IBM SPSS Modeler. The model incorporates multiple factors known to cause churn. The higher the score, the more likely the customer will churn.

CLTV: Customer Lifetime Value. A predicted CLTV is calculated using corporate formulas and existing data. The higher the value, the more valuable the customer. High value customers should be monitored for churn.

Churn Reason: A customer’s specific reason for leaving the company. Directly related to Churn Category.

This dataset is detailed in:
https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113

Downloaded from:
https://community.ibm.com/accelerators/?context=analytics&query=telco%20churn&type=Data&product=Cognos%20Analytics

There are several related datasets as documented in:
https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2018/09/12/base-samples-for-ibm-cognos-analytics
---

## Project Docs
- Architecture: docs/architecture.md
- Getting Started: docs/getting_started.md
- Airflow DAG: docs/airflow_dag.md
- Configuration: docs/configuration.md
- Data Quality & Lineage: docs/data_quality.md
- Deployment: docs/deployment.md

## Proof of Business Value (offline evaluation)

I ran an offline evaluation on the provided Telco dataset to estimate model utility and a conservative ROI for an outreach intervention. The evaluation script is at `scripts/evaluate_offline.py` and outputs a report in `reports/offline_evaluation.json`.

Summary results (from `reports/offline_evaluation.json`):

- **Model used**: trained_and_saved (simple Logistic Regression trained on numeric features + `Contract` one-hot)
- **AUC**: 0.9717
- **Precision@k** (top 10% of test set, k=211): 1.0
- **Estimated saved customers** (precision@k * k * outreach_success_rate=0.3): 63.3
- **Average CLTV (dataset mean)**: 4401.45
- **Estimated benefit**: 278,611.48
- **Outreach cost** (assumed $5 per contacted customer): 1,055.00
- **Estimated ROI**: 263.09 (benefit divided by cost)

Notes and caveats:
- This is an **offline, backtest-style** estimate. Results may be optimistic due to highly separable features.
- The outreach success rate (`0.3`) and `cost_per_outreach` (`$5`) are assumptions.

## Experiment Design & How to Run A/B Test

Use the provided scripts to assign customers deterministically, compute sample sizes, and analyze results.

- **Assign customers**:
```powershell
python scripts/ab_assign.py --input data/raw/cleaned_telco.csv --out reports/ab_assignment.csv --ratio 0.5
```

- **Compute sample size**:
```powershell
python scripts/ab_sample_size.py --p0 0.03 --mde 0.006 --alpha 0.05 --power 0.8
```

- **Analyze results**:
```powershell
python scripts/analyze_ab_results.py --assign reports/ab_assignment.csv --outcomes reports/ab_outcomes.csv --report reports/ab_analysis.json
```
