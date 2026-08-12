# Scalable CDP Documentation Index

Welcome to the documentation directory for the **Scalable Customer Data Platform (CDP)**. Below is the navigation index for architectural documents, setup instructions, and developer guides.

---

## 📐 Architecture & System Design

- **[System Architecture Document](architecture/ARCHITECTURE.md)**  
  High-level system design diagram, microservice layer descriptions (Ingestion, Profiler, Pipeline, AutoML, Serving), model routing heuristics, 100-trial Optuna tuning, and batch prediction flow.

---

## 📚 Developer & Operational Guides

- **[Local Development Setup Guide](guides/local_setup.md)**  
  Bare-metal local environment setup using Python virtual environment, SQLite fallback database, local filesystem storage, and MLflow tracking server.

- **[AutoML & Training Engine Guide](guides/automl.md)**  
  Detailed breakdown of candidate models (LightGBM, CatBoost, XGBoost, Random Forest, Logistic Regression), search space ranges, Optuna `MedianPruner` early stopping, automated class imbalance handling (`class_weight='balanced'`), and stacking ensemble generation.

- **[End-to-End Testing Guide](guides/e2e_testing.md)**  
  Running the automated 4-layer dataset integration test suite (`scripts/test_datasets_e2e.py`) across all catalog datasets, command line options, `--fast` smoke testing flag, and failure troubleshooting.

- **[Adding New Datasets Guide](guides/adding_datasets.md)**  
  Step-by-step procedure for registering new raw datasets into `data/dataset/catalog.yaml`, target type specifications, and instructions for downloading Kaggle benchmark datasets (Credit Card Churn & IBM HR Attrition).

---

## 📁 Directory Structure

```
docs/
├── README.md                 # Documentation Index (this file)
├── architecture/
│   └── ARCHITECTURE.md       # High-level architecture & sequence flows
└── guides/
    ├── local_setup.md        # Bare-metal local setup instructions
    ├── automl.md             # AutoML model routing & tuning specs
    ├── e2e_testing.md        # End-to-end integration test runner guide
    └── adding_datasets.md    # Catalog registration & data acquisition
```
