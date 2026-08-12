# Adding New Datasets Guide

This guide explains how to add and register new benchmark datasets into the CDP platform catalog (`data/dataset/catalog.yaml`).

---

## 1. Overview of Dataset Catalog

The CDP platform uses a centralized YAML catalog (`data/dataset/catalog.yaml`) to track benchmark datasets used by testing scripts, profiling benchmarks, and E2E evaluation suites.

---

## 2. Step-by-Step Procedure to Add a Dataset

### Step 1: Download & Place Data File
Place your raw data file in a dedicated subfolder under `data/dataset/`:

```
data/dataset/
└── my-new-dataset/
    └── Data_File.csv
```

### Step 2: Register Entry in `catalog.yaml`
Open `data/dataset/catalog.yaml` and append a dataset schema entry:

```yaml
  - id: my_new_dataset
    name: "My New Customer Dataset"
    file: "data/dataset/my-new-dataset/Data_File.csv"
    separator: ","                          # Comma ',', Semicolon ';', or Tab '\t'
    target: "Churn"                        # Exact target column name
    target_type: "string_binary"           # string_binary, int_binary, or numeric
    id_columns: ["CustomerID"]             # Columns to exclude from training
    expected_nulls: true                   # Set true if null values exist
    notes: "Short description of source, rows, and domain."
    status: "verified"                     # 'verified' or 'pending'
```

### Step 3: Run E2E Integration Test
Validate that the dataset uploads, profiles, trains, and predicts successfully:

```powershell
python scripts/test_datasets_e2e.py --layer 4 --dataset my_new_dataset --fast
```

---

## 3. High-Quality Benchmark Datasets (Recommended)

Below are instructions for downloading external Kaggle benchmark datasets known to achieve high ROC-AUC ($>0.92$):

### 1. Credit Card Customers (`credit_card`)
- **Source**: [Kaggle - Credit Card Customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)
- **Size**: 10,127 rows $\times$ 23 columns
- **Target**: `Attrition_Flag` (`Attrited Customer` vs `Existing Customer`)
- **Target AUC**: $>0.95$ with LightGBM / XGBoost
- **File destination**: `data/dataset/credit-card-churn/BankChurners.csv`

### 2. IBM HR Employee Attrition (`ibm_hr_attrition`)
- **Source**: [Kaggle - IBM HR Analytics](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Size**: 1,470 rows $\times$ 35 columns
- **Target**: `Attrition` (`Yes` vs `No`)
- **File destination**: `data/dataset/ibm-hr-attrition/WA_Fn-UseC_-HR-Employee-Attrition.csv`
