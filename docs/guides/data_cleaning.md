# Data Cleaning Guide for Telco Customer Churn Dataset

This document describes a practical, step-by-step process to clean the Telco customer churn dataset used in this project. The guide includes ready-to-run code snippets, common pitfalls, and recommendations for reproducible cleaning.

File references
- Dataset used in this notebook: `Telco_customer_churn.xlsx`
- Notebook: `clean_EDA.ipynb`

---

## 1. Overview

- Purpose: convert raw dataset into a clean, analysis-ready table suitable for exploratory data analysis and modeling.

- Meaning: Establishes the main goals and boundaries of the cleaning work so you can judge whether a change is appropriate for analysis or modeling. It helps prioritize tasks (e.g., fix types, remove duplicates) and communicate expected outcomes to stakeholders.

Tasks covered (high-level)
- Read and inspect the file
- Normalize column names and select relevant columns
- Convert types (numeric, dates)
- Handle missing values and duplicates
- Clean categorical values and encode
- Detect and handle outliers
- Basic feature engineering
- Save cleaned dataset

---

## 2. Prerequisites

- Python 3.8+ and common data packages: `pandas`, `numpy`, `matplotlib`, `seaborn`.
- Optional for modeling/statistics: `scipy`, `scikit-learn`.

Install recommended packages (Windows PowerShell):

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

If you use conda:

```powershell
conda install -c conda-forge pandas numpy matplotlib seaborn scipy scikit-learn
```

Make sure VS Code/Jupyter uses the same interpreter/environment you installed packages into.

- Meaning: Ensures reproducibility — having the right packages in the same environment prevents "module not found" or version mismatch errors when running cells. This step reduces friction when sharing the notebook.

---

## 3. Read the data

Python snippet:

```python
import pandas as pd
df = pd.read_excel('Telco_customer_churn.xlsx')
df.shape
df.head()
df.dtypes
```

What to check:
- Is the file located in the same folder as the notebook? If not, provide correct relative or absolute path.
- Check the number of rows/columns and a few top rows.

- Meaning: A quick initial check helps spot formatting issues, wrong sheets, or encoding problems before spending time on heavier processing. It gives a fast picture of initial data quality.

---

## 4. Normalize column names and select columns

Reasons: whitespace, hidden characters or inconsistent capitalization cause hard-to-find bugs.

```python
# strip whitespace and unify case
df.columns = df.columns.str.strip()
```

Select only relevant columns (example used in this project):

```python
cols = ['CustomerID','City','Gender','Senior Citizen','Partner','Dependents',
        'Tenure Months','Phone Service','Multiple Lines','Internet Service',
        'Online Security','Online Backup','Device Protection','Tech Support',
        'Streaming TV','Streaming Movies','Contract','Paperless Billing',
        'Payment Method','Monthly Charges','Total Charges','Churn Label',
        'Churn Value','Churn Score','CLTV','Churn Reason']
df = df[cols].copy()
```

If a listed column does not exist, inspect `df.columns` and adapt the list.

- Meaning: Normalizing column names prevents typos and hidden-character bugs. Selecting a focused subset of columns reduces memory usage and keeps the workflow simpler and reproducible.

---

## 5. Convert types

- `Total Charges` often imported as object because of stray characters – convert to numeric.

```python
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
df['Tenure Months'] = pd.to_numeric(df['Tenure Months'], errors='coerce')
```

- Re-check types with `df.info()`.

- Meaning: Ensures numeric computations and aggregations work correctly and prevents accidental string arithmetic. Use `df.info()` to verify conversions and find any remaining problematic columns.

---

## 6. Missing values strategy

Steps:

1. Quantify missingness:

```python
df.isna().sum()
```

2. Decide per-column action:
- If a column critical for analysis (e.g. `Total Charges`) has very few NaNs → drop those rows.
- If a column numeric with moderate missing → impute (median) or model-based imputation.
- If a categorical column → treat NaN as a separate category or impute with mode.

Examples:

```python
# drop rows where Total Charges is missing (common in Telco dataset)
df.dropna(subset=['Total Charges'], inplace=True)

# impute Tenure with median
df['Tenure Months'] = df['Tenure Months'].fillna(df['Tenure Months'].median())

# categorical fill
df['Internet Service'] = df['Internet Service'].fillna('Unknown')
```

Document any drops/imputations (log counts before/after) for reproducibility.

- Meaning: The chosen missing-value strategy (drop vs impute) affects bias and model outcomes. Logging counts before/after gives transparency and allows auditing or reverting the decision.

---

## 7. Duplicates

```python
dups = df.duplicated().sum()
print('duplicate rows:', dups)
df = df.drop_duplicates()
```

If duplicates exist because of ID duplication, inspect rows and decide which to keep.

- Meaning: Removing duplicates prevents inflated counts and biased statistics or models. When duplicates reflect different events for the same customer, consider aggregation rather than deletion.

---

## 8. Clean categorical values

Common tasks:

- Trim whitespace and unify case:

```python
for c in ['Gender','Internet Service','Contract','Payment Method','Churn Label']:
    df[c] = df[c].astype(str).str.strip()
```

- Map binary strings to 0/1 for modeling:

```python
df['Partner'] = df['Partner'].map({'Yes':1,'No':0})
df['Senior Citizen'] = df['Senior Citizen'].map({'Yes':1,'No':0})
df['Dependents'] = df['Dependents'].map({'Yes':1,'No':0})
```

- Use `pd.get_dummies()` for multi-class categoricals when needed for models.

- Meaning: Cleaning categorical strings (trimming, case) avoids accidental category splits. Encoding (binary/one-hot) converts human-readable labels into numeric features required by most ML algorithms.

---

## 9. Outliers

Quick IQR-based capping (winsorization):

```python
def cap_outliers(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    return series.clip(lower, upper)

df['Monthly Charges'] = cap_outliers(df['Monthly Charges'])
```

Decide whether to cap or remove outliers depending on domain knowledge and modeling needs.

- Meaning: Outliers can disproportionately influence means and model fits. Capping (winsorizing) reduces impact while preserving observations; removing should be reserved for clear data errors.

---

## 10. Transformations

- If `Total Charges` has a heavy right-skew, consider log-transform for visualization/modeling.

- Meaning: Transformations like log1p make skewed distributions more symmetric, stabilize variance, and often improve performance of linear models or distance-based algorithms.

```python
df['TotalCharges_log'] = np.log1p(df['Total Charges'])
```

---

## 11. Feature engineering

Examples useful for churn analysis:

- Average monthly spend (guard against division by zero):

```python
df['avg_monthly'] = df['Total Charges'] / df['Tenure Months'].replace({0:1})
```

- Tenure groups:

```python
df['tenure_group'] = pd.cut(df['Tenure Months'], bins=[-1,6,12,24,60,999], labels=['0-6','7-12','13-24','25-60','60+'])
```

---

- Meaning: New features (e.g., average monthly spend, tenure groups) capture domain-specific signals that raw columns may not express directly, often improving model discriminative power.

## 12. Encoding for modeling

- Binary map: `Yes/No` → `1/0`.
- One-hot: `pd.get_dummies(df, columns=[...], drop_first=True)`.
- Label encoding only when categories are ordinal.

Scale numeric features if algorithm requires it (e.g., Logistic Regression, SVM):

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_cols = ['Monthly Charges','Total Charges','Tenure Months','avg_monthly']
df[num_cols] = scaler.fit_transform(df[num_cols])
```

Note: install `scikit-learn` with `python -m pip install scikit-learn` (don't use `pip install sklearn`).

- Meaning: Proper encoding and scaling ensure numeric features are comparable and that models relying on distances or gradients behave well and converge reliably.

---

- Meaning: Saving the cleaned dataset preserves your preprocessing work so downstream experiments and model training can reuse the cleaned data without re-running expensive steps.

## 13. Save cleaned dataset

Preferred formats: Parquet (fast, preserves dtypes) or CSV.

```python
df.to_parquet('Telco_customer_churn_clean.parquet', index=False)
# or
df.to_csv('Telco_customer_churn_clean.csv', index=False)
```

---

## 14. Reproducibility & logging

- Keep a copy of raw data unmodified.
- Save the notebook with each cleaning step in separate cells with comments.
- Log row counts and missing summaries before/after each major operation.

Example logging pattern:

```python
def log_step(name):
    print(name)
    print('shape:', df.shape)
    print(df.isna().sum())

log_step('after load')
# perform op...
log_step('after dropna Total Charges')

- Meaning: Logging each major step and its effect on shape/missingness provides an audit trail to explain how data changed, aiding debugging and reproducibility.
```

---

## 15. Quick troubleshooting

- "File not found": check working directory and relative path. Use `!pwd` (or `os.getcwd()`).
- "pd.to_numeric" produced many NaNs: check for non-numeric characters (commas, currency symbols).
- Installing `sklearn` failed — use `scikit-learn` package name instead.
- Pairplot or heavy plots slow: sample with `df.sample(1000, random_state=1)`.

- Meaning: Quick troubleshooting tips save time by addressing common runtime issues (missing files, package errors, heavy plotting) so you can focus on analysis.

---

## 16. Suggested notebook cells order (recommended)

1. Imports and read data
2. Quick inspection (shape, head, dtypes)
3. Column normalization and selection
4. Type conversions
5. Missing value handling
6. Duplicates removal
7. Categorical cleaning
8. Numeric EDA (hist/KDE/boxplot)
9. Outlier handling
10. Feature engineering
11. Encoding & scaling
12. Save cleaned data

- Meaning: The suggested order gives a logical, incremental flow that is easy to follow, test, and revert — improving readability and reproducibility of the notebook.


