# AutoML & Training Engine Guide

This document details the architecture, candidate model routing rules, hyperparameter optimization strategies, class imbalance handling, and stacking ensemble mechanisms built into the CDP AutoML Training Engine (`backend/app/core/training/`).

---

## 1. AutoML Pipeline Architecture

When a training job is dispatched via `POST /api/v1/jobs/{id}/train`, the engine executes the following automated pipeline:

```
Input DataFrame + Confirmed Column Profiles
  │
  ├── 1. Data Validation & Target Encoding (String targets mapped to 0/1)
  ├── 2. Preprocessing Pipeline Generation (ColumnTransformer with Imputers & Encoders)
  ├── 3. Model Routing (Evaluates dataset characteristics to select candidate algorithms)
  ├── 4. Class Imbalance Detection (Injects class_weight='balanced' if churn rate < 25%)
  ├── 5. Optuna Hyperparameter Optimization (100 Trials + MedianPruner)
  ├── 6. Stacking Ensemble Generation (Combines top-3 best single models)
  ├── 7. Optimal Threshold Calibration (Precision-Recall F1 Maximization on Val Set)
  └── 8. MLflow Logging & Model Registry Registration
```

---

## 2. Model Candidates & Search Spaces

The `route_models()` function dynamically selects algorithms appropriate for dataset density, row count, and sparsity:

### 1. LightGBM (`LGBMClassifier`)
- **Condition**: Tabular dense datasets ($N > 2000$).
- **Search Space**:
  - `n_estimators`: `[100, 600]`
  - `learning_rate`: `[0.001, 0.3]` (log scale)
  - `max_depth`: `[3, 12]`
  - `num_leaves`: `[15, 127]`
  - `subsample`: `[0.5, 1.0]`
  - `colsample_bytree`: `[0.5, 1.0]`
  - `reg_alpha` / `reg_lambda`: `[1e-4, 10.0]` (log scale)

### 2. CatBoost (`CatBoostClassifier`)
- **Condition**: Tabular dense datasets ($N > 2000$).
- **Search Space**:
  - `n_estimators`: `[100, 500]`
  - `learning_rate`: `[0.001, 0.3]` (log scale)
  - `depth`: `[3, 10]`
  - `l2_leaf_reg`: `[1e-3, 10.0]` (log scale)
  - `subsample`: `[0.5, 1.0]`

### 3. XGBoost (`XGBClassifier`)
- **Condition**: Tabular dense datasets ($N > 2000$).
- **Search Space**:
  - `n_estimators`: `[100, 500]`
  - `learning_rate`: `[0.001, 0.3]` (log scale)
  - `max_depth`: `[3, 10]`
  - `subsample` / `colsample_bytree`: `[0.5, 1.0]`
  - `reg_alpha` / `reg_lambda`: `[1e-4, 10.0]`
  - `min_child_weight`: `[1, 10]`

### 4. Random Forest (`RandomForestClassifier`)
- **Condition**: Tabular dense datasets ($N \ge 500$).
- **Search Space**:
  - `n_estimators`: `[100, 400]`
  - `max_depth`: `[5, 30]`
  - `min_samples_leaf`: `[1, 10]`
  - `max_features`: `["sqrt", "log2"]`

### 5. Logistic Regression (`LogisticRegression`)
- **Condition**: Sparse data or small datasets ($N < 1000$).
- **Search Space**:
  - `C`: `[1e-4, 10.0]` (log scale)
  - `l1_ratio`: `[0.01, 1.0]` (ElasticNet penalty)

---

## 3. Advanced Features

### 3.1 Optuna Early Pruning (`MedianPruner`)
To prevent wasting computation on suboptimal hyperparameter combinations, trial steps are evaluated using `optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)`. Trials performing worse than the median of previous trials at the same step are aborted early.

### 3.2 Class Imbalance Handling
The engine computes positive target prevalence:
$$\text{Churn Rate} = \frac{\sum y_i}{N}$$
If $\text{Churn Rate} < 0.25$ or $> 0.75$, the engine automatically configures:
- `class_weight='balanced'` for Scikit-learn, XGBoost, and LightGBM models.
- `auto_class_weights='Balanced'` for CatBoost models.

### 3.3 Stacking Ensemble
When $\ge 2$ model families complete optimization:
1. The top-3 performing single pipelines (ranked by cross-validation ROC-AUC) are selected.
2. A `StackingClassifier` is instantiated using a `LogisticRegression` meta-learner evaluated via 5-fold Stratified Cross-Validation.
3. If the stacking ensemble score exceeds the single best model score, the stacking model is trained on the full dataset and registered as the winning model.

---

## 4. Configuration Environment Variables

Tuning parameters can be adjusted via environment variables in `.env`:

```env
# Total HPO trial budget per candidate model family
OPTUNA_N_TRIALS=100

# Timeout limit (seconds) for entire HPO search
OPTUNA_TIMEOUT_SECONDS=3600

# Toggle Stacking Ensemble generation
ENABLE_STACKING=true
```
