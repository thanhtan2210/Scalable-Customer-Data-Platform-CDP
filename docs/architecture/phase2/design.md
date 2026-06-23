# Phase 2: Dynamic Pipeline & Training Technical Design

**Status:** Design
**Goal:** Define the technical implementation strategy (HOW) for building the dynamic pipeline, generating validation schemas, routing models, and executing AutoML based on Phase 1 profiles.

## 1. Transform Registry

The Transform Registry maps the string-based strategies from `ColumnProfile` to instantiated scikit-learn compatible transformer classes.

| Transform Strategy | Transformer Class | Configuration / Notes |
| :--- | :--- | :--- |
| `standard` | `StandardScaler` | Applies standard normal scaling (mean=0, std=1). |
| `log` | `FunctionTransformer` | Applies `np.log1p` to handle highly right-skewed numeric data. |
| `power` | `PowerTransformer` | Applies Yeo-Johnson transform for variance stabilization. |
| `ohe` | `OneHotEncoder` | `handle_unknown='ignore'`, `sparse_output=False`. |
| `ordinal` | `OrdinalEncoder` | `handle_unknown='use_encoded_value'`, `unknown_value=-1`. |
| `tfidf` | `TfidfVectorizer` | `max_features=100`, `stop_words='english'`. Expects 1D string array. |
| `cyclical` | Custom Transformer | Extracts Sine/Cosine transformations from DateTime parts. |
| `passthrough` | `FunctionTransformer` | Identity function or scikit-learn's built-in `'passthrough'`. |
| `drop` | `'drop'` | Instructs the `ColumnTransformer` to exclude the column entirely. |

## 2. Pipeline Builder

The Pipeline Builder constructs a robust scikit-learn `Pipeline` tailored to the dataset.

**Processing Flow Diagram:**
```text
[Profiles Input] 
       |
       v
[Filter Columns] ---> DROP: Role IN (ID, IGNORE, TARGET)
       |
       v
[Group Columns]  ---> Numeric Cols, Categorical Cols, Text Cols, DateTime Cols
       |
       v
[Build Sub-Pipes]---> Imputer + Transformer for each group
       |
       v
[Assemble]       ---> ColumnTransformer([('num', num_pipe, num_cols), ...])
```

**Nested Pipeline Structure:**
The final object output by the builder is a nested scikit-learn Pipeline structured as follows:
1.  **Step 1 (`preprocessor`)**: A `ColumnTransformer` holding the respective sub-pipelines for each column group.
2.  **Step 2 (`classifier` or `model`)**: A placeholder (e.g., a dummy classifier) that will be hot-swapped during the AutoML Optuna trials.

## 3. Schema Generation & Pandera

This component automatically converts the dynamic profiling results into a strict Pandera Data Contract to enforce data integrity during serving.

**Role to Pandera Schema Mapping:**

| Inferred Role | Pandera Data Type | Pandera Checks / Rules |
| :--- | :--- | :--- |
| `NUMERIC` | `pa.Float` or `pa.Int` | `nullable=True` (imputation handles it later) |
| `CATEGORICAL` | `pa.String` or `pa.Category`| `nullable=True` |
| `TEXT` | `pa.String` | `nullable=True` |
| `DATETIME` | `pa.DateTime` | `nullable=True` |
| `ID` | *Excluded* | ID columns are not required for serving inference. |
| `TARGET` | *Excluded* | The target is what we are predicting; not part of inference input. |
| `IGNORE` | *Excluded* | Dropped columns are ignored. |

**Storage & Lifecycle:**
- **Cloud Mode (`STORAGE_MODE=s3`):** The schema is exported via `schema.to_json()` and saved directly to R2 at `ml_artifacts/{dataset_id}/{target_col}/schema.json`.
- **Local Mode (`STORAGE_MODE=local`):** Saved to `data/processed/{dataset_id}/{target_col}/schema.json`.
- **Phase 3 Loading:** During API startup (Phase 3), the serving service retrieves this JSON file using the `StorageClient` and reconstructs the validation gate via `DataFrameSchema.from_json()`, bypassing the need to re-run the profiling engine entirely.

## 4. Model Router

The Model Router acts as an expert system, selecting the best algorithm candidates based on dataset shape and complexity before kicking off the heavy AutoML process.

| Condition (Rows `N`, Features `F`, Classes `C`) | Recommended Model | Priority | Reason for Selection |
| :--- | :--- | :--- | :--- |
| Any condition (Baseline) | `LogisticRegression` | 1 | Fast, interpretable, acts as a solid baseline for all tabular tasks. |
| `N` > 1000 and `F` < 100 | `RandomForestClassifier` | 2 | Robust to non-linearities and unscaled features without overfitting easily. |
| `N` > 5000 | `XGBClassifier` | 3 | Highly performant on large, complex tabular datasets, but requires careful tuning. |

## 5. AutoML with Optuna

The AutoML component leverages Optuna to find the optimal hyperparameter configuration for the routed models.

**AutoML Execution Flow:**
```text
Validate Input -> Build Pipeline (Preprocessor) -> Route Models -> Define Search Space -> Optuna Optimize (n_trials) -> MLflow Log Run -> Register Best Model
```

**Search Space Definitions:**

| Model | Hyperparameters | Range / Values |
| :--- | :--- | :--- |
| `LogisticRegression` | `C` <br> `penalty` | `loguniform(1e-4, 10.0)` <br> `['l2']` |
| `RandomForestClassifier`| `n_estimators` <br> `max_depth` | `int(50, 300)` <br> `int(3, 15)` |
| `XGBClassifier` | `n_estimators` <br> `learning_rate` <br> `max_depth` | `int(50, 300)` <br> `loguniform(1e-3, 0.3)` <br> `int(3, 10)` |

**MLflow Logging Schema:**
Every successful Optuna trial and the final best run will be tagged with the following schema in MLflow:
- `dataset_hash`: MD5 hash of the raw dataset (for tracking data version).
- `target_col`: Name of the target variable.
- `model_class`: e.g., `XGBClassifier`.
- `best_roc_auc`: Float representing the cross-validated validation score.

## 6. Contract: Phase 1 & Phase 2

Strict interface boundaries exist between the Profiling Engine and the Pipeline Builder to ensure loose coupling.

**Data Contract:**
- **Input Type:** A Tuple containing:
  1.  `profiles`: `List[ColumnProfile]` (The full output from Phase 1 Orchestrator).
  2.  `suggested_target`: `str` (The column name of the target variable).
- **Output Type:** A Tuple containing:
  1.  `model_uri`: `str` (The MLflow URI pointing to the registered production model).
  2.  `schema_path`: `str` (The storage path to the generated `schema.json`).

**Explicit Failure Conditions:**
The Phase 2 pipeline build or training process MUST raise a fatal exception and fail immediately under the following conditions:
- The input `profiles` list is empty.
- The `suggested_target` is `None` or an empty string.
- Zero columns remain after filtering (meaning all columns were identified as ID, IGNORE, TARGET, or leaked).
- Target column has only one unique class (invalid for classification).

## 7. Future Enhancements (Deferred)

- **Missing Value Signal (`add_indicator=True` in SimpleImputer):** While highly beneficial for tree-based models to capture the behavioral signal of "missingness", this feature is currently deferred. Enabling it introduces cross-phase dependencies: it dynamically generates new columns during transformation, which require the `schema_gen` logic and Phase 3 API validation gates to be updated synchronously to recognize and permit these newly minted indicator columns.