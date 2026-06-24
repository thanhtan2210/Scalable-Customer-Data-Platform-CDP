# Phase 1: Profiling Engine & Data Contract Design

**Status:** Design
**Goal:** Define the domain-agnostic, behavior-driven profiling architecture that transforms raw tabular data into structured ML recipes.

## 1. ColumnProfile Schema
The `ColumnProfile` acts as the central data contract for the entire CDP. It stores statistical, semantic, and leakage metadata, serving as the definitive blueprint for pipeline building and UI rendering.

```json
{
  "name": "string (Original column name)",
  "inferred_dtype": "string (e.g., 'float64', 'object', 'datetime64')",
  "inferred_role": "enum (ID, TARGET, NUMERIC, CATEGORICAL, DATETIME, TEXT, IGNORE)",
  "confidence_score": "float (0.0 to 1.0, tracking certainty across layers)",
  "null_pct": "float (Percentage of missing values)",
  "unique_count": "int (Number of distinct values)",
  "entropy": "float (Measure of distribution randomness)",
  "mean_length": "float (Optional: Average text length for objects)",
  "regex_pattern": "string (Optional: Detected dominant semantic pattern)",
  "potential_leakage": "boolean (Flagged if highly correlated with the target)",
  "leakage_score": "float (Correlation coefficient with the target)"
}
```

## 2. Recipe Table (Role to Strategy Mapping)
Based on the `inferred_role`, the Pipeline Builder automatically assigns deterministic transformation and imputation strategies. User overrides at the UI level simply change the `inferred_role`, instantly updating the recipe.

| Inferred Role | Impute Strategy | Transform Strategy | Description |
| :--- | :--- | :--- | :--- |
| `ID` | Drop | Pass-through | High cardinality strings/ints; excluded from features. |
| `TARGET` | Drop Row | LabelEncoding | The objective variable; missing rows are unsafe to impute. |
| `NUMERIC` | Median | StandardScaler + Winsorize | Continuous values; capped for outliers and scaled. |
| `CATEGORICAL` | Mode / 'Unknown' | OneHotEncoder / Ordinal | Low-cardinality strings/ints; encoded for model ingestion. |
| `DATETIME` | Median (Time) | Date Parts Extraction | Dates split into derived features (Hour, Day, Is_Weekend). |
| `TEXT` | Constant ("") | TF-IDF / Embeddings | Long free-form text; requires NLP processing. |
| `IGNORE` | Drop Column | Drop | Columns with 100% nulls, 0 entropy, or manual exclusions. |

## 3. Layer Execution Logic
The engine executes sequentially. Layer 1 establishes the baseline. Layer 2 refines semantics. Layer 3 (LLM) is an expensive fallback triggered only when the role remains highly ambiguous.

| Layer | Input | Output Updates | Condition to Next Layer |
| :--- | :--- | :--- | :--- |
| **Layer 1: Statistical** | Raw Column Series | `inferred_dtype`, `null_pct`, `unique_count`, `entropy`, Base `inferred_role` | Always proceeds to Layer 2. |
| **Layer 2: Semantic** | Text/Object Series | `mean_length`, `regex_pattern`, Refined `inferred_role`, Adjusted `confidence_score` | If `confidence_score < 0.6`, proceed to Layer 3. |
| **Layer 3: LLM (Optional)** | Stats + Sample Values | Final `inferred_role`, Max `confidence_score` | Execution halts; profile is locked. |

### Orchestrator output contract
- **Output:** `(List[ColumnProfile], TargetAnalysis)`
- See details in [target_analysis_design.md](file:///d:/Bon%20Bon/SourceCode/AI-project/Scalable-Customer-Data-Platform-CDP/docs/architecture/phase1/target_analysis_design.md)

## 4. Zero-Assumption Target Detection
Target detection evaluates data behavior, not domain-specific naming conventions. The system scores columns to elect a `suggested_target`, prioritizing statistical suitability over naming hints.

*   **Primary Signals (Decisive):** 
    *   **Distribution:** Must be Binary (Cardinality = 2) or Low-Cardinality Classification.
    *   **Entropy:** Must be low but strictly > 0 (e.g., highly imbalanced data like 95/5 is valid, but 100/0 is a constant, not a target).
*   **Secondary Signals (Bootstrapping):**
    *   **Position:** Column located at the end or penultimate index of the DataFrame (+0.1 confidence).
    *   **Keyword Match:** Name contains generic indicators like 'target', 'label', 'churn', 'status' (+0.1 confidence).

*Note: See the updated entropy scoring scale and rationale in [target_analysis_design.md](file:///d:/Bon%20Bon/SourceCode/AI-project/Scalable-Customer-Data-Platform-CDP/docs/architecture/phase1/target_analysis_design.md#4-updated-entropy-scoring) section 4.*

## 5. Data Leakage Detection
To prevent future-predicting variables from ruining models, leakage checks are performed non-destructively. Columns are flagged, not dropped, ensuring human oversight.

*   **Mechanism:** Uses Point-Biserial correlation (Categorical Target vs. Numeric Feature) or Cramer's V (Categorical Target vs. Categorical Feature).
*   **Threshold:** If correlation > 0.95, `potential_leakage` becomes `true`, and `leakage_score` is recorded.
*   **Execution Lifecycle:**
    *   **Auto-Run:** Executed immediately after Layer 1 successfully identifies a `suggested_target`.
    *   **Manual Re-trigger:** If the user updates the target column in the Column Review UI, the frontend must call the `POST /api/v1/datasets/{dataset_id}/re-evaluate-leakage` endpoint (to be implemented in Phase 3) to recalculate scores against the new target.

## 6. Context-Blind LLM Prompting (Layer 3)
When Layer 1 and 2 fail to reach a confident conclusion, the LLM is queried without any domain assumptions, forcing it to deduce the role purely from statistical context.

```text
"Tôi có một tập dữ liệu không xác định ngành nghề. Phân tích cột có tên '{column_name}'. 
Các giá trị mẫu ngẫu nhiên: {sample_values}. 
Thống kê Layer 1 & 2: Dtype={inferred_dtype}, Nulls={null_pct}%, Cardinality={unique_count}, Pattern={regex_match}. 
Dựa trên các thông số này, hãy suy luận ngữ cảnh của cột, đề xuất Data Role (ID, CATEGORICAL, NUMERIC, DATETIME, TARGET, IGNORE) và phương pháp Impute tối ưu nhất."
```

## 7. Known Limitations (Out of Scope)
The current profiling engine is bounded by the MVP requirements and intentionally excludes complex data formats to maintain performance and simplicity.

- **Time-Series Churn:** Requires panel data processing (e.g., rolling windows, lagged features) which is unsupported; assumes a single flattened row per customer.
- **Unstructured Media:** Image and Audio columns are strictly out of scope and will cause ingestion failure or be forcefully ignored.
- **Complex Hierarchies:** Nested JSON arrays or XML blobs within columns are not automatically flattened and will likely be profiled as useless `ID` or `IGNORE` text fields.
- **Multi-label Target:** Selecting multiple targets simultaneously is out of scope — the system supports exactly one PRIMARY target; related columns are processed as AUXILIARY or LEAKAGE_SUSPECT in the TargetAnalysis.
