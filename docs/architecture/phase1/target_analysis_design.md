# Target Analysis Design: Composite Churn Signal

This document specifies the improved Target Analysis design for Phase 1, transitioning from a single-target label model (`suggested_target: str`) to a composite signal structure (**Composite Churn Signal**).

## 1. Problem with the Old Design

The initial design, which returned only a single column name representing the target (`suggested_target: str`), presented several significant limitations:
- **Hardcoded Entropy Threshold is Not Generalizable Across Domains:** Applying a rigid entropy threshold (e.g., `< 0.8`) easily missed valid target columns with a moderately imbalanced distribution (such as a churn rate of 15% - 35%, yielding a normalized entropy between `0.8` and `0.95`).
- **Omission of Related Column Information:** In real-world scenarios (especially in Churn prediction), target information is often distributed across multiple correlated columns such as `Churn Score`, `Churn Reason`, `Churn Label`, and `Churn Value`. Selecting only a single target column discarded these critical semantic relationships.
- **Violation of the Human-in-the-Loop Principle:** The system automatically selected a single target column without exposing alternative candidates alongside their confidence scores for user verification at the Column Review UI.

---

## 2. TargetAnalysis Object

The detailed structure of the `TargetAnalysis` object returned by the Orchestrator is defined using the following JSON Schema:

```json
{
  "recommended_target": "string — column name, top-1 candidate",
  "candidate_targets": [
    {
      "name": "string",
      "rank": 1,
      "score": 1.9,
      "signals": {
        "is_binary": true,
        "entropy": 0.8354,
        "entropy_score": 0.8,
        "keyword_match": true,
        "position_bonus": 0.0
      },
      "suggested_role": "TARGET"
    }
  ],
  "churn_column_group": [
    {
      "name": "string",
      "correlation_with_target": 0.99,
      "group_role": "DUPLICATE"
    }
  ],
  "recommended_auxiliary": ["string — columns that should be added to features"],
  "leakage_suspects": ["string — columns requiring user confirmation to drop"]
}
```

*Valid `suggested_role` values:* `TARGET`, `AUXILIARY`, `DUPLICATE`, `LEAKAGE`.
*Valid `group_role` values:* `PRIMARY`, `DUPLICATE`, `AUXILIARY`, `LEAKAGE_SUSPECT`.

---

## 3. Classification Rules for churn_column_group

Columns correlated with the target are detected using association metrics (Correlation / Cramer's V / Point-Biserial) with the `recommended_target` and classified according to the rules below:

| Correlation with target | group_role       | Default Action             |
|------------------------|------------------|----------------------------|
| > 0.98                 | DUPLICATE        | Auto-drop                  |
| 0.5 – 0.98             | AUXILIARY        | Add to features, user confirm |
| < 0.5                  | Not in group     | Profile normally           |

### Integration with leakage_score from Phase 1:
If a column is classified as `AUXILIARY` but also has `potential_leakage = True` (flagged by the data leakage detection logic):
* Its role in the group will be updated to **`LEAKAGE_SUSPECT`**.
* The system will flag this column, requiring the user to explicitly choose whether to keep or drop it at the Column Review UI before training.

---

## 4. Updated Entropy Scoring

The entropy scoring has been updated to a continuous scale to accurately represent the actual distribution of classification targets:

| Entropy range | Score | Rationale |
| :--- | :--- | :--- |
| `(0.75, 0.95]` | **+0.8** | Churn rate of 15–35%, the most common target distribution |
| `(0.5, 0.75]` | **+0.6** | Moderately imbalanced |
| `(0, 0.5]` | **+0.2** | Extremely imbalanced, likely noise or highly skewed feature |
| `= 0` or `> 0.95` | **+0.0** | Constant value or uniform distribution (highly balanced) |

> [!NOTE]
> These entropy ranges and score values are externalized as configuration constants in [config.py](file:///d:/Bon%20Bon/SourceCode/AI-project/Scalable-Customer-Data-Platform-CDP/backend/app/core/config.py) and can be overridden via environment variables during deployment.

---

## 5. Impact on Subsequent Phases

| Phase | Change |
| :--- | :--- |
| **Phase 1** | Orchestrator returns the `TargetAnalysis` object instead of a simple `suggested_target: str` string. |
| **Phase 2** | `build_pipeline()` is updated to accept additional `auxiliary_cols` from Target Analysis to optimize auxiliary features. |
| **Phase 3** | The profiling endpoint (`POST /api/v1/datasets/profile`) returns the `TargetAnalysis` structure embedded in the JSON response. |
| **Phase 4** | The Column Review UI displays the `churn_column_group` visually to let users review and approve keeping or dropping columns. |

---

## 6. Illustration with Telco Dataset

An example output of the `TargetAnalysis` object when running against the **Telco Customer Churn** dataset:

```json
{
  "recommended_target": "Churn Value",
  "candidate_targets": [
    {
      "name": "Churn Value",
      "rank": 1,
      "score": 1.9,
      "signals": {
        "is_binary": true,
        "entropy": 0.8354,
        "entropy_score": 0.8,
        "keyword_match": true,
        "position_bonus": 0.0
      },
      "suggested_role": "TARGET"
    },
    {
      "name": "Churn Label",
      "rank": 2,
      "score": 1.9,
      "signals": {
        "is_binary": true,
        "entropy": 0.8354,
        "entropy_score": 0.8,
        "keyword_match": true,
        "position_bonus": 0.0
      },
      "suggested_role": "TARGET"
    },
    {
      "name": "Senior Citizen",
      "rank": 3,
      "score": 1.6,
      "signals": {
        "is_binary": true,
        "entropy": 0.6400,
        "entropy_score": 0.6,
        "keyword_match": false,
        "position_bonus": 0.0
      },
      "suggested_role": "AUXILIARY"
    }
  ],
  "churn_column_group": [
    {
      "name": "Churn Value",
      "correlation_with_target": 1.0,
      "group_role": "PRIMARY"
    },
    {
      "name": "Churn Label",
      "correlation_with_target": 1.0,
      "group_role": "DUPLICATE"
    },
    {
      "name": "Churn Score",
      "correlation_with_target": 0.82,
      "group_role": "AUXILIARY"
    },
    {
      "name": "Churn Reason",
      "correlation_with_target": 0.91,
      "group_role": "LEAKAGE_SUSPECT"
    }
  ],
  "recommended_auxiliary": ["Churn Score"],
  "leakage_suspects": ["Churn Reason"]
}
```
