# Backlog & Proposed Enhancements

This document captures high-value data science and engineering proposals that are currently out of scope for the MVP but should be considered for future iterations.

## 1. Target Encoding for Categorical Features
*   **Context:** Currently, the system uses `OneHotEncoder` (with `max_categories` limits) and `OrdinalEncoder` for categorical data.
*   **Proposal:** Introduce `TargetEncoder` for categorical columns with moderate cardinality (e.g., between 5 and 50 unique values). Target encoding replaces categorical values with the mean of the target variable for that category.
*   **Benefit:** This technique often yields significantly higher predictive power (AUC) for tree-based models (Random Forest, XGBoost) on classification tasks like Churn, without the dimensionality explosion caused by One-Hot Encoding.
*   **Dependency:** Requires careful implementation within a cross-validation loop during training to prevent severe data leakage.
