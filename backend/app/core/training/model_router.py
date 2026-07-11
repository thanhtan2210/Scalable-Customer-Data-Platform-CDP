import pandas as pd
from typing import List
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from ..profiler.column_profile import ColumnProfile, DataRole


def route_models(df: pd.DataFrame, confirmed_profiles: List[ColumnProfile]):
    """
    Evaluates dataset characteristics and profiles to route to appropriate ML models.
    Tabular dense -> Tree models; Sparse/Text heavy -> Linear models.
    """
    n_rows = len(df)

    # Phân tích Profiles để dự đoán độ thưa (Sparsity)
    has_text = any(p.inferred_role == DataRole.TEXT for p in confirmed_profiles)

    # Tính tổng cardinalities của Categorical để dự đoán bùng nổ chiều
    cat_cardinality = sum(
        p.unique_count
        for p in confirmed_profiles
        if p.inferred_role == DataRole.CATEGORICAL
    )
    is_sparse_prone = has_text or (
        cat_cardinality > n_rows * 0.1
    )  # Rất nhiều categories so với số dòng

    models = []

    # 1. SPARSE / TEXT HEAVY -> LINEAR MODELS
    if is_sparse_prone or n_rows < 1000:
        models.append(
            {
                "name": "LogisticRegression_Sparse",
                "class": LogisticRegression,
                "kwargs": {
                    "max_iter": 2000,
                    "solver": "saga",
                },  # Saga tốt cho l1/l2 trên dữ liệu thưa
                "search_space": {
                    "C": ("loguniform", 1e-4, 10.0),
                    "l1_ratio": ("loguniform", 0.01, 1.0),  # Kết hợp l1/l2 (ElasticNet)
                    "penalty": ("categorical", ["elasticnet"]),
                },
            }
        )

    # 2. TABULAR DENSE -> TREE MODELS
    if not is_sparse_prone and n_rows >= 500:
        # Nếu dataset đủ lớn cho XGBoost
        if n_rows > 2000:
            models.append(
                {
                    "name": "XGBClassifier",
                    "class": XGBClassifier,
                    "kwargs": {"eval_metric": "logloss", "random_state": 42},
                    "search_space": {
                        "n_estimators": ("int", 50, 300),
                        "learning_rate": ("loguniform", 1e-3, 0.3),
                        "max_depth": ("int", 3, 10),
                        "subsample": ("loguniform", 0.5, 1.0),
                    },
                }
            )
        # Random Forest luôn là một baseline vững chắc cho dữ liệu Dense vừa và nhỏ
        models.append(
            {
                "name": "RandomForestClassifier",
                "class": RandomForestClassifier,
                "kwargs": {"random_state": 42, "n_jobs": -1},
                "search_space": {
                    "n_estimators": ("int", 50, 200),
                    "max_depth": ("int", 5, 20),
                    "min_samples_leaf": ("int", 2, 10),
                },
            }
        )

    return models
