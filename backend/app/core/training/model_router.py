import pandas as pd
from typing import List
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from ..profiler.column_profile import ColumnProfile, DataRole


def _try_import_lightgbm():
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier
    except ImportError:
        return None


def _try_import_catboost():
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier
    except ImportError:
        return None


def route_models(df: pd.DataFrame, confirmed_profiles: List[ColumnProfile]):
    """
    Evaluates dataset characteristics and profiles to route to appropriate ML models.
    Tabular dense -> Tree models (XGBoost, LightGBM, CatBoost, RandomForest)
    Sparse/Text heavy -> Linear models.

    Model selection logic:
    - n_rows < 1000: LogisticRegression (too few samples for complex trees)
    - 1000 <= n_rows < 2000: RandomForest only
    - n_rows >= 2000 (tabular): XGBoost + LightGBM + CatBoost + RandomForest
    - sparse/text data: LogisticRegression (ElasticNet)
    """
    n_rows = len(df)
    n_cols = len([p for p in confirmed_profiles if p.inferred_role not in [DataRole.ID, DataRole.IGNORE, DataRole.TARGET]])

    # Analyze profiles for sparsity
    has_text = any(p.inferred_role == DataRole.TEXT for p in confirmed_profiles)
    cat_cardinality = sum(
        p.unique_count
        for p in confirmed_profiles
        if p.inferred_role == DataRole.CATEGORICAL
    )
    is_sparse_prone = has_text or (cat_cardinality > n_rows * 0.1)

    models = []

    # ─── 1. SPARSE / TEXT HEAVY → LINEAR MODELS ───
    if is_sparse_prone or n_rows < 1000:
        models.append(
            {
                "name": "LogisticRegression_Sparse",
                "class": LogisticRegression,
                "kwargs": {
                    "max_iter": 2000,
                    "solver": "saga",
                },
                "search_space": {
                    "C": ("loguniform", 1e-4, 10.0),
                    "l1_ratio": ("loguniform", 0.01, 1.0),
                    "penalty": ("categorical", ["elasticnet"]),
                },
            }
        )

    # ─── 2. TABULAR DENSE → TREE MODELS ───
    if not is_sparse_prone and n_rows >= 500:

        # ── XGBoost (extended search space) ──
        if n_rows > 2000:
            models.append(
                {
                    "name": "XGBClassifier",
                    "class": XGBClassifier,
                    "kwargs": {
                        "eval_metric": "logloss",
                        "random_state": 42,
                        "n_jobs": -1,
                        "verbosity": 0,
                    },
                    "search_space": {
                        "n_estimators": ("int", 100, 500),
                        "learning_rate": ("loguniform", 1e-3, 0.3),
                        "max_depth": ("int", 3, 10),
                        "subsample": ("loguniform", 0.5, 1.0),
                        "colsample_bytree": ("loguniform", 0.5, 1.0),
                        "reg_alpha": ("loguniform", 1e-4, 10.0),
                        "reg_lambda": ("loguniform", 1e-4, 10.0),
                        "min_child_weight": ("int", 1, 10),
                    },
                }
            )

        # ── LightGBM (typically 1-3% higher AUC than XGBoost) ──
        LGBMClassifier = _try_import_lightgbm()
        if LGBMClassifier is not None and n_rows > 2000:
            models.append(
                {
                    "name": "LGBMClassifier",
                    "class": LGBMClassifier,
                    "kwargs": {
                        "random_state": 42,
                        "n_jobs": -1,
                        "verbose": -1,
                    },
                    "search_space": {
                        "n_estimators": ("int", 100, 600),
                        "learning_rate": ("loguniform", 1e-3, 0.3),
                        "max_depth": ("int", 3, 12),
                        "num_leaves": ("int", 15, 127),
                        "subsample": ("loguniform", 0.5, 1.0),
                        "colsample_bytree": ("loguniform", 0.5, 1.0),
                        "reg_alpha": ("loguniform", 1e-4, 10.0),
                        "reg_lambda": ("loguniform", 1e-4, 10.0),
                        "min_child_samples": ("int", 5, 50),
                    },
                }
            )

        # ── CatBoost (excellent with categoricals, no manual encoding needed) ──
        CatBoostClassifier = _try_import_catboost()
        if CatBoostClassifier is not None and n_rows > 2000:
            models.append(
                {
                    "name": "CatBoostClassifier",
                    "class": CatBoostClassifier,
                    "kwargs": {
                        "random_state": 42,
                        "verbose": 0,
                        "thread_count": -1,
                    },
                    "search_space": {
                        "n_estimators": ("int", 100, 500),
                        "learning_rate": ("loguniform", 1e-3, 0.3),
                        "depth": ("int", 3, 10),
                        "l2_leaf_reg": ("loguniform", 1e-3, 10.0),
                        "subsample": ("loguniform", 0.5, 1.0),
                        "min_data_in_leaf": ("int", 1, 30),
                    },
                }
            )

        # ── RandomForest (robust baseline) ──
        models.append(
            {
                "name": "RandomForestClassifier",
                "class": RandomForestClassifier,
                "kwargs": {"random_state": 42, "n_jobs": -1},
                "search_space": {
                    "n_estimators": ("int", 100, 400),
                    "max_depth": ("int", 5, 30),
                    "min_samples_leaf": ("int", 1, 10),
                    "max_features": ("categorical", ["sqrt", "log2"]),
                    "min_samples_split": ("int", 2, 10),
                },
            }
        )

    return models
