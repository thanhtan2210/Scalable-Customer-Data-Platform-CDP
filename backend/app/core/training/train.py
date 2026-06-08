import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
LOCAL_INPUT = BASE_DIR / "data" / "raw" / "cleaned_telco.csv"
INPUT_PATH = os.environ.get("FEATURES_PATH", str(LOCAL_INPUT))
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def _load_data(path):
    # Support local CSV for experiments, otherwise expect parquet path
    if str(path).endswith(".csv") and Path(path).exists():
        df = pd.read_csv(path)
    else:
        # try parquet (may be s3 path)
        try:
            df = pd.read_parquet(path)
        except Exception:
            raise FileNotFoundError(f"Data not found at {path}")
    return df


def _feature_pipeline(df):
    # numeric features
    num_feats = [
        "Tenure Months",
        "Monthly Charges",
        "Total Charges",
        "Churn Score",
        "CLTV",
    ]
    # categorical features
    cat_feats = [
        "Contract",
        "Payment Method",
        "Internet Service",
        "Gender",
        "Senior Citizen",
        "Partner",
        "Dependents",
        "Paperless Billing",
    ]

    # ensure columns exist in df
    num_feats = [c for c in num_feats if c in df.columns]
    cat_feats = [c for c in cat_feats if c in df.columns]

    num_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", num_pipeline, num_feats),
            ("cat", cat_pipeline, cat_feats),
        ],
        remainder="drop",
    )

    return preprocessor, num_feats + cat_feats


def train():
    print("--- Starting Training Job (with FE & CV) ---")
    df = _load_data(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    # target
    if "Churn" in df.columns:
        y = df["Churn"]
    elif "Churn Value" in df.columns:
        y = df["Churn Value"]
    else:
        raise KeyError("Target column Churn or Churn Value not found")

    X = df.drop(
        columns=[
            c
            for c in ["customerID", "CustomerID", "Churn", "Churn Value"]
            if c in df.columns
        ]
    )

    preprocessor, feat_list = _feature_pipeline(X)

    # pipeline with classifier
    pipeline = Pipeline(
        [("pre", preprocessor), ("clf", RandomForestClassifier(random_state=42))]
    )

    # cross-validation + grid search
    param_grid = {
        "clf__n_estimators": [50, 100],
        "clf__max_depth": [None, 10, 20],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)

    search.fit(X, y)

    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    # Save best estimator (includes preprocessor)
    save_path = MODEL_DIR / "churn_model.joblib"
    joblib.dump(search.best_estimator_, save_path)
    print(f"Model pipeline saved to: {save_path}")

    # Evaluate on a hold-out split for reporting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    best = search.best_estimator_
    y_proba = best.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"Hold-out ROC-AUC: {auc:.4f}")


if __name__ == "__main__":
    train()
