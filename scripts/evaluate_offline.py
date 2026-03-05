import json
import os
from pathlib import Path

try:
    import joblib
except Exception:
    joblib = None
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "cleaned_telco.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess(df):
    # Minimal preprocessing: select numeric + a few categorical features
    df = df.copy()
    df['Churn'] = df['Churn Value']
    features = [
        'Tenure Months',
        'Monthly Charges',
        'Total Charges',
        'Churn Score',
        'CLTV',
    ]
    # fill numeric missing
    for c in features:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # add simple categorical encoding: Contract
    cat = df[['Contract']].fillna('NA')
    try:
        enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
    except TypeError:
        # scikit-learn >=1.2 uses sparse_output
        enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_ohe = enc.fit_transform(cat)
    cat_cols = [f"Contract__{v}" for v in enc.categories_[0]]

    X_num = df[features].values
    X = np.hstack([X_num, cat_ohe])
    y = df['Churn'].astype(int).values
    return X, y, enc, features, cat_cols


def precision_at_k(y_true, y_score, k):
    # k is number of top scored customers
    idx = np.argsort(y_score)[::-1][:k]
    return y_true[idx].sum() / k


def run():
    df = load_data(RAW_CSV)
    X, y, enc, features, cat_cols = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = None
    model_used = 'trained_in_memory'
    if MODEL_PATH.exists() and joblib is not None:
        try:
            loaded = joblib.load(MODEL_PATH)
            # test predict on a small slice to confirm compatibility
            try:
                _ = loaded.predict_proba(X_test[:5])
                model = loaded
                model_used = str(MODEL_PATH)
            except Exception:
                model = None
        except Exception:
            model = None

    if model is None:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        if joblib is not None:
            try:
                joblib.dump(model, MODEL_PATH)
                model_used = 'trained_and_saved'
            except Exception:
                model_used = 'trained_in_memory'

    y_proba = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, y_proba))

    # precision@k (use k = top 10% of test set)
    k = max(1, int(0.1 * len(y_test)))
    prec_at_k = float(precision_at_k(y_test, y_proba, k))

    # uplift estimate: assume outreach success rate
    outreach_success_rate = 0.3
    avg_cltv = float(df['CLTV'].mean())
    saved_customers = prec_at_k * k * outreach_success_rate
    estimated_benefit = saved_customers * avg_cltv

    # cost assumptions
    cost_per_outreach = 5.0
    outreach_cost = k * cost_per_outreach
    estimated_roi = (estimated_benefit - outreach_cost) / \
        max(1.0, outreach_cost)

    summary = {
        'model_used': model_used,
        'auc': auc,
        'precision_at_k': prec_at_k,
        'k': k,
        'outreach_success_rate': outreach_success_rate,
        'avg_cltv': avg_cltv,
        'saved_customers_estimate': saved_customers,
        'estimated_benefit': estimated_benefit,
        'outreach_cost': outreach_cost,
        'estimated_roi': estimated_roi,
    }

    with open(REPORTS / 'offline_evaluation.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    with open(REPORTS / 'offline_evaluation.txt', 'w', encoding='utf-8') as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print('Offline evaluation complete. Summary:')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    run()
