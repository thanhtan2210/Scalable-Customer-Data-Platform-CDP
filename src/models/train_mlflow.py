import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parents[2]
LOCAL_INPUT = BASE_DIR / "data" / "parquet" / "processed" / "cleaned_telco.parquet"

# MLflow configuration - Use local mlruns directory
MLRUNS_DIR = BASE_DIR / "mlruns"
mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR}")
mlflow.set_experiment("CDP_Churn_Prediction")

def train():
    print("--- Starting Training with MLflow Tracking (Local Filesystem) ---")

    # 1. Load Data
    if not LOCAL_INPUT.exists():
        print(f"❌ Error: Local data not found at {LOCAL_INPUT}")
        return
        
    try:
        print(f"🚀 Reading local data from: {LOCAL_INPUT}")
        df = pd.read_parquet(LOCAL_INPUT)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 2. Prepare Data
    target = 'Churn'
    if target not in df.columns:
        print(f"❌ Error: Column '{target}' not found.")
        return

    # Drop non-numeric or ID columns
    drop_cols = ['CustomerID', 'customerID', 'tenure_bin', 'monthly_bin', 'cltv_bin', 'Churn Label', 'Churn Reason', 'Count', 'Country', 'State', 'City', 'Zip Code', 'Lat Long']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + [target])
    
    # Simple encoding
    X = pd.get_dummies(X)
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. START MLFLOW RUN
    with mlflow.start_run():
        print("🧪 Experiment started...")

        # A. Log Params
        n_estimators = 100
        max_depth = 10
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("data_source", "local_parquet")

        # B. Train
        print(f"Training Random Forest on {len(X_train)} samples...")
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # C. Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Model Accuracy: {acc:.4f}")
        mlflow.log_metric("accuracy", acc)

        # D. Log Model
        print(f"💾 Saving model to MLflow (Local Dir: {MLRUNS_DIR})...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_model"
        )

        print("✨ Done! Information logged to local mlruns folder")


if __name__ == "__main__":
    train()
