import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Paths setup
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

# MinIO paths
# IMPORTANT: Pandas uses s3:// (not s3a://)
INPUT_PATH = "s3://datalake/processed/features"
MODEL_DIR = os.path.join(BASE_DIR, 'models')


def train():
    print("--- Starting Training Job (MinIO Version) ---")

    # 1. Load data directly from MinIO
    try:
        print(f"🚀 Reading data from MinIO: {INPUT_PATH}")

        # Pandas uses s3fs to read S3 via storage_options
        df = pd.read_parquet(
            INPUT_PATH,
            storage_options={
                "key": "admin",
                "secret": "password",
                "client_kwargs": {"endpoint_url": "http://localhost:9000"}
            }
        )
        print(f"✅ Loaded {len(df)} rows.")
    except Exception as e:
        print(f"❌ Error reading file from MinIO: {e}")
        print("💡 Hint: Is Docker MinIO running?")
        print("💡 Hint: Has the Spark job written to 'datalake/processed/features'?")
        return

    # 2. Prepare X, y
    if 'Churn' not in df.columns:
        print(
            f"ERROR: Column 'Churn' not found. Available columns: {list(df.columns)}")
        return

    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn']

    print(f"Features used for training: {list(X.columns)}")

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # 4. Train
    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {acc:.4f}")

    # 6. Save model locally (can upgrade to MLflow later)
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, 'churn_model.joblib')
    joblib.dump(model, save_path)
    print(f"💾 Model saved locally to: {save_path}")


if __name__ == "__main__":
    train()
