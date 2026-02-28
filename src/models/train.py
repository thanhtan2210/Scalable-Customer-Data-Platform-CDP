import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH = "s3://datalake/processed/features"
MODEL_DIR = os.path.join(BASE_DIR, 'models')


def train():
    print("--- Starting Training Job ---")
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Data not found at {INPUT_PATH}")
        return

    # 1. Load Data
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} rows.")

    # 2. Prepare X, y
    # Drop customerID since it does not predict churn
    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn']

    print(f"Features: {list(X.columns)}")

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
    print(f"Accuracy: {acc:.4f}")

    # 6. Save Model
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, 'churn_model.joblib')
    joblib.dump(model, save_path)
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    train()
