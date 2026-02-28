import pandas as pd
import os

# Paths setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'parquet',
                          'raw', 'telco_churn.parquet')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'features')


def run_job():
    print("--- Starting ETL Job (Fix for IBM Dataset) ---")

    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Input file not found at: {INPUT_PATH}")
        return

    # 1. Read data
    df = pd.read_parquet(INPUT_PATH)
    print(f"Original columns: {list(df.columns)}")

    # 2. Normalize column names
    # Trim whitespace and lowercase: 'Churn Value' -> 'churnvalue'
    df.columns = [c.strip().lower().replace(' ', '') for c in df.columns]

    # 3. Handle target (Churn)
    if 'churnvalue' in df.columns:
        # If 'Churn Value' (1/0) exists, use it directly
        print("Found 'churnvalue', renaming to 'Churn'...")
        df['Churn'] = df['churnvalue'].fillna(0).astype(int)
    elif 'churnlabel' in df.columns:
        # If only 'Churn Label' (Yes/No)
        print("Found 'churnlabel', mapping Yes/No and renaming to 'Churn'...")
        df['Churn'] = df['churnlabel'].map(
            {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}).fillna(0).astype(int)
    elif 'churn' in df.columns:
        # If already named 'Churn'
        df['Churn'] = df['churn'].map(
            {'Yes': 1, 'No': 0, 1: 1, 0: 0}).fillna(0).astype(int)
    else:
        print("WARNING: No churn information column found (Label or Value)!")

    # 4. Handle other features
    # Map column names from Excel to canonical names required by the model
    # (Note: Your dataset may use different column names; this code tries to cover cases)

    # Total Charges
    if 'totalcharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(
            df['totalcharges'], errors='coerce').fillna(0)

    # Monthly Charges
    if 'monthlycharges' in df.columns:
        df['MonthlyCharges'] = pd.to_numeric(
            df['monthlycharges'], errors='coerce').fillna(0)

    # Tenure (your file may name it 'tenuremonths')
    if 'tenuremonths' in df.columns:
        df['tenure'] = df['tenuremonths'].astype(int)
    elif 'tenure' in df.columns:
        df['tenure'] = df['tenure'].astype(int)

    # Rename customerID to canonical name
    if 'customerid' in df.columns:
        df['customerID'] = df['customerid']

    # 5. Feature Selection
    required_cols = ['customerID', 'tenure',
                     'MonthlyCharges', 'TotalCharges', 'Churn']

    # Keep only existing columns
    final_cols = [c for c in required_cols if c in df.columns]

    print(f"Final columns to save: {final_cols}")

    if 'Churn' not in final_cols:
        print("CRITICAL: 'Churn' column still missing. Check source column names!")
        return

    df_clean = df[final_cols]

    # 6. Save file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, 'customer_features.parquet')
    df_clean.to_parquet(output_file, index=False)
    print(f"SUCCESS! Saved {len(df_clean)} rows to: {output_file}")


if __name__ == "__main__":
    run_job()
