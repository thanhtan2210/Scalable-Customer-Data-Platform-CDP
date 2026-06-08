import pandas as pd
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ensure the filename matches your local file (Telco_customer_churn.xlsx)
INPUT_FILE = os.path.join(BASE_DIR, "data", "raw", "Telco_customer_churn.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "parquet", "raw")


def convert():
    print(f"Reading from {INPUT_FILE}...")

    try:
        # Read Excel file
        df = pd.read_excel(INPUT_FILE, engine="openpyxl")
    except FileNotFoundError:
        print(f"Error: File not found at {INPUT_FILE}")
        return

    # --- FIX SECTION ---
    # Find the Total Charges column (may have whitespace in header)
    # Cast to numeric; invalid values (e.g., spaces) become NaN

    target_col = "Total Charges"
    # Check whether the column header is 'Total Charges' or 'TotalCharges'
    if target_col not in df.columns and "TotalCharges" in df.columns:
        target_col = "TotalCharges"

    if target_col in df.columns:
        print(f"Processing column: {target_col}...")
        # errors='coerce' will turn invalid strings/spaces into NaN
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        # Fill NaNs with 0
        df[target_col] = df[target_col].fillna(0)
    # -------------------------------

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save Parquet
    output_path = os.path.join(OUTPUT_DIR, "telco_churn.parquet")
    try:
        df.to_parquet(output_path, index=False)
        print(f"Success! Saved to {output_path}")
    except Exception as e:
        print(f"Error saving parquet: {e}")


if __name__ == "__main__":
    convert()
