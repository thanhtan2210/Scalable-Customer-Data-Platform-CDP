"""Schema validation for raw and processed telco data.

This module defines Pandera schemas for strict validation of:
- Raw data: ensure types, constraints, and enums before processing
- Processed data: ensure feature engineering results are valid
"""

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check

# Schema for raw input data
RawTelcoSchema = DataFrameSchema(
    {
        "CustomerID": Column(str, unique=True, nullable=False),
        "Gender": Column(str, Check.isin(["Male", "Female"]), nullable=False),
        "SeniorCitizen": Column(pa.Int, nullable=False, coerce=True),
        "Partner": Column(str, Check.isin(["Yes", "No"]), nullable=False),
        "Dependents": Column(str, Check.isin(["Yes", "No"]), nullable=False),
        "tenure": Column(
            pa.Int, Check.greater_than_or_equal_to(0), nullable=False, coerce=True
        ),
        "PhoneService": Column(str, Check.isin(["Yes", "No"]), nullable=False),
        "InternetService": Column(
            str, Check.isin(["No", "DSL", "Fiber Optic", "Cable"]), nullable=False
        ),
        "MonthlyCharges": Column(
            float, Check.greater_than_or_equal_to(0), nullable=False, coerce=True
        ),
        "TotalCharges": Column(
            float, Check.greater_than_or_equal_to(0), nullable=True, coerce=True
        ),
        "Churn": Column(pa.Int, nullable=False, coerce=True),
    },
    strict=False,
    coerce=True,
)


# Schema for processed features (after ETL)
ProcessedFeatureSchema = DataFrameSchema(
    {
        "CustomerID": Column(str, unique=True, nullable=False),
        "Gender": Column(int, Check.isin([0, 1]), nullable=False),
        "SeniorCitizen": Column(int, Check.isin([0, 1]), nullable=False),
        "Partner": Column(int, Check.isin([0, 1]), nullable=False),
        "Dependents": Column(int, Check.isin([0, 1]), nullable=False),
        "tenure": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
        "MonthlyCharges": Column(
            float, Check.greater_than_or_equal_to(0), nullable=False
        ),
        "TotalCharges": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
        "Churn": Column(int, Check.isin([0, 1]), nullable=False),
        "tenure_bin": Column(str, nullable=True, required=False),
        "monthly_bin": Column(str, nullable=True, required=False),
        "cltv_bin": Column(str, nullable=True, required=False),
    },
    strict=False,
    coerce=True,
)


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Validate raw data against RawTelcoSchema."""
    try:
        return RawTelcoSchema.validate(df)
    except pa.errors.SchemaError as e:
        raise ValueError(f"Raw schema validation failed:\n{e}") from e


def validate_processed(df: pd.DataFrame) -> pd.DataFrame:
    """Validate processed data against ProcessedFeatureSchema."""
    try:
        return ProcessedFeatureSchema.validate(df)
    except pa.errors.SchemaError as e:
        raise ValueError(f"Processed schema validation failed:\n{e}") from e


def get_schema_report(df: pd.DataFrame, schema: DataFrameSchema) -> dict:
    """Generate a detailed report of what's invalid in the DataFrame."""
    report = {"valid": True, "errors": [], "column_stats": {}}

    for col_name, col in schema.columns.items():
        if col_name not in df.columns:
            report["valid"] = False
            report["errors"].append(f"Missing required column: {col_name}")
            continue

        col_data = df[col_name]
        n_rows = len(df)
        null_count = int(col_data.isnull().sum())

        stats = {
            "dtype": str(col_data.dtype),
            "null_count": null_count,
            "null_pct": float(null_count / n_rows) if n_rows > 0 else 0.0,
            "unique_count": int(col_data.nunique()),
        }

        # Check if checks attribute exists and is not None
        checks = getattr(col, "checks", [])
        if checks:
            for check in checks:
                try:
                    check(col_data)
                except Exception as e:
                    report["valid"] = False
                    report["errors"].append(f"Column {col_name} check failed: {str(e)}")

        report["column_stats"][col_name] = stats

    return report
