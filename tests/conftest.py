import logging
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def set_log_level():
    """Set a default log level for tests so test output is stable."""
    logging.basicConfig(level=logging.ERROR)


@pytest.fixture
def tmp_df():
    """Return a small DataFrame used in tests."""
    # Provide a minimal but schema-compliant DataFrame matching RawTelcoSchema
    return pd.DataFrame(
        {
            "CustomerID": ["C1", "C2", "C3"],
            "Gender": ["Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0],
            "Partner": ["Yes", "No", "Yes"],
            "Dependents": ["No", "No", "Yes"],
            "tenure": [1, 10, 24],
            "PhoneService": ["Yes", "Yes", "No"],
            "InternetService": ["DSL", "Fiber Optic", "No"],
            "MonthlyCharges": [50.0, 75.0, 100.0],
            "TotalCharges": [100.0, 750.0, 2400.0],
            "Churn": [0, 1, 0],
        }
    )


@pytest.fixture
def tmp_csv(tmp_path, tmp_df):
    """Write the tmp_df to a CSV file in tmp_path and return the path."""
    p = tmp_path / "sample.csv"
    tmp_df.to_csv(p, index=False)
    return p
