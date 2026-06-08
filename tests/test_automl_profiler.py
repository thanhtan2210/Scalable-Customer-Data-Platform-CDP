import pandas as pd
import pytest
from src.automl.data_profiler import DataProfiler

def test_data_profiler_with_bank_mock():
    # Mocking a subset of the Bank Churn dataset (Need >10 rows for integer numericals)
    data = {
        "RowNumber": list(range(1, 13)),
        "CustomerId": [f"156346{i:02d}" for i in range(12)],
        "Surname": [f"Name_{i}" for i in range(12)],
        "CreditScore": [619, 608, 502, 699, 850, 700, 650, 550, 800, 750, 620, 610],
        "Geography": ["France", "Spain", "France", "France", "Spain", "Germany", "France", "Spain", "Germany", "France", "Spain", "Germany"],
        "Gender": ["Female"] * 12, # Making it constant for test
        "Age": [42, 41, 42, 39, 43, 25, 30, 50, 60, 35, 28, 45],
        "Balance": [0.00, 83807.86, 159660.80, 0.00, 125510.82, 0.0, 50000.0, 0.0, 100000.0, 20000.0, 0.0, 30000.0],
        "NumOfProducts": [1, 1, 3, 2, 1, 2, 1, 3, 4, 1, 2, 1], # Numeric but low cardinality -> categorical
        "HasCrCard": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0], # Binary -> categorical
        "IsActiveMember": [1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0], # Binary -> categorical
        "EstimatedSalary": [101348.88, 112542.58, 113931.57, 93826.63, 79084.10, 50000.0, 60000.0, 70000.0, 80000.0, 90000.0, 100000.0, 110000.0],
        "Exited": [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1] # Target
    }
    df = pd.DataFrame(data)

    profiler = DataProfiler()
    results = profiler.profile_data(df, target_col="Exited")

    assert results["total_rows"] == 12
    assert results["target"] == "Exited"
    
    # Check Constant
    assert "Gender" in results["constant_columns"]

    # Check IDs / High Cardinality
    assert "RowNumber" not in results["numerical_features"] # Should be caught as ID or similar if cardinality is 1.0, but here it's numeric. Let's see how our logic handles it. Actually, RowNumber will be unique_ratio = 1.0 > 0.9.
    assert "CustomerId" in results["id_columns"]
    assert "Surname" in results["id_columns"]

    # Check Numerical
    assert "CreditScore" in results["numerical_features"]
    assert "Balance" in results["numerical_features"]
    assert "EstimatedSalary" in results["numerical_features"]
    assert "Age" in results["numerical_features"]

    # Check Categorical (Numeric but low cardinality or Strings with low cardinality)
    assert "Geography" in results["categorical_features"]
    assert "HasCrCard" in results["categorical_features"]
    assert "IsActiveMember" in results["categorical_features"]
    assert "NumOfProducts" in results["categorical_features"]

    # Ensure Target is not in features
    assert "Exited" not in results["numerical_features"]
    assert "Exited" not in results["categorical_features"]
