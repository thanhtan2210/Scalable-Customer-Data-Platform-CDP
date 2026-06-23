import pandas as pd
import pytest
from backend.app.core.profiler.orchestrator import run_profiling
from backend.app.core.profiler.column_profile import DataRole

def test_profiler_bank_dataset():
    # Generate 25 rows to ensure cardinality ratio <= 0.9 and unique_count > 15
    data = {
        "customer_id": [f"C{i}" for i in range(25)],
        "credit_score": [600 + i for i in range(20)] + [600] * 5,
        "geography": ["France", "Spain", "France", "France", "Spain"] * 5,
        "balance": [0.0, 1000.5, 500.0, 0.0, 120000.0] * 5,
        "is_active": [1, 1, 0, 0, 1] * 5,
        "exited": [0, 0, 1, 0, 0] * 5  # Target candidate
    }
    df = pd.DataFrame(data)
    profiles, _ = run_profiling(df)
    
    names = [p.name for p in profiles]
    assert "credit_score" in names
    
    # credit_score should be NUMERIC
    cs_profile = next(p for p in profiles if p.name == "credit_score")
    assert cs_profile.inferred_role == DataRole.NUMERIC
    
    # customer_id should be ID
    id_profile = next(p for p in profiles if p.name == "customer_id")
    assert id_profile.inferred_role == DataRole.ID

def test_profiler_telco_dataset():
    data = {
        "tenure": [1, 24, 12, 1, 72],
        "monthly_charges": [29.85, 56.95, 53.85, 42.30, 70.70],
        "contract": ["Month-to-month", "One year", "Month-to-month", "Month-to-month", "Two year"],
        "email": ["test@gmail.com", "user@yahoo.com", "admin@company.io", "info@net.vn", "web@site.com"]
    }
    df = pd.DataFrame(data)
    profiles, _ = run_profiling(df)
    
    # email should be dropped/ignored (role='IGNORE' in our Layer 2)
    email_profile = next(p for p in profiles if p.name == "email")
    assert email_profile.inferred_role == DataRole.IGNORE

def test_profiler_hr_dataset():
    data = {
        "employee_id": ["E001", "E002", "E003", "E004", "E005"],
        "satisfaction_level": [0.38, 0.80, 0.11, 0.72, 0.37],
        "last_evaluation": [0.53, 0.86, 0.62, 0.87, 0.52],
        "average_montly_hours": [157, 262, 272, 223, 159],
        "department": ["sales", "sales", "sales", "sales", "sales"], # Constant/Low card
        "salary_comment": ["Bad performance this month", "Excellent work", "N/A", "Steady progress", "Needs improvement and more training"]
    }
    df = pd.DataFrame(data)
    profiles, _ = run_profiling(df)
    
    text_profile = next(p for p in profiles if p.name == "salary_comment")
    assert text_profile is not None
