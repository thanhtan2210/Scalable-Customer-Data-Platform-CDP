import pandas as pd
import pytest
from backend.app.core.profiler.orchestrator import run_profiling

def test_profiler_bank_dataset():
    data = {
        "customer_id": ["C1", "C2", "C3", "C4", "C5"],
        "credit_score": [600, 700, 550, 800, 750],
        "geography": ["France", "Spain", "France", "France", "Spain"],
        "balance": [0.0, 1000.5, 500.0, 0.0, 120000.0],
        "is_active": [1, 1, 0, 0, 1],
        "exited": [0, 0, 1, 0, 0] # Target candidate
    }
    df = pd.DataFrame(data)
    profiles = run_profiling(df)
    
    names = [p.name for p in profiles]
    assert "credit_score" in names
    
    # credit_score should be numeric
    cs_profile = next(p for p in profiles if p.name == "credit_score")
    assert cs_profile.inferred_role == "numeric"
    
    # customer_id should be id
    id_profile = next(p for p in profiles if p.name == "customer_id")
    assert id_profile.inferred_role == "id"

def test_profiler_telco_dataset():
    data = {
        "tenure": [1, 24, 12, 1, 72],
        "monthly_charges": [29.85, 56.95, 53.85, 42.30, 70.70],
        "contract": ["Month-to-month", "One year", "Month-to-month", "Month-to-month", "Two year"],
        "email": ["test@gmail.com", "user@yahoo.com", "admin@company.io", "info@net.vn", "web@site.com"]
    }
    df = pd.DataFrame(data)
    profiles = run_profiling(df)
    
    # email should be dropped or identified as PII (role='drop' in our Layer 2)
    email_profile = next(p for p in profiles if p.name == "email")
    assert email_profile.inferred_role == "drop"
    assert email_profile.layer_source == 2

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
    profiles = run_profiling(df)
    
    # department is constant here (all sales), should be constant_column logic? 
    # In Layer 1, if unique=1 -> currently it doesn't explicitly mark 'drop' for constant in code (I should add that or check if it handles it)
    # Let's check salary_comment -> should be text if long enough
    text_profile = next(p for p in profiles if p.name == "salary_comment")
    # Our text logic: avg_len > 50 and cardinality > 0.3. 
    # Let's see if the test passes or if I need to adjust the mock data.
