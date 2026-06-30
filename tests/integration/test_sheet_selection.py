import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import io
import json
from backend.app.main import app

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.db.models import Base, Dataset, Profile
from backend.app.db.session import get_db

# Mock DB setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_sheet.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

@pytest.fixture(autouse=True)
def setup_db_override():
    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.clear()

@pytest.fixture(autouse=True)
def clean_db():
    db = TestingSessionLocal()
    db.query(Profile).delete()
    db.query(Dataset).delete()
    db.commit()
    db.close()

client = TestClient(app)
API_KEY = "test-api-key"
HEADERS = {"X-API-Key": API_KEY}

@patch("backend.app.api.v1.datasets.storage")
def test_upload_excel_multiple_sheets_and_select(mock_storage):
    db = next(override_get_db())
    
    # 1. Create a mock Excel file with 2 sheets
    excel_buf = io.BytesIO()
    df1 = pd.DataFrame({"colA": [1, 2], "colB": [3, 4]})
    df2 = pd.DataFrame({"colX": [10, 20], "colY": [30, 40]})
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        df1.to_excel(writer, sheet_name="Sheet1", index=False)
        df2.to_excel(writer, sheet_name="Sheet2", index=False)
    excel_bytes = excel_buf.getvalue()
    
    # 2. Upload file
    mock_storage.upload_file.return_value = None
    file = {"file": ("test_multi.xlsx", excel_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    
    resp = client.post("/api/v1/datasets/upload", files=file, headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["requires_sheet_selection"] is True
    assert data["sheets"] == ["Sheet1", "Sheet2"]
    assert data["row_count"] is None
    assert data["col_count"] is None
    
    dataset_id = data["dataset_id"]
    
    # Verify DB state: dataset should be created but with null rows/cols
    dataset_in_db = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    assert dataset_in_db is not None
    assert dataset_in_db.row_count is None
    assert dataset_in_db.filename == "test_multi.xlsx"
    
    # Mock download return for the select-sheet call
    mock_storage.download_file.return_value = excel_bytes
    
    # 3. Call select-sheet
    payload = {"sheet_name": "Sheet2"}
    resp_select = client.post(f"/api/v1/datasets/{dataset_id}/select-sheet", json=payload, headers=HEADERS)
    assert resp_select.status_code == 200
    
    select_data = resp_select.json()
    assert select_data["requires_sheet_selection"] is False
    assert select_data["row_count"] == 2
    assert select_data["col_count"] == 2
    
    # Verify DB update: dataset should now point to parquet format and have row_count set
    db.refresh(dataset_in_db)
    assert dataset_in_db.row_count == 2
    assert dataset_in_db.col_count == 2
    assert dataset_in_db.filename == "test_multi.parquet"
    assert dataset_in_db.r2_path.endswith(".parquet")
