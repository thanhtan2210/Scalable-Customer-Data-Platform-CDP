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
from backend.app.core.profiler.target_analysis import TargetAnalysis, CompositeTargetConfig, SynthesisStrategy

# Mock DB setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_confirm.db"
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

client = TestClient(app)
API_KEY = "test-api-key"
HEADERS = {"X-API-Key": API_KEY}

@patch("backend.app.api.v1.datasets.storage")
def test_confirm_composite_confirmed_false(mock_storage):
    db = next(override_get_db())
    
    # 1. Create mock dataset and profile in db
    dataset_id = "test-ds-1"
    new_dataset = Dataset(
        id=dataset_id,
        user_id="default_user",
        filename="test.csv",
        r2_path=f"raw/default_user/{dataset_id}/test.csv",
        status="profiled"
    )
    
    mock_target_analysis = TargetAnalysis(
        recommended_target="Churn",
        candidate_targets=[],
        churn_column_group=[],
        recommended_auxiliary=[],
        leakage_suspects=[],
        composite_target=CompositeTargetConfig(
            strategy=SynthesisStrategy.WEIGHTED,
            source_columns=["col_aux"],
            cpi_column_name="cpi_score",
            requires_confirmation=True
        )
    )
    
    new_profile = Profile(
        dataset_id=dataset_id,
        profiles_json=[],
        suggested_target=mock_target_analysis.json()
    )
    
    db.merge(new_dataset)
    db.merge(new_profile)
    db.commit()
    
    # 2. Call endpoint with confirmed = False
    payload = {
        "confirmed": False
    }
    
    resp = client.post(f"/api/v1/datasets/{dataset_id}/confirm-composite", json=payload, headers=HEADERS)
    
    assert resp.status_code == 200
    assert resp.json()["cpi_attached"] is False
    assert resp.json()["composite_target"] is None
    
    # Verify DB update
    updated_profile = db.query(Profile).filter(Profile.dataset_id == dataset_id).first()
    updated_target_analysis = TargetAnalysis.parse_raw(updated_profile.suggested_target)
    assert updated_target_analysis.composite_target is None


@patch("backend.app.api.v1.datasets.storage")
def test_confirm_composite_confirmed_true(mock_storage):
    db = next(override_get_db())
    
    dataset_id = "test-ds-2"
    new_dataset = Dataset(
        id=dataset_id,
        user_id="default_user",
        filename="test.csv",
        r2_path=f"raw/default_user/{dataset_id}/test.csv",
        status="profiled"
    )
    
    # Profile with auxiliary cols
    from backend.app.core.profiler.target_analysis import ChurnColumnGroupItem, GroupRole
    mock_target_analysis = TargetAnalysis(
        recommended_target="Churn",
        candidate_targets=[],
        churn_column_group=[
            ChurnColumnGroupItem(name="Churn", correlation_with_target=1.0, group_role=GroupRole.PRIMARY),
            ChurnColumnGroupItem(name="col_aux1", correlation_with_target=0.7, group_role=GroupRole.AUXILIARY),
            ChurnColumnGroupItem(name="col_aux2", correlation_with_target=0.6, group_role=GroupRole.AUXILIARY)
        ],
        recommended_auxiliary=["col_aux1", "col_aux2"],
        leakage_suspects=[],
        composite_target=CompositeTargetConfig(
            strategy=SynthesisStrategy.WEIGHTED,
            source_columns=["col_aux1", "col_aux2"],
            cpi_column_name="cpi_score",
            requires_confirmation=True
        )
    )
    
    new_profile = Profile(
        dataset_id=dataset_id,
        profiles_json=[],
        suggested_target=mock_target_analysis.json()
    )
    
    db.merge(new_dataset)
    db.merge(new_profile)
    db.commit()
    
    # Mock R2 download returning a small CSV
    df = pd.DataFrame({
        "Churn": [0, 1, 0, 1],
        "col_aux1": [1.0, 5.0, 2.0, 6.0],
        "col_aux2": [0.5, 0.9, 0.4, 0.8]
    })
    mock_storage.download_file.return_value = df.to_csv(index=False).encode("utf-8")
    mock_storage.upload_file.return_value = None
    
    # 2. Call endpoint with confirmed = True
    payload = {
        "confirmed": True,
        "selected_strategy": "WEIGHTED",
        "selected_source_columns": ["col_aux1", "col_aux2"]
    }
    
    resp = client.post(f"/api/v1/datasets/{dataset_id}/confirm-composite", json=payload, headers=HEADERS)
    
    assert resp.status_code == 200
    data = resp.json()
    assert data["cpi_attached"] is True
    assert data["composite_target"]["strategy"] == "WEIGHTED"
    assert data["composite_target"]["requires_confirmation"] is False
    assert len(data["composite_target"]["weights"]) == 2
    
    # Verify R2 upload was called
    mock_storage.upload_file.assert_called_once()
    
    # Verify DB update
    updated_dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    assert updated_dataset.r2_path == f"processed/{dataset_id}/with_cpi.parquet"
    assert updated_dataset.filename == "with_cpi.parquet"
    
    updated_profile = db.query(Profile).filter(Profile.dataset_id == dataset_id).first()
    assert any(p["name"] == "cpi_score" for p in updated_profile.profiles_json)
