import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.db.session import get_db
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.db.models import Base

SQLALCHEMY_DATABASE_URL = "sqlite:///./tests/test_rate_limiting.db"
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
    # Force enable limiter for rate limiting tests
    app.state.limiter.enabled = True
    yield
    app.state.limiter.enabled = False
    app.dependency_overrides.clear()

client = TestClient(app)
API_KEY = "test-api-key"
HEADERS = {"X-API-Key": API_KEY}

def test_rate_limiting_triggered():
    # Send 11 mock upload requests to /api/v1/datasets/upload.
    # The limit is 10/minute, so the 11th request must trigger a 429.
    responses = []
    for _ in range(11):
        resp = client.post(
            "/api/v1/datasets/upload",
            headers=HEADERS,
            files={"file": ("test.csv", b"col1,col2\n1,2", "text/csv")}
        )
        responses.append(resp)
        
    # The first 10 requests should pass the rate limiter (they might have 200 or parsing error, but NOT 429)
    for r in responses[:10]:
        assert r.status_code != 429
        
    # The 11th request must be rate limited with HTTP 429
    assert responses[10].status_code == 429
    assert "Rate limit" in responses[10].text
