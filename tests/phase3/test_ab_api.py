import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

HEADERS = {"X-API-Key": "test-api-key"}
client = TestClient(app)

def test_ab_assign_variant():
    # 1. Test standard assign variant
    payload = {
        "customer_id": "cust_12345",
        "ratio": 0.5
    }
    resp = client.post("/api/v1/ab/assign", headers=HEADERS, json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["customer_id"] == "cust_12345"
    assert data["ab_group"] in ["A", "B"]

    # 2. Test missing customer_id returns 400
    payload_bad = {
        "customer_id": "",
        "ratio": 0.5
    }
    resp = client.post("/api/v1/ab/assign", headers=HEADERS, json=payload_bad)
    assert resp.status_code == 400

def test_ab_log_exposure():
    payload = {
        "customer_id": "cust_12345",
        "ab_group": "A",
        "event": "click_buy"
    }
    resp = client.post("/api/v1/ab/log_exposure", headers=HEADERS, json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["stored"] in ["db", "jsonl"]

def test_ab_health():
    resp = client.get("/api/v1/ab/health", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
