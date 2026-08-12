import uuid
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_register_new_user():
    email = f"test_{uuid.uuid4().hex[:6]}@example.com"
    r = client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "password123",
            "full_name": "Test User",
        },
    )
    assert r.status_code == 201
    data = r.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["user"]["email"] == email


def test_register_duplicate_email():
    email = f"dup_{uuid.uuid4().hex[:6]}@example.com"
    client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "password123"},
    )
    r = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "password123"},
    )
    assert r.status_code == 400


def test_login_valid():
    email = f"login_{uuid.uuid4().hex[:6]}@example.com"
    client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "password123"},
    )
    r = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "password123"},
    )
    assert r.status_code == 200
    assert "access_token" in r.json()


def test_login_wrong_password():
    email = f"login_wrong_{uuid.uuid4().hex[:6]}@example.com"
    client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "password123"},
    )
    r = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "wrongpassword"},
    )
    assert r.status_code == 401


def test_refresh_token():
    email = f"refresh_{uuid.uuid4().hex[:6]}@example.com"
    reg = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "password123"},
    )
    refresh_token = reg.json()["refresh_token"]
    r = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert r.status_code == 200
    assert "access_token" in r.json()


def test_get_me():
    email = f"me_{uuid.uuid4().hex[:6]}@example.com"
    reg = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "password123"},
    )
    token = reg.json()["access_token"]
    r = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    assert r.json()["email"] == email


def test_protected_endpoint_no_token():
    r = client.get("/api/v1/monitoring/metrics")
    assert r.status_code == 401


def test_change_password():
    email = f"pwd_{uuid.uuid4().hex[:6]}@example.com"
    reg = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "oldpassword1"},
    )
    token = reg.json()["access_token"]
    r = client.post(
        "/api/v1/auth/change-password",
        headers={"Authorization": f"Bearer {token}"},
        json={"current_password": "oldpassword1", "new_password": "newpassword1"},
    )
    assert r.status_code == 200
