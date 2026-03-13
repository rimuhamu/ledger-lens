import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
from src.main import app
from src.auth import get_current_user
from src.models import User

# ============================================================================
# BASIC ENDPOINTS
# ============================================================================

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "LedgerLens Analyst" in response.json()["status"]

# ============================================================================
# AUTHENTICATION ROUTES
# ============================================================================

@patch('src.api.routes.auth.turso_db')
@patch('src.api.routes.auth.hash_password')
@patch('src.api.routes.auth.create_access_token')
def test_register_success(mock_create_token, mock_hash_password, mock_turso_db, client):
    """Test successful user registration"""
    # Setup mocks
    mock_hash_password.return_value = "hashed_password"
    mock_create_token.return_value = "test_token"
    mock_turso_db.get_user_by_email.return_value = None
    mock_turso_db.create_user.return_value = User(
        id="user123",
        email="test@example.com",
        password="hashed_password",
        created_at="2023-01-01T00:00:00"
    )
    
    response = client.post("/auth/register", json={
        "email": "test@example.com",
        "password": "password123"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["access_token"] == "test_token"
    assert data["user"]["email"] == "test@example.com"

@patch('src.api.routes.auth.turso_db')
def test_register_duplicate_email(mock_turso_db, client):
    """Test registration with existing email"""
    mock_turso_db.get_user_by_email.return_value = User(
        id="existing_user",
        email="test@example.com",
        password="hashed",
        created_at="2023-01-01T00:00:00"
    )
    
    response = client.post("/auth/register", json={
        "email": "test@example.com",
        "password": "password123"
    })
    
    assert response.status_code == 409
    assert "already registered" in response.json()["detail"]

@patch('src.api.routes.auth.turso_db')
def test_register_missing_email(mock_turso_db, client):
    """Test registration with missing email"""
    response = client.post("/auth/register", json={
        "email": "",
        "password": "password123"
    })
    
    assert response.status_code == 400
    assert "required" in response.json()["detail"]

@patch('src.api.routes.auth.turso_db')
def test_register_short_password(mock_turso_db, client):
    """Test registration with password too short"""
    mock_turso_db.get_user_by_email.return_value = None
    
    response = client.post("/auth/register", json={
        "email": "test@example.com",
        "password": "short"
    })
    
    assert response.status_code == 400
    assert "at least 8 characters" in response.json()["detail"]

@patch('src.api.routes.auth.turso_db')
@patch('src.api.routes.auth.verify_password')
@patch('src.api.routes.auth.create_access_token')
def test_login_success(mock_create_token, mock_verify_password, mock_turso_db, client):
    """Test successful login"""
    mock_verify_password.return_value = True
    mock_create_token.return_value = "test_token"
    mock_turso_db.get_user_by_email.return_value = User(
        id="user123",
        email="test@example.com",
        password="hashed_password",
        created_at="2023-01-01T00:00:00"
    )
    
    response = client.post("/auth/login", json={
        "email": "test@example.com",
        "password": "password123"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["access_token"] == "test_token"
    assert data["user"]["email"] == "test@example.com"

@patch('src.api.routes.auth.turso_db')
@patch('src.api.routes.auth.verify_password')
def test_login_invalid_password(mock_verify_password, mock_turso_db, client):
    """Test login with invalid password"""
    mock_verify_password.return_value = False
    mock_turso_db.get_user_by_email.return_value = User(
        id="user123",
        email="test@example.com",
        password="hashed_password",
        created_at="2023-01-01T00:00:00"
    )
    
    response = client.post("/auth/login", json={
        "email": "test@example.com",
        "password": "wrongpassword"
    })
    
    assert response.status_code == 401
    assert "Invalid" in response.json()["detail"]

@patch('src.api.routes.auth.turso_db')
def test_login_nonexistent_user(mock_turso_db, client):
    """Test login with non-existent user"""
    mock_turso_db.get_user_by_email.return_value = None
    
    response = client.post("/auth/login", json={
        "email": "nonexistent@example.com",
        "password": "password123"
    })
    
    assert response.status_code == 401
    assert "Invalid" in response.json()["detail"]

@pytest.mark.asyncio
async def test_get_me_success(client):
    """Test getting current user profile"""
    # Uses default override returning mock_user_instance
    
    response = client.get("/auth/me")
    
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "test@example.com"
    assert data["id"] == "user123"

def test_get_me_unauthorized(client):
    """Test getting profile without authentication"""
    def raise_unauthorized():
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    app.dependency_overrides[get_current_user] = raise_unauthorized
    try:
        response = client.get("/auth/me")
    finally:
        # Restore mock user
        # In fixture setup we used a specific lambda, we can't easily restore 'override_get_current_user' 
        # because we don't have it. 
        # But `client` fixture uses app.dependency_overrides.clear() at the end.
        pass
    
    # Assert on the response captured inside the try block
    assert response.status_code == 401
