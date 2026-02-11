import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock
from src.main import app
from src.api.dependencies import get_vector_store, get_object_store, get_analysis_service
from src.core.services.analysis_service import AnalysisService
from src.infrastructure.storage.vector.base import VectorStore
from src.infrastructure.storage.object.base import ObjectStore
from src.models import User
from src.auth import get_current_user
from fastapi import HTTPException

client = TestClient(app)

# Mocks for overrides
mock_vector_store_instance = MagicMock(spec=VectorStore)
mock_object_store_instance = MagicMock(spec=ObjectStore)
mock_analysis_service_instance = MagicMock(spec=AnalysisService)

def override_get_vector_store():
    return mock_vector_store_instance

def override_get_object_store():
    return mock_object_store_instance

def override_get_analysis_service():
    return mock_analysis_service_instance

app.dependency_overrides[get_vector_store] = override_get_vector_store
app.dependency_overrides[get_object_store] = override_get_object_store
app.dependency_overrides[get_analysis_service] = override_get_analysis_service

# Auth override
mock_user_instance = User(
    id="user123",
    email="test@example.com",
    password="hashed",
    created_at="2023-01-01T00:00:00"
)

def override_get_current_user():
    return mock_user_instance

app.dependency_overrides[get_current_user] = override_get_current_user

# ============================================================================
# BASIC ENDPOINTS
# ============================================================================

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "LedgerLens Analyst" in response.json()["status"]

# ============================================================================
# AUTHENTICATION ROUTES
# ============================================================================

@patch('src.api.routes.auth.turso_db')
@patch('src.api.routes.auth.hash_password')
@patch('src.api.routes.auth.create_access_token')
def test_register_success(mock_create_token, mock_hash_password, mock_turso_db):
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
def test_register_duplicate_email(mock_turso_db):
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
def test_register_missing_email(mock_turso_db):
    """Test registration with missing email"""
    response = client.post("/auth/register", json={
        "email": "",
        "password": "password123"
    })
    
    assert response.status_code == 400
    assert "required" in response.json()["detail"]

@patch('src.api.routes.auth.turso_db')
def test_register_short_password(mock_turso_db):
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
def test_login_success(mock_create_token, mock_verify_password, mock_turso_db):
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
def test_login_invalid_password(mock_verify_password, mock_turso_db):
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
def test_login_nonexistent_user(mock_turso_db):
    """Test login with non-existent user"""
    mock_turso_db.get_user_by_email.return_value = None
    
    response = client.post("/auth/login", json={
        "email": "nonexistent@example.com",
        "password": "password123"
    })
    
    assert response.status_code == 401
    assert "Invalid" in response.json()["detail"]

@pytest.mark.asyncio
async def test_get_me_success():
    """Test getting current user profile"""
    # Uses default override returning mock_user_instance
    
    response = client.get("/auth/me")
    
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "test@example.com"
    assert data["id"] == "user123"

def test_get_me_unauthorized():
    """Test getting profile without authentication"""
    def raise_unauthorized():
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    app.dependency_overrides[get_current_user] = raise_unauthorized
    try:
        response = client.get("/auth/me")
    finally:
        app.dependency_overrides[get_current_user] = override_get_current_user
    
    # Assert on the response captured inside the try block
    assert response.status_code == 401

# ============================================================================
# DOCUMENT ROUTES
# ============================================================================

@patch('src.api.routes.documents.OpenAIEmbeddings')
def test_upload_document_success(mock_embeddings):
    """Test successful document upload"""
    # Auth handled by dependency override
    
    # Mock embeddings
    mock_embed_instance = MagicMock()
    mock_embed_instance.embed_documents.return_value = [[0.1] * 1536]
    mock_embeddings.return_value = mock_embed_instance
    
    # Mock vector store
    mock_vector_store_instance.upsert.return_value = None
    
    # Mock object store
    mock_object_store_instance.upload_file.return_value = None
    
    # Create a fake PDF file
    from io import BytesIO
    pdf_content = b"%PDF-1.4\n%fake pdf content"
    
    with patch('src.api.routes.documents.PyPDFLoader') as mock_loader:
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [
            MagicMock(page_content="Test content", metadata={"page": 0})
        ]
        mock_loader.return_value = mock_loader_instance
        
        response = client.post(
            "/documents/upload",
            files={"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")},
            data={"ticker": "AAPL"}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "document_id" in data

def test_upload_document_non_pdf():
    """Test upload with non-PDF file"""
    # Auth handled by dependency override
    
    from io import BytesIO
    
    response = client.post(
        "/documents/upload",
        files={"file": ("test.txt", BytesIO(b"text content"), "text/plain")},
        data={"ticker": "AAPL"}
    )
    
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]

def test_list_documents_success():
    """Test listing user's documents"""
    # Auth handled by dependency override
    
    # Mock vector store query response
    mock_match = MagicMock()
    mock_match.metadata = {
        "document_id": "doc123",
        "ticker": "AAPL",
        "filename": "test.pdf",
        "created_at": "2023-01-01T00:00:00",
        "s3_key": "user123/test.pdf"
    }
    
    mock_results = MagicMock()
    mock_results.matches = [mock_match]
    mock_vector_store_instance.query.return_value = mock_results
    
    response = client.get("/documents/")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["document_id"] == "doc123"

def test_list_documents_empty():
    """Test listing documents for user with no documents"""
    # Auth handled by dependency override
    
    mock_results = MagicMock()
    mock_results.matches = []
    mock_vector_store_instance.query.return_value = mock_results
    
    response = client.get("/documents/")
    
    assert response.status_code == 200
    assert response.json() == []

def test_get_document_success():
    """Test getting specific document"""
    # Auth handled by dependency override
    
    mock_match = MagicMock()
    mock_match.metadata = {
        "document_id": "doc123",
        "ticker": "AAPL",
        "filename": "test.pdf",
        "created_at": "2023-01-01T00:00:00",
        "s3_key": "user123/test.pdf"
    }
    
    mock_results = MagicMock()
    mock_results.matches = [mock_match]
    mock_vector_store_instance.query.return_value = mock_results
    
    response = client.get("/documents/doc123")
    
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == "doc123"

def test_get_document_not_found():
    """Test getting non-existent document"""
    # Auth handled by dependency override
    
    mock_results = MagicMock()
    mock_results.matches = []
    mock_vector_store_instance.query.return_value = mock_results
    
    response = client.get("/documents/nonexistent")
    
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]

# ============================================================================
# ANALYSIS ROUTES
# ============================================================================

def test_analyze_document_success():
    """Test successful document analysis"""
    # Auth handled by dependency override
    
    # Mock analysis service
    mock_analysis_service_instance.analyze_document = AsyncMock(return_value={
        "answer": "Test analysis result",
        "is_valid": True,
        "intelligence_hub_data": {"key": "value"}
    })
    
    response = client.post(
        "/analysis/doc123",
        json={"query": "What is the revenue?"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Test analysis result"
    assert data["verification_status"] == "PASS"

def test_analyze_document_validation_fail():
    """Test analysis with validation failure"""
    # Auth handled by dependency override
    
    mock_analysis_service_instance.analyze_document = AsyncMock(return_value={
        "answer": "Invalid result",
        "is_valid": False,
        "intelligence_hub_data": {}
    })
    
    response = client.post(
        "/analysis/doc123",
        json={"query": "What is the revenue?"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["verification_status"] == "FAIL"

def test_analyze_document_error():
    """Test analysis with service error"""
    # Auth handled by dependency override
    
    mock_analysis_service_instance.analyze_document = AsyncMock(
        side_effect=Exception("Analysis failed")
    )
    
    response = client.post(
        "/analysis/doc123",
        json={"query": "What is the revenue?"}
    )
    
    assert response.status_code == 500
    assert "Analysis failed" in response.json()["detail"]
