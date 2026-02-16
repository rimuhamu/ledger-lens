import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock
from src.main import app
from src.api.dependencies import get_vector_store, get_object_store, get_analysis_service
from src.infrastructure.storage.vector.base import VectorStore
from src.infrastructure.storage.object.base import ObjectStore
from src.models import User, Document
from src.auth import get_current_user



@patch('src.api.routes.documents.turso_db')
def test_delete_document_success(mock_turso_db, client, mock_vector_store, mock_object_store):
    """Test successful document deletion"""
    
    # Setup mock document
    mock_doc = Document(
        id="doc123",
        user_id="user123",
        filename="test.pdf",
        ticker="AAPL",
        s3_key="user123/test.pdf",
        created_at="2023-01-01T00:00:00",
        analysis_status="completed"
    )
    mock_turso_db.get_document.return_value = mock_doc
    
    # Execute delete
    response = client.delete("/api/documents/doc123")
    
    # Assertions
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    # Verify DB interactions
    mock_turso_db.get_document.assert_called_with("doc123")
    mock_turso_db.delete_document.assert_called_with("doc123")
    
    # Verify Vector Store interaction
    # Verify Vector Store interaction
    mock_vector_store.delete.assert_called_with(filter={"document_id": "doc123", "user_id": "user123"})
    
    # Verify Object Store interactions
    # Should delete PDF and Analysis JSON
    # Verify Object Store interactions
    # Should delete PDF and Analysis JSON
    mock_object_store.delete_file.assert_any_call("user123/test.pdf")
    mock_object_store.delete_file.assert_any_call("user123/doc123/analysis.json")

@patch('src.api.routes.documents.turso_db')
def test_delete_document_not_found(mock_turso_db, client):
    """Test delete non-existent document"""
    mock_turso_db.get_document.return_value = None
    
    response = client.delete("/api/documents/nonexistent")
    
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]

@patch('src.api.routes.documents.turso_db')
def test_delete_document_unauthorized(mock_turso_db, client):
    """Test delete document belonging to another user"""
    mock_doc = Document(
        id="doc123",
        user_id="other_user", # Different user
        filename="test.pdf",
        ticker="AAPL",
        s3_key="other/test.pdf",
        created_at="2023-01-01T00:00:00",
        analysis_status="completed"
    )
    mock_turso_db.get_document.return_value = mock_doc
    
    response = client.delete("/api/documents/doc123")
    
    assert response.status_code == 403
    assert "Not authorized" in response.json()["detail"]
