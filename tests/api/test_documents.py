from unittest.mock import patch, MagicMock
from io import BytesIO

# ============================================================================
# DOCUMENT ROUTES
# ============================================================================

@patch('src.api.routes.documents.OpenAIEmbeddings')
def test_upload_document_success(mock_embeddings, client, mock_vector_store, mock_object_store):
    """Test successful document upload"""
    # Auth handled by dependency override
    
    # Mock embeddings
    mock_embed_instance = MagicMock()
    mock_embed_instance.embed_documents.return_value = [[0.1] * 1536]
    mock_embeddings.return_value = mock_embed_instance
    
    # Mock vector store
    mock_vector_store.upsert.return_value = None
    
    # Mock object store
    mock_object_store.upload_file.return_value = None
    
    # Create a fake PDF file
    pdf_content = b"%PDF-1.4\n%fake pdf content"
    
    with patch('src.api.routes.documents.PyMuPDFLoader') as mock_loader:
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [
            MagicMock(page_content="Test content", metadata={"page": 0})
        ]
        mock_loader.return_value = mock_loader_instance
        
        response = client.post(
            "/api/documents/upload",
            files={"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")},
            data={"ticker": "AAPL"}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "document_id" in data

def test_upload_document_non_pdf(client):
    """Test upload with non-PDF file"""
    # Auth handled by dependency override
    
    response = client.post(
        "/api/documents/upload",
        files={"file": ("test.txt", BytesIO(b"text content"), "text/plain")},
        data={"ticker": "AAPL"}
    )
    
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]

@patch('src.api.routes.documents.turso_db')
def test_list_documents_success(mock_turso_db, client):
    """Test listing user's documents"""
    # Auth handled by dependency override
    
    # Mock TursoDB response
    mock_doc = MagicMock()
    mock_doc.id = "doc123"
    mock_doc.ticker = "AAPL"
    mock_doc.filename = "test.pdf"
    mock_doc.created_at = "2023-01-01T00:00:00"
    mock_doc.s3_key = "user123/test.pdf"
    
    mock_turso_db.list_user_documents.return_value = [mock_doc]
    
    response = client.get("/api/documents/")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["document_id"] == "doc123"

@patch('src.api.routes.documents.turso_db')
def test_list_documents_empty(mock_turso_db, client):
    """Test listing documents for user with no documents"""
    # Auth handled by dependency override
    
    mock_turso_db.list_user_documents.return_value = []
    
    response = client.get("/api/documents/")
    
    assert response.status_code == 200
    assert response.json() == []

def test_get_document_success(client, mock_vector_store):
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
    mock_vector_store.query.return_value = mock_results
    
    response = client.get("/api/documents/doc123")
    
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == "doc123"

def test_get_document_not_found(client, mock_vector_store):
    """Test getting non-existent document"""
    # Auth handled by dependency override
    
    mock_results = MagicMock()
    mock_results.matches = []
    mock_vector_store.query.return_value = mock_results
    
    response = client.get("/api/documents/nonexistent")
    
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]

def test_get_document_analysis_structure(client, mock_object_store):
    """Test getting document analysis preserves new fields during transformation"""
    mock_data = {
        "answer": "Test answer",
        "is_valid": True,
        "intelligence_hub_data": {"sentiment": "positive"},
        "confidence_metrics": {"score": 0.9},
        "retrieval_scores": [0.8, 0.9],
        "retrieved_sources": ["doc1", "doc2"],
        "generation_logprobs": [-0.1, -0.2]
    }
    mock_object_store.get_json.return_value = mock_data
    
    response = client.get("/api/documents/doc123/analysis")
    
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Test answer"
    assert data["confidence_metrics"] == {"score": 0.9}
    assert data["retrieval_scores"] == [0.8, 0.9]
    assert data["retrieved_sources"] == ["doc1", "doc2"]
    assert data["generation_logprobs"] == [-0.1, -0.2]
