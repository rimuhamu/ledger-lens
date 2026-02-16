from unittest.mock import AsyncMock

# ============================================================================
# ANALYSIS ROUTES
# ============================================================================

def test_analyze_document_success(client, mock_analysis_service):
    """Test successful document analysis"""
    # Auth handled by dependency override
    
    # Mock analysis service
    mock_analysis_service.analyze_document = AsyncMock(return_value={
        "answer": "Test analysis result",
        "is_valid": True,
        "intelligence_hub_data": {"key": "value"}
    })
    
    response = client.post(
        "/api/analysis/doc123",
        json={"query": "What is the revenue?"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Test analysis result"
    assert data["verification_status"] == "PASS"

def test_analyze_document_validation_fail(client, mock_analysis_service):
    """Test analysis with validation failure"""
    # Auth handled by dependency override
    
    mock_analysis_service.analyze_document = AsyncMock(return_value={
        "answer": "Invalid result",
        "is_valid": False,
        "intelligence_hub_data": {}
    })
    
    response = client.post(
        "/api/analysis/doc123",
        json={"query": "What is the revenue?"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["verification_status"] == "FAIL"

def test_analyze_document_error(client, mock_analysis_service):
    """Test analysis with service error"""
    # Auth handled by dependency override
    
    mock_analysis_service.analyze_document = AsyncMock(
        side_effect=Exception("Analysis failed")
    )
    
    response = client.post(
        "/api/analysis/doc123",
        json={"query": "What is the revenue?"}
    )
    
    assert response.status_code == 500
    assert "Analysis failed" in response.json()["detail"]
