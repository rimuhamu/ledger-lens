import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.core.services.analysis_service import AnalysisService
from src.models import Document

# ============================================================================
# PROGRESS TRACKING TESTS
# ============================================================================

@patch('src.api.routes.analysis.turso_db')
def test_get_analysis_status_success(mock_turso_db, client, mock_object_store, mock_analysis_service):
    """Test getting analysis status from S3"""
    # Auth handled by dependency override
    
    # Mock status data in S3
    mock_status_data = {
        "status": "in_progress",
        "current_stage": "analysis",
        "stage_index": 1,
        "total_stages": 4,
        "status_message": "Analyzing document"
    }
    
    # Mock get_analysis_status to return data
    mock_analysis_service.get_analysis_status = AsyncMock(return_value=mock_status_data)
    
    response = client.get("/api/analysis/doc123/status")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "in_progress"
    assert data["current_stage"] == "analysis"
    assert data["stage_index"] == 1
    assert data["message"] == "Analyzing document"

@patch('src.api.routes.analysis.turso_db')
def test_get_analysis_status_fallback_db(mock_turso_db, client, mock_analysis_service):
    """Test fallback to DB when S3 status missing"""
    # Auth handled by dependency override
    
    # Mock S3 return None
    mock_analysis_service.get_analysis_status = AsyncMock(return_value=None)
    
    # Mock DB document
    mock_doc = Document(
        id="doc123",
        user_id="user123",
        filename="test.pdf",
        ticker="AAPL",
        s3_key="key",
        created_at="now",
        analysis_status="completed"
    )
    mock_turso_db.get_document.return_value = mock_doc
    
    response = client.get("/api/analysis/doc123/status")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["current_stage"] == "complete"
    assert data["stage_index"] == 4

@patch('src.api.routes.analysis.turso_db')
def test_get_analysis_status_not_found(mock_turso_db, client, mock_analysis_service):
    """Test status for non-existent document"""
    # Auth handled by dependency override
    
    # Mock S3 return None
    mock_analysis_service.get_analysis_status = AsyncMock(return_value=None)
    
    # Mock DB return None
    mock_turso_db.get_document.return_value = None
    
    response = client.get("/api/analysis/nonexistent/status")
    
    assert response.status_code == 404

def test_workflow_saves_progress(mock_object_store):
    """Test that workflow saves progress to S3"""
    from src.core.workflows.financial_analysis import FinancialAnalysisWorkflow
    from src.core.workflows.state import AnalysisState
    
    # Mock agents
    researcher = AsyncMock()
    analyst = AsyncMock()
    validator = AsyncMock()
    intelligence = AsyncMock()
    
    # Setup workflow with object store
    workflow = FinancialAnalysisWorkflow(
        researcher, analyst, validator, intelligence, 
        object_store=mock_object_store
    )
    
    # Test _save_progress directly
    state = AnalysisState(
        document_id="doc123",
        user_id="user123", 
        current_stage="research",
        status="in_progress",
        stage_index=0,
        total_stages=4,
        status_message="Test message",
        # Required fields
        question="", context="", contexts=[], answer="", is_valid=False,
        intelligence_hub_data={}, geopolitical_context="",
        retrieval_scores=[], retrieved_sources=[], generation_logprobs=[],
        confidence_metrics={}
    )
    
    workflow._save_progress(state)
    
    # Verify save_json called
    mock_object_store.save_json.assert_called_once()
    args = mock_object_store.save_json.call_args
    data = args[0][0]
    key = args[0][1]
    
    assert data["current_stage"] == "research"
    assert "user123/doc123/status.json" in key
