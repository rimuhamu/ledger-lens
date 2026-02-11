import pytest
from unittest.mock import MagicMock, AsyncMock
from src.core.agents.researcher import Researcher
from src.core.workflows.state import AnalysisState

@pytest.mark.asyncio
async def test_researcher_execution(mock_vector_store):
    # Setup
    researcher = Researcher(vector_store=mock_vector_store)
    
    # Mock embedding
    researcher.embeddings = MagicMock()
    researcher.embeddings.embed_query.return_value = [0.1] * 1536
    
    # Mock vector store response
    mock_match = MagicMock()
    mock_match.metadata = {"text": "Retrieved Context"}
    
    mock_results = MagicMock()
    mock_results.matches = [mock_match]
    
    mock_vector_store.query.return_value = mock_results
    
    # Prepare state
    state = AnalysisState(
        question="What is the revenue?",
        context="",
        contexts=[],
        answer="",
        is_valid=False,
        intelligence_hub_data={},
        geopolitical_context="",
        document_id="doc_123",
        user_id="user_123"
    )
    
    # Execute
    result_state = await researcher.execute(state)
    
    # Assert
    assert "Retrieved Context" in result_state["context"]
    assert len(result_state["contexts"]) == 1
    mock_vector_store.query.assert_called_once()
