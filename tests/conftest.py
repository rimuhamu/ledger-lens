import pytest
import sys
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass

# Mock Turso database BEFORE importing anything that uses it
@dataclass
class MockUser:
    id: str
    email: str
    password: str
    created_at: str

mock_turso_client = MagicMock()
mock_turso_db = MagicMock()

# Patch libsql_client before it gets imported by models.py
sys.modules['libsql_client'] = MagicMock()

# Now we can safely import
from src.infrastructure.storage.vector.base import VectorStore
from src.infrastructure.storage.object.base import ObjectStore
from src.core.workflows.state import AnalysisState
from src.models import User

@pytest.fixture
def mock_vector_store():
    store = MagicMock(spec=VectorStore)
    store.query.return_value = []
    return store

@pytest.fixture
def mock_object_store():
    store = MagicMock(spec=ObjectStore)
    return store

@pytest.fixture
def test_user():
    return User(
        id="test_user_id",
        email="test@example.com",
        password="hashed_password",
        created_at="2023-01-01T00:00:00"
    )

@pytest.fixture
def sample_analysis_state():
    return AnalysisState(
        question="Test Question",
        context="Test Context",
        contexts=["Test Context"],
        answer="",
        is_valid=False,
        intelligence_hub_data={},
        geopolitical_context="",
        document_id="doc_123",
        user_id="user_123"
    )
