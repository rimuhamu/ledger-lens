import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from src.main import app
from src.api.dependencies import get_vector_store, get_object_store, get_analysis_service
from src.auth import get_current_user
from src.infrastructure.storage.vector.base import VectorStore
from src.infrastructure.storage.object.base import ObjectStore
from src.core.services.analysis_service import AnalysisService
from src.models import User

@pytest.fixture
def mock_vector_store():
    return MagicMock(spec=VectorStore)

@pytest.fixture
def mock_object_store():
    return MagicMock(spec=ObjectStore)

@pytest.fixture
def mock_analysis_service():
    return MagicMock(spec=AnalysisService)

@pytest.fixture
def mock_user():
    return User(
        id="user123",
        email="test@example.com",
        password="hashed",
        created_at="2023-01-01T00:00:00"
    )

@pytest.fixture
def client(mock_vector_store, mock_object_store, mock_analysis_service, mock_user):
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    app.dependency_overrides[get_object_store] = lambda: mock_object_store
    app.dependency_overrides[get_analysis_service] = lambda: mock_analysis_service
    app.dependency_overrides[get_current_user] = lambda: mock_user
    
    with TestClient(app) as c:
        yield c
        
    app.dependency_overrides.clear()
