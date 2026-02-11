import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from src.main import app
from src.api.dependencies import get_vector_store, get_object_store, get_analysis_service
from src.core.services.analysis_service import AnalysisService
from src.infrastructure.storage.vector.base import VectorStore
from src.infrastructure.storage.object.base import ObjectStore

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

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "LedgerLens Analyst" in response.json()["status"]
