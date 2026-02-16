from functools import lru_cache
from typing import Generator

from src.infrastructure.storage.vector.pinecone import PineconeVectorStore
from src.infrastructure.storage.object.s3 import S3ObjectStore
from src.core.services.analysis_service import AnalysisService
from src.infrastructure.storage.vector.base import VectorStore
from src.infrastructure.storage.object.base import ObjectStore

@lru_cache()
def get_vector_store() -> VectorStore:
    return PineconeVectorStore()

@lru_cache()
def get_object_store() -> ObjectStore:
    return S3ObjectStore()

@lru_cache()
def get_analysis_service() -> AnalysisService:
    vector_store = get_vector_store()
    object_store = get_object_store()
    return AnalysisService(vector_store, object_store)
