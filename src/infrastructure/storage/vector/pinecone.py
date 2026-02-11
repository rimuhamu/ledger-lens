import os
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from src.infrastructure.storage.vector.base import VectorStore
from src.utils.logger import get_logger

class PineconeVectorStore(VectorStore):
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "ledgerlens")
        self.region = os.getenv("PINECONE_REGION", "us-east-1")
        
        if not self.api_key:
            self.logger.warning("PINECONE_API_KEY not set")
            return

        self.pc = Pinecone(api_key=self.api_key)
        self._ensure_index()
        self.index = self.pc.Index(self.index_name)

    def _ensure_index(self):
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            self.logger.info(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # OpenAI embeddings dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=self.region
                )
            )

    def upsert(self, vectors: List[Dict[str, Any]]) -> None:
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
            self.logger.info(f"Upserted batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")

    def query(
        self, 
        vector: List[float], 
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> Any:
        return self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=include_metadata,
            filter=filter
        )

    def delete(self, filter: Dict[str, Any]) -> None:
        self.index.delete(filter=filter)
        self.logger.info(f"Deleted vectors with filter: {filter}")
