from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from src.core.agents.base import BaseAgent
from src.core.workflows.state import AnalysisState
from src.infrastructure.storage.vector.base import VectorStore
from src.core.services.geopolitical_service import get_geopolitical_service

class Researcher(BaseAgent[AnalysisState]):
    def __init__(self, vector_store: VectorStore):
        super().__init__()
        self.vector_store = vector_store
        self.geo_service = get_geopolitical_service()
        self.embeddings = OpenAIEmbeddings(api_key=self.settings.OPENAI_API_KEY)

    async def execute(self, state: AnalysisState) -> AnalysisState:
        self._log_execution(state)
        question = state["question"]
        document_id = state.get("document_id")
        
        # Determine country for geopolitical risks
        # (Simplified logic - could use LLM here too)
        geopolitical_context = ""
        # TODO: Add logic to extract country if needed, for now skipping complex extraction
        
        # Embed query
        query_vector = self.embeddings.embed_query(question)
        
        # Query Vector Store
        results = self.vector_store.query(
            vector=query_vector,
            top_k=8,
            filter={"document_id": document_id} if document_id else None
        )
        
        # Extract content
        # Note: VectorStore.query returns implementation-specific result (Pinecone QueryResponse)
        # We need to adapt it. 
        # Ideally, the VectorStore interface should return a standard Dict.
        # Let's assume PineconeVectorStore returns a list of matches or we parse it there.
        # But `src/infrastructure/storage/vector/pinecone.py` returns `self.index.query(...)`.
        # This returns a Pinecone Object.
        
        chunks = []
        if hasattr(results, 'matches'):
            for match in results.matches:
                if hasattr(match, 'metadata') and hasattr(match.metadata, '__getitem__'):
                    if 'text' in match.metadata:
                        chunks.append(match.metadata['text'])
        
        state["context"] = "\n\n".join(chunks)
        state["contexts"] = chunks
        state["geopolitical_context"] = geopolitical_context
        
        return state
