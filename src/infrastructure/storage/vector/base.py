from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

class VectorStore(ABC):
    @abstractmethod
    def upsert(self, vectors: List[Dict[str, Any]]) -> None:
        """Insert or update vectors"""
        pass
    
    @abstractmethod
    def query(
        self, 
        vector: List[float], 
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> Any:
        """Query similar vectors"""
        pass
    
    @abstractmethod
    def delete(self, filter: Dict[str, Any]) -> None:
        """Delete vectors by filter"""
        pass
