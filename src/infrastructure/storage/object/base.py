from abc import ABC, abstractmethod
from typing import Optional

class ObjectStore(ABC):
    @abstractmethod
    def upload_file(self, file_path: str, key: str) -> str:
        """Upload a file and return its key"""
        pass
    
    @abstractmethod
    def download_file(self, key: str, local_path: str) -> str:
        """Download a file to a local path"""
        pass
    
    @abstractmethod
    def delete_file(self, key: str) -> None:
        """Delete a file by key"""
        pass
