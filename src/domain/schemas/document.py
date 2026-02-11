from typing import Optional, List
from pydantic import BaseModel

class DocumentResponse(BaseModel):
    document_id: str
    ticker: str
    filename: str
    created_at: str
    s3_key: Optional[str] = None
    
class DocumentIngestResponse(BaseModel):
    document_id: str
    num_chunks: int
    num_pages: int
    s3_key: str
    status: str
