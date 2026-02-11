from dataclasses import dataclass
from typing import Optional

@dataclass
class Document:
    id: str
    user_id: str
    ticker: str
    filename: str
    created_at: str
    s3_key: Optional[str] = None
    num_chunks: Optional[int] = None
    num_pages: Optional[int] = None
