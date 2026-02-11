
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    # API
    API_TITLE: str = "LedgerLens API"
    API_VERSION: str = "2.0.0"
    DEBUG: bool = False
    
    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_TEMPERATURE: float = 0.0
    
    # Vector Store
    VECTOR_STORE_PROVIDER: str = "pinecone"  # or "chroma"
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX_NAME: str = "ledgerlens"
    PINECONE_REGION: str = "us-east-1"
    
    # Object Storage
    OBJECT_STORE_PROVIDER: str = "s3"  # or "local"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str = "ledgerlens-documents"
    
    # Database
    TURSO_DATABASE_URL: Optional[str] = None
    TURSO_AUTH_TOKEN: Optional[str] = None
    
    # Authentication
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = 24
    
    # Document Processing
    CHUNK_SIZE: int = 2000
    CHUNK_OVERLAP: int = 400
    MAX_FILE_SIZE_MB: int = 50
    
    # External APIs
    NEWS_API_KEY: Optional[str] = None
    ENABLE_GEOPOLITICAL_ANALYSIS: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore" # Allow extra fields in .env

@lru_cache()
def get_settings() -> Settings:
    return Settings()
