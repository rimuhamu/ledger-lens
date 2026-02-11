import os
import shutil
import time
from typing import List
from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Depends
from fastapi.responses import FileResponse

from src.models import User
from src.auth import get_current_user
from src.api.dependencies import get_vector_store, get_object_store, get_analysis_service
from src.infrastructure.storage.vector.base import VectorStore
from src.infrastructure.storage.object.base import ObjectStore
from src.core.services.analysis_service import AnalysisService
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import uuid

# We need a Pydantic schema for Document List response if not already defined
# src/domain/schemas/document.py has DocumentResponse

from src.domain.schemas.document import DocumentResponse, DocumentIngestResponse

router = APIRouter(prefix="/documents", tags=["Documents"])

UPLOAD_DIR = os.path.join(os.getcwd(), "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload", response_model=DocumentIngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    ticker: str = Form(...),
    current_user: User = Depends(get_current_user),
    vector_store: VectorStore = Depends(get_vector_store),
    object_store: ObjectStore = Depends(get_object_store),
    # db is used in legacy for "ingest_document" which did everything
    # We should refactor ingestion logic to a service or proper infrastructure usage.
    # For this step, I might need to reproduce `ingest_document` logic here or in a service.
    # The Legacy `database.py` had `ingest_document`.
    # Phase 2 created `PineconeVectorStore` and `S3ObjectStore`.
    # We need to glue them together.
    # I'll create `DocumentService` or put logic here for now to match strict plan.
    # Plan says: "Uses ObjectStore and VectorStore via dependencies".
):
    user_id = current_user.id
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
         # Save temporarily
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 1. Upload to Object Store (S3)
        s3_key = f"{user_id}/{file.filename}"
        object_store.upload_file(file_path, s3_key)
        
        # 2. Ingest to Vector Store (Pinecone)
        # We need to parse PDF first.
        # Legacy `database.py` used `PyPDFLoader` and `RecursiveCharacterTextSplitter`.
        # I should probably move that logic to `src/utils/document_processor.py` or similar.
        # But to keep it simple and actionable, I'll allow inline import or helper method here.
        # Legacy dependencies: langchain_community.document_loaders, langchain_text_splitters
        
        
        # loader = PyPDFLoader(file_path) # these are now imported at top

        
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(pages)
        
        # Prepare vectors with OpenAI Embeddings
        # Note: PineconeVectorStore.upsert expects dicts with values/metadata 
        # OR we use the LangChain pinecone wrapper if we want.
        # But our interface says `upsert(vectors: List[Dict])`.
        # `PineconeVectorStore` implementation uses `self.index.upsert(vectors=batch)`.
        # So we need to format for Pinecone: (id, values, metadata).
        
        embeddings = OpenAIEmbeddings()
        # Embed all texts
        texts = [chunk.page_content for chunk in chunks]
        vectors = embeddings.embed_documents(texts)
        
        document_id = str(uuid.uuid4())
        
        upsert_batch = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            chunk_id = f"{document_id}_{i}"
            metadata = {
                "document_id": document_id,
                "text": chunk.page_content,
                "page": chunk.metadata.get("page", 0),
                "source": file.filename,
                "ticker": ticker,
                "user_id": user_id,
                "filename": file.filename,
                "timestamp": str(time.time())
            }
            upsert_batch.append({
                "id": chunk_id,
                "values": vector,
                "metadata": metadata
            })
            
        vector_store.upsert(upsert_batch)
        
        # Cleanup temp file
        if os.path.exists(file_path):
            os.remove(file_path)
            
        return DocumentIngestResponse(
            document_id=document_id,
            num_chunks=len(chunks),
            num_pages=len(pages),
            s3_key=s3_key,
            status="success"
        )

    except Exception as e:
        if os.path.exists(file_path):
             os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[DocumentResponse])
async def list_documents(
    current_user: User = Depends(get_current_user),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    List all saved documents for the authenticated user.
    """
    user_id = current_user.id
    
    # Use Pinecone's query with dummy vector to retrieve metadata
    # We need a dummy vector of correct dimension (1536 for OpenAI)
    dummy_vector = [0.0] * 1536
    
    # Query with metadata filter for user_id
    try:
        results = vector_store.query(
            vector=dummy_vector,
            top_k=10000,
            filter={"user_id": user_id},
            include_metadata=True
        )
        
        # Deduplicate by document_id
        seen = set()
        documents = []
        
        # Handle Pinecone results structure
        matches = getattr(results, 'matches', [])
        
        for match in matches:
            if not hasattr(match, 'metadata'):
                continue
                
            metadata = match.metadata
            doc_id = metadata.get("document_id")
            
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                documents.append(DocumentResponse(
                    document_id=doc_id,
                    ticker=metadata.get("ticker", "UNKNOWN"),
                    filename=metadata.get("filename", "unknown"),
                    created_at=str(metadata.get("created_at", "")),
                    s3_key=metadata.get("s3_key")
                ))
        
        return documents
    except Exception as e:
        # In case of pinecone error or connection issue
        print(f"Error listing documents: {e}")
        return []

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Get detailed information about a specific document"""
    user_id = current_user.id
    
    dummy_vector = [0.0] * 1536
    
    try:
        results = vector_store.query(
            vector=dummy_vector,
            top_k=1,
            filter={"document_id": document_id, "user_id": user_id},
            include_metadata=True
        )
        
        matches = getattr(results, 'matches', [])
        
        if matches and hasattr(matches[0], 'metadata'):
            metadata = matches[0].metadata
            return DocumentResponse(
                document_id=metadata.get("document_id"),
                ticker=metadata.get("ticker", "UNKNOWN"),
                filename=metadata.get("filename", "unknown"),
                created_at=str(metadata.get("created_at", "")),
                s3_key=metadata.get("s3_key")
            )
    except Exception as e:
        print(f"Error fetching document: {e}")
        
    raise HTTPException(status_code=404, detail="Document not found or access denied")
