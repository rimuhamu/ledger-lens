import os
import shutil
import time
import json
from typing import List
from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Depends, BackgroundTasks
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


from src.domain.schemas.document import DocumentResponse, DocumentIngestResponse

router = APIRouter(prefix="/documents", tags=["Documents"])

UPLOAD_DIR = os.path.join(os.getcwd(), "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload", response_model=DocumentIngestResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    ticker: str = Form(...),
    current_user: User = Depends(get_current_user),
    vector_store: VectorStore = Depends(get_vector_store),
    object_store: ObjectStore = Depends(get_object_store),
    analysis_service: AnalysisService = Depends(get_analysis_service)
):
    user_id = current_user.id
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        # Save file temporarily for processing
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Upload to S3
        s3_key = f"{user_id}/{file.filename}"
        object_store.upload_file(file_path, s3_key)
        
        # Parse PDF and split into chunks

        
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(pages)
        
        # Generate embeddings and prepare vectors for Pinecone
        
        embeddings = OpenAIEmbeddings()

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
        
        # Clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)
            
        # Schedule background analysis
        background_tasks.add_task(
            background_analysis_task,
            document_id,
            user_id,
            analysis_service,
            object_store
        )

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
    
    # Query with dummy vector (1536-dim for OpenAI) to retrieve document metadata
    dummy_vector = [0.0] * 1536
    

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
        print(f"Error listing documents: {e}")
        return []

async def background_analysis_task(
    document_id: str,
    user_id: str,
    analysis_service: AnalysisService,
    object_store: ObjectStore
):
    try:
        print(f"Starting background analysis for doc {document_id}")
        # Default question for auto-analysis
        question = "Provide a comprehensive financial analysis of this document, including key highlights, risks, and strategic outlook."
        
        result = await analysis_service.analyze_document(
            question=question,
            document_id=document_id,
            user_id=user_id
        )
        
        # Save result to S3
        analysis_key = f"{user_id}/{document_id}/analysis.json"
        object_store.save_json(result, analysis_key)
        print(f"Background analysis completed and saved to {analysis_key}")
        
    except Exception as e:
        print(f"Background analysis failed for doc {document_id}: {e}")

@router.get("/{document_id}/analysis")
async def get_document_analysis(
    document_id: str,
    current_user: User = Depends(get_current_user),
    object_store: ObjectStore = Depends(get_object_store)
):
    """Retrieve the analysis result for a document"""
    user_id = current_user.id
    analysis_key = f"{user_id}/{document_id}/analysis.json"
    
    try:
        analysis_data = object_store.get_json(analysis_key)
        if not analysis_data:
            return {"status": "processing", "message": "Analysis not yet available"}
            
        return analysis_data
    except Exception as e:
        print(f"Error fetching analysis: {e}")
        return {"status": "processing", "message": "Analysis not yet available"}

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
