import os
import shutil
import time
import json
from datetime import datetime
from typing import List
from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Depends, BackgroundTasks
from fastapi.responses import FileResponse

from src.models import User, turso_db
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
        
        # Create document record in TursoDB
        turso_db.create_document(
            document_id=document_id,
            user_id=user_id,
            filename=file.filename,
            ticker=ticker,
            s3_key=s3_key
        )
        
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
                "created_at": datetime.utcnow().isoformat() + "Z"
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
    current_user: User = Depends(get_current_user)
):
    """
    List all saved documents for the authenticated user from TursoDB.
    """
    try:
        docs = turso_db.list_user_documents(current_user.id)
        
        return [
            DocumentResponse(
                document_id=doc.id,
                ticker=doc.ticker,
                filename=doc.filename,
                created_at=doc.created_at,
                s3_key=doc.s3_key
            ) for doc in docs
        ]
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
        
        # Extract metrics for DB
        ih_data = result.get("intelligence_hub_data", {})
        sentiment = ih_data.get("sentiment", {})
        risk = ih_data.get("risk", {})
        
        # Normalize score to 0-100 float
        raw_score = sentiment.get("score", 0)
        # Handle if score is str "75" or int 75 or float 0.75? Assuming 0-100 based on UI
        try:
            score_val = float(raw_score)
        except:
            score_val = 0.0

        # Determine label if not present (simple heuristic if missing)
        label = "neutral"
        if score_val > 60: label = "bullish"
        elif score_val < 40: label = "bearish"
        
        # Update TursoDB
        turso_db.update_document_analysis(
            document_id=document_id,
            sentiment_score=score_val,
            sentiment_label=label,
            ai_score=98.4, # Mocked validation score for now, or extract if available
            risk_level=risk.get("level", "low").lower(),
            summary=result.get("answer", "")[:500] # Truncate summary for DB
        )

        # Save full result to S3
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
        
        # Transform old format to new format for backward compatibility
        if 'intelligence_hub_data' in analysis_data and 'intelligence_hub' not in analysis_data:
            analysis_data = {
                "answer": analysis_data.get("answer"),
                "verification_status": "PASS" if analysis_data.get("is_valid") else "FAIL",
                "intelligence_hub": analysis_data.get("intelligence_hub_data", {}),
                "metadata": {"document_id": document_id}
            }
            
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
