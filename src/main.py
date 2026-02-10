import os
import shutil
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from graph import app as agent_graph
from database import Database
from auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
)
from models import User, turso_db, UserRegister, UserLogin, UserResponse, TokenResponse

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="LedgerLens API",
    description="Multi-step reasoning agent for financial report analysis with RAG",
    version="2.0.0"
)

# Initialize database (Pinecone + S3)
db = Database()


class AnalysisRequest(BaseModel):
    query: str
    document_id: Optional[str] = None

@app.post("/auth/register", response_model=TokenResponse)
async def register(body: UserRegister):
    """
    Register a new user account.

    Returns a JWT access token on success.
    """
    if not body.email or not body.password:
        raise HTTPException(status_code=400, detail="Email and password are required")

    if len(body.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    # Check if email already exists
    existing = turso_db.get_user_by_email(body.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    user = turso_db.create_user(body.email, hash_password(body.password))

    token = create_access_token(user.id, user.email)

    return TokenResponse(
        access_token=token,
        user=UserResponse(
            id=user.id,
            email=user.email,
            created_at=user.created_at,
        ),
    )


@app.post("/auth/login", response_model=TokenResponse)
async def login(body: UserLogin):
    """
    Authenticate with email and password.

    Returns a JWT access token on success.
    """
    user = turso_db.get_user_by_email(body.email)
    if not user or not verify_password(body.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(user.id, user.email)

    return TokenResponse(
        access_token=token,
        user=UserResponse(
            id=user.id,
            email=user.email,
            created_at=user.created_at,
        ),
    )


@app.get("/auth/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get the currently authenticated user's profile."""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        created_at=current_user.created_at,
    )

@app.get("/")
def read_root():
    return {
        "status": "LedgerLens Analyst is Online",
        "version": "2.0.0",
        "storage": "Pinecone + S3",
        "model": "GPT-4o-mini / LangGraph"
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pinecone": "connected",
        "s3": "connected"
    }

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    ticker: str = Form(...),
    current_user: User = Depends(get_current_user),
):
    """
    Upload and save a document permanently for future analysis.
    
    Features:
    - PDF saved to S3
    - Embeddings stored in Pinecone
    - Can ask multiple questions later
    - Build document library
    
    Requires JWT authentication (Authorization: Bearer <token>).
    """
    user_id = current_user.id

    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        # Save temporarily
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Ingest document (saved to Pinecone + S3)
        result = db.ingest_document(
            file_path=file_path,
            user_id=user_id,
            ticker=ticker,
            filename=file.filename,
        )

        # Define a broad question to extract key metrics
        analysis_question = "Analyze the financial performance, key risks, and strategic outlook for this period."

        # Retrieve context for this document
        # Retry logic for reading from Pinecone (eventual consistency)
        import time
        max_retries = 10  # Increased for eventual consistency
        contexts = []
        for i in range(max_retries):
            contexts = db.query_documents(
                query=analysis_question,
                document_id=result["document_id"],
                user_id=user_id,
                top_k=8
            )
            if contexts:
                break
            time.sleep(2)  # Wait for indexing
        
        if not contexts:
            print(f"Warning: No context found for analysis after {max_retries} attempts.")


        intelligence_data = {}
        if contexts:
            # Run the agent graph
            state = {
                "question": analysis_question,
                "context": "\n\n".join([c["content"] for c in contexts]),
                "contexts": [c["content"] for c in contexts],
                "document_id": result["document_id"]
            }
            analysis_result = await agent_graph.ainvoke(state)
            intelligence_data = analysis_result.get("intelligence_hub_data", {})

        return {
            "message": "Document uploaded and analyzed successfully",
            "document_id": result["document_id"],
            "ticker": ticker,
            "filename": file.filename,
            "num_chunks": result["num_chunks"],
            "num_pages": result["num_pages"],
            "s3_key": result["s3_key"],
            "intelligence_analysis": intelligence_data
        }
        
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/{document_id}/analyze")
async def analyze_saved_document(
    document_id: str,
    request: AnalysisRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Analyze a previously uploaded document.
    
    Can ask multiple questions on the same document without re-uploading.
    Requires JWT authentication (Authorization: Bearer <token>).
    """
    user_id = current_user.id

    try:
        # Query specific document (with user verification)
        contexts = db.query_documents(
            query=request.query,
            document_id=document_id,
            user_id=user_id,  # Verify ownership
            top_k=8
        )
        
        if not contexts:
            raise HTTPException(
                status_code=404, 
                detail="Document not found or access denied"
            )
        
        # Run analysis
        state = {
            "question": request.query,
            "context": "\n\n".join([c["content"] for c in contexts]),
            "contexts": [c["content"] for c in contexts],
            "document_id": document_id
        }
        
        result = await agent_graph.ainvoke(state)
        
        return {
            "answer": result["answer"],
            "verification_status": "PASS" if result["is_valid"] else "FAIL",
            "intelligence_hub": result.get("intelligence_hub_data", {}),
            "metadata": {
                "document_id": document_id,
                "ticker": contexts[0]["ticker"],
                "num_contexts": len(contexts)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents(current_user: User = Depends(get_current_user)):
    """
    List all saved documents for the authenticated user.
    
    Returns document metadata including:
    - document_id
    - ticker
    - filename
    - upload date
    - S3 storage location
    """
    user_id = current_user.id
    documents = db.list_user_documents(user_id)
    
    return {
        "user_id": user_id,
        "count": len(documents),
        "documents": documents
    }


@app.get("/documents/{document_id}")
async def get_document_info(
    document_id: str,
    current_user: User = Depends(get_current_user),
):
    """Get detailed information about a specific document"""
    user_id = current_user.id
    doc_meta = db.get_document_metadata(document_id, user_id)
    
    if not doc_meta:
        raise HTTPException(status_code=404, detail="Document not found or access denied")
    
    return doc_meta


@app.get("/documents/{document_id}/download")
async def download_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    Download the original PDF file from S3.
    
    Requires JWT authentication (Authorization: Bearer <token>).
    """
    user_id = current_user.id
    
    # Get document metadata
    doc_meta = db.get_document_metadata(document_id, user_id)
    
    if not doc_meta:
        raise HTTPException(status_code=404, detail="Document not found or access denied")
    
    # Download from S3 to temp location
    temp_path = os.path.join(UPLOAD_DIR, f"{document_id}_{doc_meta['filename']}")
    
    result = db.download_document(document_id, user_id, temp_path)
    
    if not result:
        raise HTTPException(status_code=500, detail="Failed to download document from S3")
    
    # Return file and cleanup after sending
    return FileResponse(
        path=temp_path,
        filename=doc_meta['filename'],
        media_type='application/pdf',
        background=None  # Could add cleanup task here
    )


@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    Delete a saved document (removes from both Pinecone and S3).
    
    Requires JWT authentication (Authorization: Bearer <token>).
    """
    user_id = current_user.id
    
    # Verify document exists and user has access
    doc_meta = db.get_document_metadata(document_id, user_id)
    
    if not doc_meta:
        raise HTTPException(status_code=404, detail="Document not found or access denied")
    
    # Delete from both Pinecone and S3
    db.delete_document(document_id, user_id)
    
    return {
        "message": "Document deleted successfully",
        "document_id": document_id,
        "filename": doc_meta["filename"]
    }