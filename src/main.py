import os
import shutil
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from graph import app as agent_graph
from database import Database

# Directory for uploaded files
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="LedgerLens Analyst API")

class AnalysisRequest(BaseModel):
    ticker: str
    query: str

@app.get("/")
def read_root():
    return {"status": "LedgerLens Analyst is Online", "model": "GPT-4o-mini / LangGraph"}

@app.post("/analyze")
async def analyze_report(request: AnalysisRequest):
    """
    Triggers the multi-step reasoning agent to analyze the BCA 2024 report.
    Returns structured intelligence hub data including sentiment, risk, and highlights.
    """
    try:
        inputs = {"question": f"For {request.ticker}: {request.query}"}
        
        result = await agent_graph.ainvoke(inputs)
        
        return {
            "answer": result["answer"],
            "verification_status": "PASS" if result["is_valid"] else "FAIL",
            "intelligence_hub": result.get("intelligence_hub_data", {}),
            "metadata": {"source": "BCA Annual Report 2024"}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-and-analyze")
async def upload_and_analyze(
    file: UploadFile = File(...),
    ticker: str = Form(...)
):
    """
    Upload a PDF annual report and analyze it.
    Includes real-time geopolitical risk analysis from external sources.
    
    - **file**: PDF file to upload
    - **ticker**: Company ticker symbol
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file temporarily
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create a temporary vector store for this document
        temp_db_path = os.path.join(UPLOAD_DIR, f"vectorstore_{ticker}")
        temp_db = Database(db_path=temp_db_path)
        
        # Ingest the uploaded document
        success = temp_db.ingest_document(file_path)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process the uploaded document")
        
        # Temporarily update the global retriever in nodes.py
        from nodes import set_retriever
        temp_retriever = temp_db.get_retriever()
        set_retriever(temp_retriever)
        
        # Run the analysis with a comprehensive default question
        inputs = {"question": f"Provide a comprehensive financial analysis for {ticker} including key highlights, financial performance, growth metrics, risk factors, and outlook."}
        result = await agent_graph.ainvoke(inputs)
        
        # Cleanup temporary files
        os.remove(file_path)
        shutil.rmtree(temp_db_path, ignore_errors=True)
        
        return {
            "answer": result["answer"],
            "verification_status": "PASS" if result["is_valid"] else "FAIL",
            "intelligence_hub": result.get("intelligence_hub_data", {}),
            "metadata": {
                "source": file.filename,
                "ticker": ticker
            }
        }
        
    except Exception as e:
        # Cleanup on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Reset to default retriever
        from nodes import reset_retriever
        reset_retriever()