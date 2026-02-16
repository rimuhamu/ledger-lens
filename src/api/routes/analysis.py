from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any

from src.models import User, turso_db
from src.auth import get_current_user
from src.api.dependencies import get_analysis_service, get_object_store
from src.core.services.analysis_service import AnalysisService

router = APIRouter(prefix="/analysis", tags=["Analysis"])

class AnalysisRequest(BaseModel):
    query: str

@router.post("/{document_id}", response_model=Dict[str, Any])
async def analyze_document(
    document_id: str,
    request: AnalysisRequest,
    current_user: User = Depends(get_current_user),
    service: AnalysisService = Depends(get_analysis_service),
    object_store = Depends(get_object_store)
):
    """
    Analyze a document.
    """
    try:
        result = await service.analyze_document(
            question=request.query,
            document_id=document_id,
            user_id=current_user.id
        )
        
        # Save result to S3 for persistence
        analysis_key = f"{current_user.id}/{document_id}/analysis.json"
        object_store.save_json(result, analysis_key)
        
        # Extract metrics for DB
        ih_data = result.get("intelligence_hub_data", {})
        sentiment = ih_data.get("sentiment", {})
        risk = ih_data.get("risk", {})
        
        try:
            score_val = float(sentiment.get("score", 0))
        except:
            score_val = 0.0

        label = "neutral"
        if score_val > 60: label = "bullish"
        elif score_val < 40: label = "bearish"
        
        # Update TursoDB
        turso_db.update_document_analysis(
            document_id=document_id,
            sentiment_score=score_val,
            sentiment_label=label,
            ai_score=98.4,
            risk_level=risk.get("level", "low").lower(),
            summary=result.get("answer", "")[:500]
        )
        
        return {
            "answer": result.get("answer"),
            "verification_status": "PASS" if result.get("is_valid") else "FAIL",
            "intelligence_hub": result.get("intelligence_hub_data", {}),
            "confidence_metrics": result.get("confidence_metrics", {}),
            "sources": list(set(result.get("retrieved_sources", []))),
            "metadata": {
                "document_id": document_id
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{document_id}/status", response_model=Dict[str, Any])
async def get_analysis_status(
    document_id: str,
    current_user: User = Depends(get_current_user),
    service: AnalysisService = Depends(get_analysis_service),
    object_store = Depends(get_object_store)
):
    """
    Get the current status of an analysis.
    Returns progress information including current stage and status.
    """
    try:
        # First check if there's a status file
        status_data = await service.get_analysis_status(document_id, current_user.id)
        
        if status_data:
            # Return the progress information
            return {
                "status": status_data.get("status", "pending"),
                "current_stage": status_data.get("current_stage", "pending"),
                "stage_index": status_data.get("stage_index", 0),
                "total_stages": status_data.get("total_stages", 4),
                "message": status_data.get("status_message", "")
            }
        
        # If no status file, check the database for document status
        doc = turso_db.get_document(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if the document belongs to the user
        if doc.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to access this document")
        
        # Return basic status from database
        if doc.analysis_status == "completed":
            return {
                "status": "completed",
                "current_stage": "complete",
                "stage_index": 4,
                "total_stages": 4,
                "message": "Analysis completed"
            }
        elif doc.analysis_status == "failed":
            return {
                "status": "failed",
                "current_stage": "failed",
                "stage_index": 0,
                "total_stages": 4,
                "message": "Analysis failed"
            }
        else:
            return {
                "status": "pending",
                "current_stage": "pending",
                "stage_index": 0,
                "total_stages": 4,
                "message": "Analysis queued"
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
