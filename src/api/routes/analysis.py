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
