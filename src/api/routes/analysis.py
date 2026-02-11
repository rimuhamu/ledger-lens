from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any

from src.models import User
from src.auth import get_current_user
from src.api.dependencies import get_analysis_service
from src.core.services.analysis_service import AnalysisService

router = APIRouter(prefix="/analysis", tags=["Analysis"])

class AnalysisRequest(BaseModel):
    query: str

@router.post("/{document_id}", response_model=Dict[str, Any])
async def analyze_document(
    document_id: str,
    request: AnalysisRequest,
    current_user: User = Depends(get_current_user),
    service: AnalysisService = Depends(get_analysis_service)
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
        
        return {
            "answer": result.get("answer"),
            "verification_status": "PASS" if result.get("is_valid") else "FAIL",
            "intelligence_hub": result.get("intelligence_hub_data", {}),
            "metadata": {
                "document_id": document_id
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
