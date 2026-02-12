from fastapi import APIRouter, Depends
from typing import Dict, Any

from src.models import User, turso_db
from src.auth import get_current_user

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

@router.get("/stats")
async def get_dashboard_stats(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get aggregated statistics for the user's dashboard.
    """
    return turso_db.get_dashboard_stats(current_user.id)
