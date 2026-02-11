from typing import List
from pydantic import BaseModel, Field
from src.domain.schemas.analysis import KeyHighlight, SentimentData, RiskData, RiskFactor

class AIIntelligenceHubData(BaseModel):
    """Complete JSON structure for AI Intelligence Hub UI"""
    key_highlights: List[KeyHighlight] = Field(..., description="List of key financial highlights")
    sentiment: SentimentData = Field(..., description="Sentiment score and trend")
    risk: RiskData = Field(..., description="Overall risk assessment")
    risk_factors: List[RiskFactor] = Field(..., description="Specific risk factors identified")
    suggested_questions: List[str] = Field(..., description="AI-suggested follow-up questions")
