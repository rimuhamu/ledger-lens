from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field


class RiskSeverity(str, Enum):
    LOW = "LOW"
    MED = "MED"
    HIGH = "HIGH"


class RiskLevel(str, Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"


class KeyHighlight(BaseModel):
    """Individual highlight item with metric details"""
    icon: str = Field("check", description="Icon type: chart, growth, calendar, alert, check, globe, chain, dollar")
    text: str = Field(..., description="Highlight description with key metrics")
    metric_value: Optional[str] = Field(None, description="Key metric value highlighted in the text")


class RiskFactor(BaseModel):
    """Individual risk factor with severity level"""
    icon: str = Field("alert", description="Icon type: globe, chain, alert, dollar, chart")
    name: str = Field(..., description="Risk factor name")
    severity: RiskSeverity = Field(..., description="Risk severity: LOW, MED, or HIGH")


class SentimentData(BaseModel):
    """Sentiment score with trend information"""
    score: int = Field(..., ge=0, le=100, description="Sentiment score from 0-100")
    change: Optional[str] = Field(None, description="Score change indicator (e.g., '+12%', '-5%', or None if not available)")
    description: str = Field(..., description="Brief explanation of the sentiment")


class RiskData(BaseModel):
    """Overall risk level assessment"""
    level: RiskLevel = Field(..., description="Overall risk level")
    description: str = Field(..., description="Brief risk explanation")
