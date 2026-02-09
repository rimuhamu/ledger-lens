from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from enum import Enum


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
    icon: Literal["chart", "growth", "calendar", "alert", "check"] = "check"
    text: str = Field(..., description="Highlight description with key metrics")
    metric_value: Optional[str] = Field(None, description="Key metric value highlighted in the text")


class RiskFactor(BaseModel):
    """Individual risk factor with severity level"""
    icon: Literal["globe", "chain", "alert", "dollar", "chart"] = "alert"
    name: str = Field(..., description="Risk factor name")
    severity: RiskSeverity = Field(..., description="Risk severity: LOW, MED, or HIGH")


class SentimentData(BaseModel):
    """Sentiment score with trend information"""
    score: int = Field(..., ge=0, le=100, description="Sentiment score from 0-100")
    change: str = Field(..., description="Score change indicator, e.g., '+12%' or '-5%'")
    description: str = Field(..., description="Brief explanation of the sentiment")


class RiskData(BaseModel):
    """Overall risk level assessment"""
    level: RiskLevel = Field(..., description="Overall risk level")
    description: str = Field(..., description="Brief risk explanation")


class AIIntelligenceHubData(BaseModel):
    """Complete JSON structure for AI Intelligence Hub UI"""
    key_highlights: List[KeyHighlight] = Field(..., description="List of key financial highlights")
    sentiment: SentimentData = Field(..., description="Sentiment score and trend")
    risk: RiskData = Field(..., description="Overall risk assessment")
    risk_factors: List[RiskFactor] = Field(..., description="Specific risk factors identified")
    suggested_questions: List[str] = Field(..., description="AI-suggested follow-up questions")


# Example output structure for reference:
EXAMPLE_OUTPUT = {
    "key_highlights": [
        {
            "icon": "growth",
            "text": "Revenue increased by 26.2% YoY, primarily driven by Data Center compute demand.",
            "metric_value": "26.2%"
        },
        {
            "icon": "chart",
            "text": "Gross margin expanded to 72.7% from 53.6% in the previous fiscal year.",
            "metric_value": "72.7%"
        },
        {
            "icon": "calendar",
            "text": "Announced a 10-for-1 stock split effective June 2024.",
            "metric_value": "10-for-1"
        }
    ],
    "sentiment": {
        "score": 84,
        "change": "+12%",
        "description": "Strongly Bullish outlook based on R&D pipeline."
    },
    "risk": {
        "level": "Low",
        "description": "Supply chain volatility identified."
    },
    "risk_factors": [
        {
            "icon": "globe",
            "name": "Geopolitical Restrictions",
            "severity": "HIGH"
        },
        {
            "icon": "chain",
            "name": "Supply Chain Concentration",
            "severity": "MED"
        }
    ],
    "suggested_questions": [
        "Summarize the Data Center segment",
        "What are the major cash flow risks?",
        "Explain the R&D spending growth"
    ]
}
