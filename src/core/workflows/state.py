from typing import TypedDict, List, Optional, Any

class AnalysisState(TypedDict):
    question: str
    context: str
    contexts: List[str]
    answer: str
    is_valid: bool
    intelligence_hub_data: dict
    geopolitical_context: str
    document_id: Optional[str]
    user_id: Optional[str]
