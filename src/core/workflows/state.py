from typing import TypedDict, List, Optional, Any, Dict

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
    retrieval_scores: List[float]
    retrieved_sources: List[str]
    generation_logprobs: List[float]
    confidence_metrics: Dict[str, Any]
    current_stage: str
    stage_index: int
    total_stages: int
    status_message: str
    status: str
