from typing import Dict, Any, Optional, List
import math
from src.core.agents.researcher import Researcher
from src.core.agents.analyst import Analyst
from src.core.agents.validator import Validator
from src.core.agents.intelligence_hub import IntelligenceHub
from src.core.workflows.financial_analysis import FinancialAnalysisWorkflow
from src.core.workflows.state import AnalysisState
from src.infrastructure.storage.vector.base import VectorStore
from src.utils.logger import get_logger

class AnalysisService:
    def __init__(self, vector_store: VectorStore, object_store=None):
        self.logger = get_logger(self.__class__.__name__)
        self.object_store = object_store
        
        # Initialize Agents
        self.researcher = Researcher(vector_store)
        self.analyst = Analyst()
        self.validator = Validator()
        self.intelligence_hub = IntelligenceHub()
        
        # Initialize Workflow
        self.workflow = FinancialAnalysisWorkflow(
            self.researcher,
            self.analyst,
            self.validator,
            self.intelligence_hub,
            self.object_store
        )

    def _calculate_confidence_metrics(self, state: AnalysisState) -> Dict[str, Any]:
        """
        Calculate confidence metrics based on retrieval and generation data.
        """
        retrieval_scores = state.get("retrieval_scores", [])
        retrieved_contexts = state.get("contexts", [])
        logprobs = state.get("generation_logprobs", [])
        
        # 1. Source Match (Average Retrieval Score)
        # Pinecone cosine similarity is -1 to 1, but for OpenAI embeddings usually 0.7-1.0
        # We normalize or just present as percentage.
        avg_score = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0
        source_match_val = f"{avg_score:.0%}"
        
        # 2. Context Density (Chunks used vs Retrieved/Requested)
        # For now, we assume all retrieved chunks are "used" in context.
        # We can define density as ratio of chunks with high score (>0.75)
        high_quality_chunks = [s for s in retrieval_scores if s > 0.75]
        density_ratio = len(high_quality_chunks) / len(retrieval_scores) if retrieval_scores else 0.0
        density_val = f"{len(high_quality_chunks)}/{len(retrieval_scores)} chunks > 0.75"
        
        # 3. AI Certainty (Avg Logprob)
        # Logprobs are negative log likelihoods. 
        # Token confidence = exp(logprob)
        # Avg confidence = exp(avg_logprob)
        if logprobs:
            avg_logprob = sum(logprobs) / len(logprobs)
            confidence = math.exp(avg_logprob)
        else:
            confidence = 0.0
            
        ai_certainty_val = f"{confidence:.0%}"
        
        # Overall Level
        overall = "low"
        if avg_score > 0.8 and confidence > 0.8:
            overall = "high"
        elif avg_score > 0.7 and confidence > 0.6:
            overall = "moderate"
            
        return {
            "overall_level": overall,
            "metrics": [
                {
                    "label": "Source Match",
                    "value": source_match_val,
                    "ratio": round(avg_score, 2)
                },
                {
                    "label": "AI Certainty",
                    "value": ai_certainty_val,
                    "ratio": round(confidence, 2)
                },
                {
                    "label": "Context Density",
                    "value": density_val,
                    "ratio": round(density_ratio, 2)
                }
            ]
        }

    async def analyze_document(
        self,
        question: str,
        document_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Run the full analysis workflow for a document.
        """
        self.logger.info(f"Starting analysis for doc {document_id}, question: {question}")
        
        initial_state = AnalysisState(
            question=question,
            document_id=document_id,
            user_id=user_id,
            context="",
            contexts=[],
            answer="",
            is_valid=False,
            intelligence_hub_data={},
            geopolitical_context="",
            retrieval_scores=[],
            retrieved_sources=[],
            generation_logprobs=[],
            confidence_metrics={},
            current_stage="pending",
            stage_index=0,
            total_stages=4,
            status_message="Analysis queued",
            status="pending"
        )
        
        final_state = await self.workflow.run(initial_state)
        
        # Calculate Confidence Metrics
        final_state["confidence_metrics"] = self._calculate_confidence_metrics(final_state)
        
        return final_state
    
    async def get_analysis_status(self, document_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the current analysis status from S3.
        Returns the progress state or None if not found.
        """
        if not self.object_store:
            return None
            
        try:
            # Try to get the status file
            status_key = f"{user_id}/{document_id}/status.json"
            status_data = self.object_store.get_json(status_key)
            return status_data
        except Exception as e:
            self.logger.debug(f"No status found for {document_id}: {e}")
            return None
