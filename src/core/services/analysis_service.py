from typing import Dict, Any, Optional
from src.core.agents.researcher import Researcher
from src.core.agents.analyst import Analyst
from src.core.agents.validator import Validator
from src.core.agents.intelligence_hub import IntelligenceHub
from src.core.workflows.financial_analysis import FinancialAnalysisWorkflow
from src.core.workflows.state import AnalysisState
from src.infrastructure.storage.vector.base import VectorStore
from src.utils.logger import get_logger

class AnalysisService:
    def __init__(self, vector_store: VectorStore):
        self.logger = get_logger(self.__class__.__name__)
        
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
            self.intelligence_hub
        )

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
            geopolitical_context=""
        )
        
        final_state = await self.workflow.run(initial_state)
        
        return final_state
