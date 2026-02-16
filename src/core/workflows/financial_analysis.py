from typing import Any, Dict
from langgraph.graph import StateGraph, START, END
from src.core.workflows.state import AnalysisState
from src.core.agents.researcher import Researcher
from src.core.agents.analyst import Analyst
from src.core.agents.validator import Validator
from src.core.agents.intelligence_hub import IntelligenceHub

class FinancialAnalysisWorkflow:
    def __init__(
        self,
        researcher: Researcher,
        analyst: Analyst,
        validator: Validator,
        intelligence_hub: IntelligenceHub,
        object_store=None
    ):
        self.researcher = researcher
        self.analyst = analyst
        self.validator = validator
        self.intelligence_hub = intelligence_hub
        self.object_store = object_store
        
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AnalysisState)
        
        # Add nodes
        workflow.add_node("researcher", self._research_node)
        workflow.add_node("analyst", self._analyst_node)
        workflow.add_node("validator", self._validator_node)
        workflow.add_node("intelligence_hub", self._intelligence_hub_node)
        
        # Define edges
        workflow.add_edge(START, "researcher")
        workflow.add_edge("researcher", "analyst")
        workflow.add_edge("analyst", "validator")
        
        workflow.add_conditional_edges(
            "validator",
            self._check_validation
        )
        
        workflow.add_edge("intelligence_hub", END)
        
        return workflow.compile()
    
    # Tiny wrappers to match LangGraph node signature if needed
    # But BaseAgent.execute is async and takes state, which matches standard async node pattern 
    # if we wrap it slightly or if LangGraph supports it directly.
    # To be safe, we wrap.
    
    def _save_progress(self, state: AnalysisState):
        """Save progress state to S3 for status tracking."""
        if not self.object_store:
            return
        
        document_id = state.get("document_id")
        user_id = state.get("user_id")
        
        if not document_id or not user_id:
            return
        
        # Save minimal progress data
        progress_data = {
            "status": state.get("status", "in_progress"),
            "current_stage": state.get("current_stage", ""),
            "stage_index": state.get("stage_index", 0),
            "total_stages": state.get("total_stages", 4),
            "status_message": state.get("status_message", "")
        }
        
        status_key = f"{user_id}/{document_id}/status.json"
        self.object_store.save_json(progress_data, status_key)
    
    async def _research_node(self, state: AnalysisState) -> Dict:
        # LangGraph nodes usually return a dict of updates
        # Update progress before execution
        state["current_stage"] = "research"
        state["stage_index"] = 0
        state["total_stages"] = 4
        state["status_message"] = "Retrieving relevant documents from vector store"
        state["status"] = "in_progress"
        
        # Save progress
        self._save_progress(state)
        
        new_state = await self.researcher.execute(state)
        return new_state

    async def _analyst_node(self, state: AnalysisState) -> Dict:
        # Update progress
        state["current_stage"] = "analysis"
        state["stage_index"] = 1
        state["status_message"] = "Analyzing document with AI model"
        
        # Save progress
        self._save_progress(state)
        
        new_state = await self.analyst.execute(state)
        return new_state

    async def _validator_node(self, state: AnalysisState) -> Dict:
        # Update progress
        state["current_stage"] = "validation"
        state["stage_index"] = 2
        state["status_message"] = "Validating analysis results"
        
        # Save progress
        self._save_progress(state)
        
        new_state = await self.validator.execute(state)
        return new_state

    async def _intelligence_hub_node(self, state: AnalysisState) -> Dict:
        # Update progress
        state["current_stage"] = "intelligence"
        state["stage_index"] = 3
        state["status_message"] = "Generating intelligence hub data and insights"
        
        new_state = await self.intelligence_hub.execute(state)
        
        # Mark as complete after intelligence hub
        new_state["current_stage"] = "complete"
        new_state["status"] = "completed"
        new_state["status_message"] = "Analysis completed successfully"
        
        # Save final progress
        self._save_progress(new_state)
        
        return new_state

    def _check_validation(self, state: AnalysisState):
        if state.get("is_valid"):
            return "intelligence_hub"
        return "researcher" 

    async def run(self, input_state: AnalysisState):
        return await self.graph.ainvoke(input_state)
