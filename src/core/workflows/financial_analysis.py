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
        intelligence_hub: IntelligenceHub
    ):
        self.researcher = researcher
        self.analyst = analyst
        self.validator = validator
        self.intelligence_hub = intelligence_hub
        
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
    
    async def _research_node(self, state: AnalysisState) -> Dict:
        # LangGraph nodes usually return a dict of updates
        new_state = await self.researcher.execute(state)
        return new_state

    async def _analyst_node(self, state: AnalysisState) -> Dict:
        new_state = await self.analyst.execute(state)
        return new_state

    async def _validator_node(self, state: AnalysisState) -> Dict:
        new_state = await self.validator.execute(state)
        return new_state

    async def _intelligence_hub_node(self, state: AnalysisState) -> Dict:
        new_state = await self.intelligence_hub.execute(state)
        return new_state

    def _check_validation(self, state: AnalysisState):
        if state.get("is_valid"):
            return "intelligence_hub"
        return "researcher" 

    async def run(self, input_state: AnalysisState):
        return await self.graph.ainvoke(input_state)
