from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from src.core.agents.base import BaseAgent
from src.core.workflows.state import AnalysisState

class Validator(BaseAgent[AnalysisState]):
    async def execute(self, state: AnalysisState) -> AnalysisState:
        self._log_execution(state)
        answer = state.get("answer", "")
        context = state.get("context", "")
        geopolitical_context = state.get("geopolitical_context", "")
        
        full_context = context
        if geopolitical_context:
            full_context += "\n\n" + geopolitical_context

        prompt = ChatPromptTemplate.from_template("""
        You are a Quality Controller for Financial AI. 
        Verify if the following answer is fully supported by the context.
        
        Context: {context}
        Answer: {answer}
        
        If the answer contains figures NOT found in the context, or if it says it cannot find 
        information that IS in the context, respond with 'FAIL'. 
        Otherwise, respond with 'PASS'.
        """)

        chain = prompt | self.llm
        verification = await chain.ainvoke({"context": full_context, "answer": answer})
        
        state["is_valid"] = "PASS" in verification.content.upper()
        return state
