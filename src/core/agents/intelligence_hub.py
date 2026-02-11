from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from src.core.agents.base import BaseAgent
from src.core.workflows.state import AnalysisState
from src.domain.schemas.intelligence_hub import AIIntelligenceHubData

class IntelligenceHub(BaseAgent[AnalysisState]):
    async def execute(self, state: AnalysisState) -> AnalysisState:
        self._log_execution(state)
        context = state.get("context", "")
        question = state.get("question", "")
        answer = state.get("answer", "")
        geopolitical_context = state.get("geopolitical_context", "")
        
        full_context = context
        if geopolitical_context:
            full_context += "\n\n" + geopolitical_context

        parser = PydanticOutputParser(pydantic_object=AIIntelligenceHubData)

        prompt = ChatPromptTemplate.from_template("""
        You are a financial intelligence analyst. Extract REAL financial metrics and insights.
        
        Context:
        {context}
        
        Question: {question}
        Answer: {answer}
        
        Generate a comprehensive intelligence report (Key Highlights, Sentiment, Risk, FAQs).
        Strictly follow the format instructions.
        
        {format_instructions}
        """)

        chain = prompt | self.llm | parser

        try:
            result = await chain.ainvoke({
                "context": full_context,
                "question": question,
                "answer": answer,
                "format_instructions": parser.get_format_instructions()
            })
            state["intelligence_hub_data"] = result.model_dump()
        except Exception as e:
            self.logger.error(f"Error generating intelligence hub data: {e}")
            # Simplified error handling - in production, might want retry or fallback
            state["intelligence_hub_data"] = {}
            
        return state
