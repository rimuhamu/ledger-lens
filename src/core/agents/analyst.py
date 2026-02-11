from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from src.core.agents.base import BaseAgent
from src.core.workflows.state import AnalysisState

class Analyst(BaseAgent[AnalysisState]):
    async def execute(self, state: AnalysisState) -> AnalysisState:
        self._log_execution(state)
        context = state.get("context", "")
        question = state["question"]
        geopolitical_context = state.get("geopolitical_context", "")
        
        full_context = context
        if geopolitical_context:
            full_context += "\n\n" + geopolitical_context

        prompt = ChatPromptTemplate.from_template("""
            You are a strict financial analyst assistant. 
            Your task is to answer the user's question based ONLY on the provided context below.

            Context:
            {context}

            Question: 
            {question}

            Instructions:
            1. Citations: Every claim you make must include a reference to the specific context chunk.
            2. No Outside Knowledge: If the answer is not in the context, strictly state you cannot answer.
            3. Do not speculate.
            4. Incorporate geopolitical risk data if provided.

            Answer:
        """)

        chain = prompt | self.llm
        response = await chain.ainvoke({"context": full_context, "question": question})
        
        state["answer"] = response.content
        return state
