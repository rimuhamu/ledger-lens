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

        # Bind logprobs to the LLM
        # LangChain ChatOpenAI supports logprobs=True
        llm_with_logprobs = self.llm.bind(logprobs=True)
        chain = prompt | llm_with_logprobs
        
        response = await chain.ainvoke({"context": full_context, "question": question})
        
        # Extract logprobs from response metadata
        # Structure depends on the provider, for OpenAI it's usually in response.response_metadata['logprobs']['content']
        logprobs_data = []
        if hasattr(response, 'response_metadata'):
            meta = response.response_metadata
            if 'logprobs' in meta and 'content' in meta['logprobs']:
                # content is a list of dicts, each having 'logprob'
                for item in meta['logprobs']['content']:
                    if 'logprob' in item:
                        logprobs_data.append(item['logprob'])
        
        state["answer"] = response.content
        state["generation_logprobs"] = logprobs_data
        return state
