import os
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from database import Database
from typing import TypedDict, List

# Define the state structure
class AgentState(TypedDict):
    question: str
    context: str
    contexts: List[str]  # RAGAS expects a list of context strings
    answer: str
    is_valid: bool

# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db = Database(db_path=os.path.join(PROJECT_ROOT, "data", "vectorstore"))
_default_retriever = db.get_retriever()
_current_retriever = _default_retriever

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

def set_retriever(new_retriever):
    """Set a custom retriever (e.g., for uploaded documents)"""
    global _current_retriever
    _current_retriever = new_retriever

def reset_retriever():
    """Reset to the default retriever"""
    global _current_retriever
    _current_retriever = _default_retriever

def get_current_retriever():
    """Get the currently active retriever"""
    return _current_retriever

def research_node(state: Dict[str, Any]):
    """
    Research node
    Uses RAG to find facts in the document.
    """
    print("--- RESEARCHING FINANCIAL DATA ---")
    question = state["question"]

    docs = retriever.invoke(question)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    contexts = [doc.page_content for doc in docs]  # List format for RAGAS

    return {
        "context": context,
        "contexts": contexts
    }

def analyst_node(state: Dict[str, Any]):
    """
    Analyst node
    Synthesizes the raw data into a professional thesis.
    """
    print("--- ANALYZING FINANCIAL DATA ---")
    context = state["context"]
    question = state["question"]

    prompt = ChatPromptTemplate.from_template("""
        You are a strict financial analyst assistant. 
        Your task is to answer the user's question based ONLY on the provided context below.

        Context:
        {context}

        Question: 
        {question}

        Instructions:
        1. citations: Every claim you make must include a reference to the specific context chunk (e.g., [Source 1]).
        2. No Outside Knowledge: If the answer is not in the context, strictly state: "I cannot answer this based on the provided documents."
        3. Do not speculate or make up numbers.

        Answer:
    """)

    chain = prompt | llm

    response = chain.invoke({"context": context, "question": question})
    return {"answer": response.content}

def validator_node(state: Dict[str, Any]):
    """
    Validator node
    Checks for hallucinations or missing data.
    """
    print("--- VALIDATING ANSWER ---")
    answer = state["answer"]
    context = state["context"]

    prompt = ChatPromptTemplate.from_template("""
    You are a Quality Controller for Financial AI. 
    Verify if the following answer is fully supported by the context.
    
    Context: {context}
    Answer: {answer}
    
    If the answer contains figures NOT found in the context, or if it says it cannot find 
    information that IS in the context, respond with 'FAIL'. 
    Otherwise, respond with 'PASS'.
    """)

    chain = prompt | llm

    verification = chain.invoke({"context": context, "answer": answer})
    print(verification.content)
    is_valid = "PASS" in verification.content.upper()

    return {"is_valid": is_valid}