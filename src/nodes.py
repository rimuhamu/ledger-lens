import os
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from database import Database

db = Database()
retriever = db.get_retriever()
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

def research_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Research node
    Uses RAG to find facts in the document.
    """
    print("--- RESEARCHING FINANCIAL DATA ---")
    question = state["question"]

    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    return {"context": context}

def analyst_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyst node
    Synthesizes the raw data into a professional thesis.
    """
    print("--- ANALYZING FINANCIAL DATA ---")
    context = state["context"]
    question = state["question"]

    prompt = ChatPromptTemplate.from_template("""
    You are a Senior Investment Analyst at Scalable Capital. 
    Use the following snippets from the BCA 2024 Annual Report to answer the query.
    
    Context: {context}
    Question: {question}
    
    Format your answer with:
    1. Key Metric (e.g., Loan Growth, CASA Ratio)
    2. Analysis (What does this mean for investors?)
    3. Source Reference (Page number or section if available)
    """)

    chain = prompt | llm

    response = chain.invoke({"context": context, "question": question})
    return {"answer": response.content}

def validator_node(state: Dict[str, Any]) -> Dict[str, Any]:
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