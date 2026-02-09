import os
import logging
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from database import Database
from typing import TypedDict, List
from langchain_core.output_parsers import PydanticOutputParser
from analysis_schema import AIIntelligenceHubData

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the state structure
class AgentState(TypedDict):
    question: str
    context: str
    contexts: List[str]  # RAGAS expects a list of context strings
    answer: str
    is_valid: bool
    intelligence_hub_data: dict  # Structured data for AI Intelligence Hub UI

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
    logger.info("--- RESEARCHING FINANCIAL DATA ---")
    question = state["question"]
    logger.info(f"Question: {question}")

    retriever = get_current_retriever()
    docs = retriever.invoke(question)
    logger.info(f"Retrieved {len(docs)} documents")
    
    context = "\n\n".join([doc.page_content for doc in docs])
    contexts = [doc.page_content for doc in docs]  # List format for RAGAS
    logger.info(f"Total context length: {len(context)} characters")

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

def intelligence_hub_node(state: Dict[str, Any]):
    """
    Intelligence Hub Node
    Generates structured JSON output for the AI Intelligence Hub UI.
    Extracts key highlights, sentiment score, risk level, and risk factors.
    """
    print("--- GENERATING INTELLIGENCE HUB DATA ---")
    context = state["context"]
    question = state.get("question", "")
    answer = state.get("answer", "")

    # Set up the Pydantic output parser
    parser = PydanticOutputParser(pydantic_object=AIIntelligenceHubData)

    prompt = ChatPromptTemplate.from_template("""
You are a financial intelligence analyst. Analyze the provided context and answer to generate a structured intelligence report.

Context from financial documents:
{context}

Analysis Question: {question}

Previous Analysis Answer: {answer}

Based on the above, generate a comprehensive intelligence hub report with:

1. **Key Highlights** (3-5 highlights): Extract the most important financial metrics and events.
   - Each highlight should have an icon type: "growth" for positive growth, "chart" for metrics, "calendar" for dates/events, "alert" for warnings, "check" for achievements
   - Include the specific metric value when available (e.g., "26.2%", "$5.2B")

2. **Sentiment Score** (0-100): Rate the overall sentiment based on:
   - Financial performance indicators
   - Growth trajectory
   - Management outlook
   - Market positioning
   - Provide the change as a percentage (e.g., "+12%" for bullish, "-8%" for bearish)
   - Describe the sentiment briefly (e.g., "Strongly Bullish outlook based on R&D pipeline")

3. **Risk Level**: Overall assessment - "Low", "Moderate", or "High"
   - Provide a brief explanation

4. **Risk Factors** (2-4 factors): Identify specific risks with severity levels
   - Each risk should have an icon: "globe" for geopolitical, "chain" for supply chain, "dollar" for financial, "alert" for general, "chart" for market
   - Severity must be: "LOW", "MED", or "HIGH"

5. **Suggested Questions** (3 questions): Based on the analysis, suggest follow-up questions the user might want to ask.

{format_instructions}
""")

    chain = prompt | llm | parser

    try:
        result = chain.invoke({
            "context": context,
            "question": question,
            "answer": answer,
            "format_instructions": parser.get_format_instructions()
        })
        return {"intelligence_hub_data": result.model_dump()}
    except Exception as e:
        logger.error(f"Error generating intelligence hub data: {e}")
        # Return a default structure on error
        return {
            "intelligence_hub_data": {
                "key_highlights": [],
                "sentiment": {"score": 50, "change": "0%", "description": "Unable to analyze sentiment"},
                "risk": {"level": "Moderate", "description": "Analysis unavailable"},
                "risk_factors": [],
                "suggested_questions": ["What is the company's financial performance?"]
            }
        }