import os
import logging
from typing import Any, Dict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, List
from langchain_core.output_parsers import PydanticOutputParser
from analysis_schema import AIIntelligenceHubData
from geopolitical_service import get_geopolitical_service

load_dotenv()

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
    geopolitical_context: str  # Additional context from external APIs
    document_id: str  # Optional document ID for saved documents

# Database instance will be injected from main.py
_db_instance = None

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


def set_database(db):
    """Set the database instance (called from main.py)"""
    global _db_instance
    _db_instance = db


def get_database():
    """Get the current database instance"""
    if _db_instance is None:
        # Lazy import to avoid circular dependency
        from database import Database
        return Database()
    return _db_instance


def research_node(state: Dict[str, Any]):
    """
    Research node
    Uses Pinecone to find facts in the document + external geopolitical data.
    """
    logger.info("--- RESEARCHING FINANCIAL DATA ---")
    question = state["question"]
    document_id = state.get("document_id")
    logger.info(f"Question: {question}")
    
    # Get database instance
    db = get_database()
    
    # Query Pinecone
    chunks = db.query_documents(
        query=question,
        document_id=document_id,
        top_k=8
    )
    
    logger.info(f"Retrieved {len(chunks)} chunks from Pinecone")
    
    context = "\n\n".join([chunk["content"] for chunk in chunks])
    contexts = [chunk["content"] for chunk in chunks]  # List format for RAGAS
    logger.info(f"Total context length: {len(context)} characters")

    # Extract country/region from context for geopolitical analysis
    geopolitical_context = ""
    try:
        geo_service = get_geopolitical_service()
        
        # Try to identify country from ticker and filename using LLM
        ticker = chunks[0].get("ticker") if chunks else None
        filename = chunks[0].get("filename") if chunks else None
        
        country = _identify_country_with_llm(ticker, filename)
        
        if country:
            logger.info(f"Fetching geopolitical risks for: {country}")
            geo_risks = geo_service.get_country_risks(country)
            
            if geo_risks:
                geopolitical_context = "\n\n--- EXTERNAL GEOPOLITICAL RISK DATA ---\n"
                geopolitical_context += f"Real-time geopolitical risks for {country} (from external sources):\n"
                for risk in geo_risks:
                    geopolitical_context += f"- {risk['name']} [{risk['severity']}]: {risk.get('description', '')[:200]}\n"
                    geopolitical_context += f"  Source: {risk['source']}, Date: {risk.get('date', 'Recent')}\n"
                
                logger.info(f"Added {len(geo_risks)} external geopolitical risk factors")
            else:
                logger.info("No external geopolitical risks found")
        else:
            logger.info("Could not identify country for geopolitical analysis")
            
    except Exception as e:
        logger.warning(f"Failed to fetch geopolitical data: {e}")
        geopolitical_context = ""

    return {
        "context": context,
        "contexts": contexts,
        "geopolitical_context": geopolitical_context
    }


def _identify_country_with_llm(ticker: str = None, filename: str = None) -> str:
    """
    Identify country using LLM based on ticker and filename.
    Removes hardcoded lists in favor of dynamic AI reasoning.
    """
    if not ticker and not filename:
        return None
        
    try:
        prompt = ChatPromptTemplate.from_template("""
        Identify the primary country where this company is headquartered based on the following information:
        Ticker: {ticker}
        Filename: {filename}
        
        Return ONLY the country name in lowercase (e.g., "united states", "indonesia", "china").
        If you are unsure or cannot identify the country, return "None".
        Do not add any explanation or punctuation.
        """)
        
        chain = prompt | llm
        
        response = chain.invoke({
            "ticker": ticker if ticker else "Unknown",
            "filename": filename if filename else "Unknown"
        })
        
        country = response.content.strip().lower()
        
        if country == "none" or not country:
            return None
            
        logger.info(f"LLM identified country: {country} (Ticker: {ticker}, File: {filename})")
        return country
        
    except Exception as e:
        logger.error(f"Error in AI country identification: {e}")
        return None


def analyst_node(state: Dict[str, Any]):
    """
    Analyst node
    Synthesizes the raw data into a professional thesis.
    """
    logger.info("--- ANALYZING FINANCIAL DATA ---")
    context = state["context"]
    question = state["question"]
    geopolitical_context = state.get("geopolitical_context", "")
    
    # Combine document context with external geopolitical data
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
        1. Citations: Every claim you make must include a reference to the specific context chunk (e.g., [Source 1]).
        2. No Outside Knowledge: If the answer is not in the context, strictly state: "I cannot answer this based on the provided documents."
        3. Do not speculate or make up numbers.
        4. If external geopolitical risk data is provided, incorporate it into your analysis and cite the source.

        Answer:
    """)

    chain = prompt | llm

    response = chain.invoke({"context": full_context, "question": question})
    return {"answer": response.content}


def validator_node(state: Dict[str, Any]):
    """
    Validator node
    Checks for hallucinations or missing data.
    """
    logger.info("--- VALIDATING ANSWER ---")
    answer = state["answer"]
    context = state["context"]
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

    chain = prompt | llm

    verification = chain.invoke({"context": full_context, "answer": answer})
    logger.info(f"Validation result: {verification.content}")
    is_valid = "PASS" in verification.content.upper()

    return {"is_valid": is_valid}


def intelligence_hub_node(state: Dict[str, Any]):
    """
    Intelligence Hub Node
    Generates structured JSON output for the AI Intelligence Hub UI.
    Extracts key highlights, sentiment score, risk level, and risk factors.
    Integrates external geopolitical risk data when available.
    """
    logger.info("--- GENERATING INTELLIGENCE HUB DATA ---")
    context = state["context"]
    question = state.get("question", "")
    answer = state.get("answer", "")
    geopolitical_context = state.get("geopolitical_context", "")
    
    # Combine all context
    full_context = context
    if geopolitical_context:
        full_context += "\n\n" + geopolitical_context
    
    # Log what we're sending to the LLM for debugging
    logger.info(f"Context length: {len(full_context)} characters")
    logger.info(f"Context preview (first 500 chars):\n{full_context[:500]}")
    logger.info(f"Context preview (last 500 chars):\n{full_context[-500:]}")

    # Set up the Pydantic output parser
    parser = PydanticOutputParser(pydantic_object=AIIntelligenceHubData)

    prompt = ChatPromptTemplate.from_template("""
You are a financial intelligence analyst. Your task is to extract REAL financial metrics and insights from the provided context.

**CRITICAL RULES:**
1. ONLY use numbers and facts that ACTUALLY appear in the context below
2. DO NOT invent, estimate, or use placeholder values
3. If a specific metric isn't in the context, find similar metrics that ARE present
4. Be flexible - extract whatever financial metrics are available

Context from financial documents:
{context}

Analysis Question: {question}
Previous Analysis Answer: {answer}

**YOUR TASK:**

Extract a comprehensive intelligence report with the following sections:

## 1. KEY HIGHLIGHTS (3-5 items)

Look for ANY of these types of metrics in the context:
- Revenue figures and growth rates
- Profit/earnings numbers  
- Margin percentages (gross margin, operating margin, net margin)
- Cash flow data
- Major business events (acquisitions, product launches, restructuring)
- Segment performance data
- Key operational metrics

**INSTRUCTIONS:**
- For EACH highlight, include the EXACT number/percentage from the context in the `metric_value` field
- The `text` field should explain the metric in context
- Choose appropriate icons:
  * "growth" - for revenue/growth metrics
  * "chart" - for margins/profitability  
  * "dollar" - for cash/financial metrics
  * "calendar" - for time-based events
  * "check" - for achievements/milestones

**Example (if context mentions "Revenue was $130.5 billion, up 114% year-over-year"):**
{{
  "icon": "growth",
  "text": "Revenue surged to $130.5 billion, driven by exceptional data center demand",
  "metric_value": "114%"
}}

## 2. SENTIMENT SCORE (0-100)

Base the sentiment score on ACTUAL performance indicators in the context:
- Look for growth rates, profitability trends, management commentary
- Positive indicators: high growth rates (>20%), expanding margins, positive outlook statements
- Negative indicators: declining growth, shrinking margins, risk warnings

Scoring guide:
- 76-100: Exceptional growth (>50% YoY), strong profitability, very positive outlook
- 56-75: Solid growth (10-50% YoY), healthy margins, positive trends
- 31-55: Moderate growth (0-10%), mixed signals
- 0-30: Declining metrics, losses, negative outlook

**The `change` field** should reflect YoY sentiment shift if mentioned, or estimate based on growth trends.

**The `description` field** must cite specific evidence from context (e.g., "Bullish based on 114% revenue growth and 75% gross margin expansion")

## 3. RISK LEVEL

Choose ONE: "Low", "Moderate", or "High"

Base this on:
- Business risks mentioned in the document
- Margin pressures or declining trends
- Competitive/regulatory concerns cited
- External geopolitical risks (if provided)

The `description` must reference specific risks found in the context.

## 4. RISK FACTORS (2-4 items)

Extract SPECIFIC risks mentioned in:
A) The financial documents (look for risk sections, MD&A, footnotes)
B) External geopolitical data (if the "EXTERNAL GEOPOLITICAL RISK DATA" section exists)

**STRICT RULES:**
- Only include risks explicitly mentioned in the context or external data
- DO NOT add generic risks like "Supply Chain Concentration" unless specifically discussed
- External geopolitical risks from APIs are REAL DATA - include them when present
- Use severity from external sources as-is

Icon guide:
- "globe" - geopolitical, regulatory, international risks
- "chain" - supply chain, operational risks  
- "dollar" - financial, liquidity, currency risks
- "alert" - general business risks
- "chart" - market volatility, competitive risks

## 5. SUGGESTED QUESTIONS (3 questions)

Generate follow-up questions based on:
- Topics mentioned but not fully explored
- Specific business segments or products referenced
- Financial metrics that could be analyzed further
- Geopolitical risks that might impact the business

Make questions SPECIFIC to this document - use actual company names, product lines, or metrics mentioned.

**BAD (too generic):** "What are the company's growth strategies?"
**GOOD:** "How is the Data Center segment performing relative to Graphics?"

{format_instructions}

**FINAL CHECKLIST BEFORE RESPONDING:**
☐ Every metric_value contains an ACTUAL number from the context (not "N/A" or placeholder)
☐ Sentiment score is justified by specific evidence from context
☐ Risk factors are either from the document OR external data sources (not invented)
☐ Suggested questions reference specific topics from THIS document
☐ If certain data isn't available, I've adapted to extract what IS available rather than using placeholders

Now analyze the context and generate the intelligence report.
""")

    chain = prompt | llm | parser

    try:
        result = chain.invoke({
            "context": full_context,
            "question": question,
            "answer": answer,
            "format_instructions": parser.get_format_instructions()
        })
        
        intelligence_data = result.model_dump()
        
        # Log what was extracted
        logger.info("✓ Intelligence Hub data generated successfully")
        logger.info(f"  - Key highlights: {len(intelligence_data.get('key_highlights', []))}")
        logger.info(f"  - Sentiment score: {intelligence_data.get('sentiment', {}).get('score', 'N/A')}")
        logger.info(f"  - Risk level: {intelligence_data.get('risk', {}).get('level', 'N/A')}")
        logger.info(f"  - Risk factors: {len(intelligence_data.get('risk_factors', []))}")
        
        # Debug: Log actual extracted highlights
        for i, highlight in enumerate(intelligence_data.get('key_highlights', []), 1):
            logger.info(f"  Highlight {i}: {highlight.get('text', '')[:80]}... (metric: {highlight.get('metric_value', 'None')})")
        
        if geopolitical_context:
            logger.info("  ✓ Integrated external geopolitical data")
        
        return {"intelligence_hub_data": intelligence_data}
        
    except Exception as e:
        logger.error(f"Error generating intelligence hub data: {e}", exc_info=True)
        logger.error(f"Context that caused error (first 1000 chars): {full_context[:1000]}")
        
        # Return a minimal error structure
        return {
            "intelligence_hub_data": {
                "key_highlights": [
                    {
                        "icon": "alert",
                        "text": f"Error extracting highlights: {str(e)[:100]}",
                        "metric_value": None
                    }
                ],
                "sentiment": {
                    "score": 50, 
                    "change": None, 
                    "description": "Unable to analyze sentiment - see error details"
                },
                "risk": {
                    "level": "Moderate", 
                    "description": "Unable to assess risk from available context"
                },
                "risk_factors": [
                    {
                        "icon": "alert",
                        "name": "Analysis Error",
                        "severity": "MED"
                    }
                ],
                "suggested_questions": [
                    "Please check the PDF extraction quality",
                    "Try uploading the document again",
                    "Contact support if the issue persists"
                ]
            }
        }