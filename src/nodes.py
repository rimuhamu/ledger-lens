import os
import logging
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from database import Database
from typing import TypedDict, List
from langchain_core.output_parsers import PydanticOutputParser
from analysis_schema import AIIntelligenceHubData
from geopolitical_service import get_geopolitical_service

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
    Uses RAG to find facts in the document + external geopolitical data.
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

    # Extract country/region from context for geopolitical analysis
    geopolitical_context = ""
    try:
        geo_service = get_geopolitical_service()
        
        # Try to identify country from context or question
        country = _extract_country_from_text(context + " " + question)
        
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

def _extract_country_from_text(text: str) -> str:
    """
    Extract country name from text using keyword matching.
    Looks for country names, cities, currencies, stock exchanges, and common identifiers.
    Returns the most likely country mentioned.
    """
    text_lower = text.lower()
    
    # Expanded country patterns with currencies, exchanges, cities, company suffixes
    priority_countries = {
        "indonesia": [
            "indonesia", "indonesian", "jakarta", "bandung", "surabaya",
            "rupiah", "idr", "idx", "bei", "bank indonesia", "ojk",
            "pt ", "tbk", "bca", "bri", "bni", "mandiri", "telkom",
            "pertamina", "garuda"
        ],
        "united states": [
            "united states", "u.s.", "usa", "america", "american",
            "new york", "california", "nasdaq", "nyse", "sec",
            "usd", "dollar", "fed", "federal reserve"
        ],
        "china": [
            "china", "chinese", "beijing", "shanghai", "shenzhen",
            "yuan", "rmb", "cny", "hong kong", "hkd"
        ],
        "singapore": [
            "singapore", "singaporean", "sgx", "sgd", "mas"
        ],
        "malaysia": [
            "malaysia", "malaysian", "kuala lumpur", "klse", "bursa",
            "ringgit", "myr", "bank negara"
        ],
        "thailand": [
            "thailand", "thai", "bangkok", "baht", "thb", "set"
        ],
        "vietnam": [
            "vietnam", "vietnamese", "hanoi", "ho chi minh", "dong", "vnd"
        ],
        "philippines": [
            "philippines", "philippine", "manila", "peso", "php", "pse"
        ],
        "japan": [
            "japan", "japanese", "tokyo", "osaka", "yen", "jpy",
            "nikkei", "tse", "boj"
        ],
        "india": [
            "india", "indian", "mumbai", "delhi", "bangalore",
            "rupee", "inr", "nse", "bse", "rbi"
        ],
        "south korea": [
            "south korea", "korea", "korean", "seoul", "won", "krw", "kospi"
        ],
        "australia": [
            "australia", "australian", "sydney", "melbourne",
            "aud", "asx", "rba"
        ],
        "united kingdom": [
            "united kingdom", "uk", "britain", "british", "london",
            "pound", "sterling", "gbp", "lse", "ftse"
        ],
        "germany": [
            "germany", "german", "frankfurt", "berlin", "munich",
            "euro", "eur", "dax"
        ],
        "france": [
            "france", "french", "paris", "euro", "eur", "cac"
        ],
        "brazil": [
            "brazil", "brazilian", "sao paulo", "real", "brl", "bovespa"
        ],
        "mexico": [
            "mexico", "mexican", "peso", "mxn"
        ],
        "canada": [
            "canada", "canadian", "toronto", "cad", "tsx"
        ]
    }
    
    # Count mentions
    country_scores = {}
    for country, keywords in priority_countries.items():
        score = 0
        for keyword in keywords:
            # Use word boundary matching for short keywords
            if len(keyword) <= 3:
                # For short keywords like "idr", "usd", check word boundaries
                import re
                pattern = r'\b' + re.escape(keyword) + r'\b'
                score += len(re.findall(pattern, text_lower))
            else:
                score += text_lower.count(keyword)
        if score > 0:
            country_scores[country] = score
    
    if country_scores:
        # Return country with most mentions
        return max(country_scores, key=country_scores.get)
    
    return None

def analyst_node(state: Dict[str, Any]):
    """
    Analyst node
    Synthesizes the raw data into a professional thesis.
    """
    print("--- ANALYZING FINANCIAL DATA ---")
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
        1. citations: Every claim you make must include a reference to the specific context chunk (e.g., [Source 1]).
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
    print("--- VALIDATING ANSWER ---")
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
    print(verification.content)
    is_valid = "PASS" in verification.content.upper()

    return {"is_valid": is_valid}

def intelligence_hub_node(state: Dict[str, Any]):
    """
    Intelligence Hub Node
    Generates structured JSON output for the AI Intelligence Hub UI.
    Extracts key highlights, sentiment score, risk level, and risk factors.
    Integrates external geopolitical risk data when available.
    """
    print("--- GENERATING INTELLIGENCE HUB DATA ---")
    context = state["context"]
    question = state.get("question", "")
    answer = state.get("answer", "")
    geopolitical_context = state.get("geopolitical_context", "")
    
    # Combine all context
    full_context = context
    if geopolitical_context:
        full_context += "\n\n" + geopolitical_context

    # Set up the Pydantic output parser
    parser = PydanticOutputParser(pydantic_object=AIIntelligenceHubData)

    prompt = ChatPromptTemplate.from_template("""
You are a financial intelligence analyst. Analyze the provided context and answer to generate a structured intelligence report based ONLY on the actual information found in the documents and external data sources.

Context from financial documents:
{context}

Analysis Question: {question}

Previous Analysis Answer: {answer}

CRITICAL INSTRUCTIONS:
- Extract ALL values directly from the provided context - DO NOT use placeholder or example values
- Base the sentiment score on actual performance metrics found in the documents
- Identify REAL risk factors mentioned in the context OR provided by external geopolitical data sources
- When external geopolitical data is available (marked as "EXTERNAL GEOPOLITICAL RISK DATA"), incorporate those risks into your risk_factors section
- Every number, percentage, and claim must be traceable to the actual context or external data
- If specific information is not available, acknowledge gaps appropriately

Generate a comprehensive intelligence hub report with:

1. **Key Highlights** (3-5 highlights): Extract the ACTUAL most important financial metrics and events from the context.
   - Each highlight must reference REAL data from the documents (actual revenue figures, growth rates, events, etc.)
   - Icon types: "growth" for positive metrics, "chart" for financial metrics, "calendar" for time-based events, "alert" for warnings/challenges, "check" for achievements
   - Include the EXACT metric value from the context (e.g., if the doc says revenue grew 15.3%, use "15.3%")
   - Example format: "Total loans reached Rp921.9 trillion, up 13.8% YoY" (using actual numbers from context)
   - DO NOT use generic statements - be specific with actual figures

2. **Sentiment Score** (0-100): Calculate based ONLY on ACTUAL indicators found in the context:
   - Analyze real financial performance (revenue growth, profitability, margins)
   - Consider actual growth trajectory mentioned in documents
   - Review management's stated outlook if available
   - Evaluate market position based on context
   - Scoring guide: 
     * 0-30 (Bearish): Declining metrics, losses, negative outlook
     * 31-55 (Neutral): Flat growth, mixed signals
     * 56-75 (Moderately Bullish): Solid growth, positive trends
     * 76-100 (Strongly Bullish): Exceptional growth, strong fundamentals
   - Change percentage should reflect actual YoY comparisons if available, otherwise estimate based on trend direction
   - Description must reference specific context evidence (e.g., "Bullish based on 18% loan growth and expanding margins to 72.5%")
   - DO NOT use generic sentiment descriptions

3. **Risk Level**: Overall assessment based on ACTUAL risks mentioned in context or external data
   - "Low": Strong fundamentals with minimal concerns mentioned in documents
   - "Moderate": Some challenges noted but appear manageable based on context
   - "High": Significant risks, challenges, or adverse conditions discussed
   - Explanation must cite specific evidence from context or external sources (e.g., "Moderate due to regulatory changes mentioned in report and recent trade tensions from external data")

4. **Risk Factors** (2-4 factors): Identify SPECIFIC risks from TWO SOURCES:
   
   A. DOCUMENT-BASED RISKS: Extract from the financial documents
   - Look for risks mentioned in risk sections, MD&A, footnotes, or business discussions
   
   B. EXTERNAL GEOPOLITICAL RISKS: If "EXTERNAL GEOPOLITICAL RISK DATA" section is present in context
   - Use the actual risks provided by external APIs (NewsAPI, World Bank, GDELT)
   - These come from real-time data and should be included when available
   - Format them appropriately with correct icons and severity levels
   
   IMPORTANT RULES:
   - DO NOT fabricate generic examples (like "Supply Chain Concentration" unless actually mentioned)
   - External geopolitical risks from APIs are REAL DATA and should be included
   - Prioritize HIGH severity risks, then MED, then LOW
   - Icon selection based on actual risk type identified:
     * "globe": Geopolitical, regulatory, or international risks (use for external geopolitical data)
     * "chain": Supply chain, operational, or logistics risks
     * "dollar": Financial, liquidity, or currency risks
     * "alert": General business or emerging risks
     * "chart": Market volatility or competitive risks
   - Severity from external data sources is already provided - use it as-is
   - If a risk comes from external data, you can reference it like: "Economic Sanctions [HIGH] - Real-time geopolitical monitoring indicates..."
   
   Examples of proper extraction:
   - Document mentions "credit risk increased due to NPL": {{"icon": "dollar", "name": "Credit Risk Exposure", "severity": "HIGH"}}
   - External data shows "Economic Sanctions [HIGH]": {{"icon": "globe", "name": "Economic Sanctions", "severity": "HIGH"}}
   - Document mentions "supply chain pressures": {{"icon": "chain", "name": "Supply Chain Pressures", "severity": "MED"}}

5. **Suggested Questions** (3 questions): Generate relevant follow-up questions based on:
   - Specific topics mentioned in the context that could use deeper analysis
   - Related financial aspects that would naturally follow from the current analysis
   - Areas partially covered that could be expanded (e.g., if growth is mentioned, ask about drivers)
   - If external geopolitical risks were identified, suggest questions to explore their impact
   - Make questions SPECIFIC to the actual company/report, not generic templates
   - Examples: 
     * "What drove the 13.8% loan growth in 2024?" (if loan growth mentioned)
     * "How might recent trade sanctions impact revenue in key markets?" (if geopolitical risks found)
     * NOT: "What are the company's growth strategies?" (too generic)

{format_instructions}

FINAL REMINDER: 
- Document-based data must be traceable to the financial report context
- External geopolitical risks from APIs are REAL-TIME DATA from NewsAPI, World Bank, and GDELT - include them when available
- Do not fabricate risks that aren't in either source
- If context lacks specific information for document-based risks, acknowledge it rather than inventing placeholder data
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
        
        # Log the sources used
        if geopolitical_context:
            logger.info(f"Intelligence Hub generated with external geopolitical data")
            logger.info(f"Risk factors identified: {len(intelligence_data.get('risk_factors', []))}")
        
        return {"intelligence_hub_data": intelligence_data}
        
    except Exception as e:
        logger.error(f"Error generating intelligence hub data: {e}")
        # Return a minimal structure on error with clear indication of failure
        return {
            "intelligence_hub_data": {
                "key_highlights": [
                    {
                        "icon": "alert",
                        "text": "Unable to extract highlights from the provided context",
                        "metric_value": None
                    }
                ],
                "sentiment": {
                    "score": 50, 
                    "change": "N/A", 
                    "description": "Insufficient data to analyze sentiment"
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
                    "Please provide more specific financial documents for analysis"
                ]
            }
        }