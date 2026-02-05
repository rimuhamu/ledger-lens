from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from graph import app as agent_graph

app = FastAPI(title="LedgerLens Analyst API")

class AnalysisRequest(BaseModel):
    ticker: str
    query: str

@app.get("/")
def read_root():
    return {"status": "LedgerLens Analyst is Online", "model": "GPT-4o / LangGraph"}

@app.post("/analyze")
async def analyze_report(request: AnalysisRequest):
    """
    Triggers the multi-step reasoning agent to analyze the BCA 2024 report.
    """
    try:
        inputs = {"question": f"For {request.ticker}: {request.query}"}
        
        result = await agent_graph.ainvoke(inputs)
        
        return {
            "answer": result["answer"],
            "verification_status": "PASS" if result["is_valid"] else "FAIL",
            "metadata": {"source": "BCA Annual Report 2024"}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))