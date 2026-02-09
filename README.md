# LedgerLens 📊

A multi-step reasoning agent for analyzing financial reports using RAG (Retrieval-Augmented Generation) and LangGraph.

## Overview

LedgerLens is an AI-powered financial analyst that leverages a multi-agent architecture to:
- **Research** relevant data from financial documents using vector search
- **Analyze** and synthesize findings into professional insights
- **Validate** answers to prevent hallucinations and ensure accuracy

Built with LangChain, LangGraph, and ChromaDB, it provides a FastAPI endpoint for querying financial reports with built-in verification.

## Features

- 🔍 **RAG-based Document Search** — Semantic retrieval from PDF financial reports
- 📤 **Dynamic Document Upload** — Upload and analyze any PDF report via API
- 🤖 **Multi-Agent Workflow** — Research → Analyst → Validator → Intelligence Hub pipeline
- 🧠 **AI Intelligence Hub** — Structured output with sentiment scores, risk levels, and key highlights
- 🌐 **Real-time Geopolitical Risks** — External data from NewsAPI, World Bank, and GDELT
- ✅ **Built-in Verification** — Automatic hallucination detection and answer validation
- 📈 **RAGAS Evaluation** — Comprehensive evaluation suite with industry-standard metrics
- 🚀 **FastAPI Backend** — RESTful API for seamless integration

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────────┐
│  Researcher │ ──▶ │   Analyst    │ ──▶ │  Validator  │ ──▶ │ Intelligence Hub │
│   (RAG)     │     │   (LLM)      │     │   (LLM)     │     │     (LLM)        │
└─────────────┘     └──────────────┘     └──────┬──────┘     └────────┬─────────┘
                                                │                     │
                                               FAIL                  PASS
                                                │                     │
                                                ▼                     ▼
                                        [Re-research]              [END]
                                                                     │
                                                         ┌───────────┴───────────┐
                                                         │  Structured Output:   │
                                                         │  • Key Highlights     │
                                                         │  • Sentiment Score    │
                                                         │  • Risk Level         │
                                                         │  • Risk Factors       │
                                                         └───────────────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- OpenAI API key

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/rimuhamu/ledger-lens.git
   cd ledger-lens
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your-openai-api-key
   NEWS_API_KEY=your-newsapi-key  # Optional: for real-time geopolitical news
   ```

5. **Run setup script**
   
   Downloads the sample financial report and initializes the vector database:
   ```bash
   python scripts/setup.py
   ```

## Usage

### Start the API Server

```bash
cd src
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. Health Check
```bash
GET /
```

#### 2. Analyze Pre-loaded Document (BCA Report)
```bash
POST /analyze
Content-Type: application/json

{
  "ticker": "BBCA",
  "query": "What was the total loan portfolio in 2024?"
}
```

**Response:**
```json
{
  "answer": "BCA's total loan portfolio in 2024 was Rp921.9 trillion...",
  "verification_status": "PASS",
  "metadata": {
    "source": "BCA Annual Report 2024"
  }
}
```

#### 3. Upload & Analyze Any Report
```bash
POST /upload-and-analyze
Content-Type: multipart/form-data

file: <your-report.pdf>
ticker: AAPL
```

**Using cURL:**
```bash
curl -X POST "http://localhost:8000/upload-and-analyze" \
  -F "file=@/path/to/report.pdf" \
  -F "ticker=AAPL"
```

**Using Python:**
```python
import requests

url = "http://localhost:8000/upload-and-analyze"
files = {'file': open('report.pdf', 'rb')}
data = {'ticker': 'AAPL'}

response = requests.post(url, files=files, data=data)
print(response.json())
```

**Response:**
```json
{
  "answer": "Apple's total revenue in 2024 was $394.3 billion...",
  "verification_status": "PASS",
  "intelligence_hub": {
    "key_highlights": [
      {
        "icon": "growth",
        "text": "Revenue increased by 26.2% YoY, driven by Services growth.",
        "metric_value": "26.2%"
      }
    ],
    "sentiment": {
      "score": 84,
      "change": "+12%",
      "description": "Strongly Bullish outlook based on product pipeline."
    },
    "risk": {
      "level": "Low",
      "description": "Supply chain diversification mitigates risks."
    },
    "risk_factors": [
      {"icon": "globe", "name": "Geopolitical Restrictions", "severity": "MED"},
      {"icon": "chain", "name": "Supply Chain Concentration", "severity": "LOW"}
    ],
    "suggested_questions": [
      "Summarize the Services segment performance",
      "What are the major capital expenditure plans?",
      "Explain the R&D spending growth"
    ]
  },
  "metadata": {
    "source": "apple_annual_report.pdf",
    "ticker": "AAPL"
  }
}
```

### Run Evaluation

Evaluate the agent using RAGAS metrics:

```bash
cd src
python eval.py
```

This runs the agent against predefined test cases and outputs metrics including:
- **Answer Relevancy** — How relevant is the answer to the question?
- **Faithfulness** — Is the answer grounded in the retrieved context?
- **Context Recall** — How much of the ground truth is captured?
- **Context Precision** — Are relevant contexts ranked higher?

## Project Structure

```
ledger-lens/
├── data/
│   ├── raw/              # PDF source documents
│   ├── uploads/          # Temporary uploads (auto-cleanup)
│   └── vectorstore/      # ChromaDB vector database
├── scripts/
│   └── setup.py          # Data download and indexing
├── src/
│   ├── analysis_schema.py     # Pydantic models for Intelligence Hub
│   ├── database.py            # Vector store operations
│   ├── geopolitical_service.py # External risk data fetching
│   ├── graph.py               # LangGraph workflow definition
│   ├── main.py                # FastAPI application
│   ├── nodes.py               # Agent node implementations
│   └── eval.py                # RAGAS evaluation suite
├── .env                  # Environment variables
├── requirements.txt      # Python dependencies
└── README.md
```

## How Upload & Analyze Works

1. **Upload**: User sends PDF via multipart/form-data with ticker symbol
2. **Temporary Storage**: File saved to `data/uploads/`
3. **Vectorization**: Document is chunked and embedded into a temporary vector store
4. **Geopolitical Enrichment**: Country detected from document, external risk data fetched
5. **Analysis**: LangGraph workflow processes the document
   - Researcher retrieves relevant context + external geopolitical data
   - Analyst synthesizes the answer with risk awareness
   - Validator checks for hallucinations
   - Intelligence Hub extracts structured insights
6. **Cleanup**: Temporary files and vector store automatically deleted
7. **Response**: Validated answer with Intelligence Hub data returned

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | OpenAI GPT-4o-mini |
| Orchestration | LangGraph |
| Embeddings | OpenAI Embeddings |
| Vector Store | ChromaDB |
| Document Loader | PyMuPDF |
| API Framework | FastAPI |
| Geopolitical Data | NewsAPI, World Bank, GDELT |
| Evaluation | RAGAS |

## Key Features

✅ **No Pre-processing Required** — Upload and analyze in one request  
✅ **Real-time Risk Data** — Geopolitical risks from multiple external sources  
✅ **Automatic Cleanup** — Temporary files deleted after analysis  
✅ **Thread-Safe** — Each request uses isolated vector store  
✅ **Structured Output** — Sentiment, risk, and highlights in JSON format  
✅ **Retry Logic** — Graceful handling of rate-limited APIs  

## Tips for Best Results

1. **Use text-based PDFs** (not scanned images)
2. **Use standard ticker symbols** (e.g., AAPL, GOOGL)
3. **Annual reports work best** — structured financial documents yield better insights

## Limitations

- Large PDFs (>50MB) may take longer to process
- Each upload is processed independently (no session memory)
- OCR PDFs may have reduced accuracy

## License

MIT License

---

## What's Next?

- [ ] Add support for multiple file formats (DOCX, TXT)
- [ ] Implement persistent session storage for uploaded documents
- [ ] Add batch processing for multiple queries
- [ ] Build web frontend for easier interaction
- [ ] Add support for comparative analysis across multiple reports