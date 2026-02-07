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
- 🤖 **Multi-Agent Workflow** — Research → Analyst → Validator pipeline with LangGraph
- ✅ **Built-in Verification** — Automatic hallucination detection and answer validation
- 📈 **RAGAS Evaluation** — Comprehensive evaluation suite with industry-standard metrics
- 🚀 **FastAPI Backend** — RESTful API for seamless integration

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Researcher │ ──▶ │   Analyst    │ ──▶ │  Validator  │
│   (RAG)     │     │   (LLM)      │     │   (LLM)     │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                                    ┌───────────┴───────────┐
                                    │                       │
                                   PASS                   FAIL
                                    │                       │
                                    ▼                       │
                                  [END]                     │
                                                            ▼
                                                    [Re-research]
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

#### Health Check
```bash
GET /
```

#### Analyze Financial Data
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
│   └── vectorstore/      # ChromaDB vector database
├── scripts/
│   └── setup.py          # Data download and indexing
├── src/
│   ├── database.py       # Vector store operations
│   ├── graph.py          # LangGraph workflow definition
│   ├── main.py           # FastAPI application
│   ├── nodes.py          # Agent node implementations
│   └── eval.py           # RAGAS evaluation suite
├── .env                  # Environment variables
├── requirements.txt      # Python dependencies
└── README.md
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | OpenAI GPT-4o-mini |
| Orchestration | LangGraph |
| Embeddings | OpenAI Embeddings |
| Vector Store | ChromaDB |
| Document Loader | PyMuPDF |
| API Framework | FastAPI |
| Evaluation | RAGAS |

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.