# LedgerLens

**LedgerLens** is an intelligent, multi-agent financial analysis platform. It leverages Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to analyze financial documents (PDFs) and extract actionable insights, risk assessments, and geopolitical context.

## 🚀 Features

- **Multi-Agent Workflow**: Orchestrates specialized agents (Researcher, Analyst, Validator) for accurate analysis.
- **RAG Powered**: Uses Pinecone vector search to ground answers in document data.
- **Geopolitical Context**: Integrates external data to assess macro risks.
- **Modular Architecture**: Built with a clean, layered design for scalability and maintainability.

## 📚 Documentation

- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Deployment Guide](docs/deployment.md)

## 🛠️ Quick Start

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/rimuhamu/ledger-lens.git
    cd ledgerlens
    ```

2.  **Set up environment**:
    Copy `.env.example` to `.env` and fill in your API keys (OpenAI, Pinecone, AWS, Turso).

3.  **Install dependencies**:
    ```bash
    pip install -e .
    ```

4.  **Run the API**:
    ```bash
    uvicorn src.main:app --reload
    ```

5.  **Access Documentation**:
    Open `http://localhost:8000/docs` for the interactive Swagger UI.

## 🏗️ Project Structure

```
ledgerlens/
├── src/
│   ├── api/            # FastAPI routes and dependencies
│   ├── core/           # Agents, Services, and Workflows
│   ├── domain/         # Entities and Schemas
│   ├── infrastructure/ # Storage adapters (Pinecone, S3)
│   ├── utils/          # Logging and helpers
│   └── main.py         # Application entry point
├── tests/              # Unit and Integration tests
├── docs/               # Detailed documentation
└── config/             # Configuration files
```

## 🧪 Testing

Run the test suite to verify functionality:

```bash
pip install pytest
pytest
```