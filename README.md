# LedgerLens

**LedgerLens** is an intelligent, multi-agent financial analysis platform. It leverages Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to analyze financial documents (PDFs) and extract actionable insights, risk assessments, and geopolitical context.

## 🚀 Features

- **Multi-Agent Workflow**: Orchestrates specialized agents (Researcher, Analyst, Validator, Intelligence Hub) for comprehensive analysis.
- **RAG Powered**: Uses Pinecone vector search to ground answers in document data.
- **Geopolitical Risk Analysis**: Integrates real-time geopolitical data (via NewsAPI) to assess macro risks relative to document entities.
- **Intelligence Hub**: Generates executive summaries, extracted metrics, and risk assessments in a structured JSON format.
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
    Copy `.env.example` to `.env` and fill in your configuration:
    ```bash
    cp .env.example .env
    ```
    Key variables include:
    - `OPENAI_API_KEY`: Required for LLM and embeddings.
    - `PINECONE_API_KEY`: Required if using Pinecone for vector storage.
    - `AWS_ACCESS_KEY_ID` & `AWS_SECRET_ACCESS_KEY`: Required for S3 object storage.
    - `TURSO_DATABASE_URL` & `TURSO_AUTH_TOKEN`: Required for database persistence.
    - `JWT_SECRET_KEY`: Required for authentication (generate with `openssl rand -hex 32`).
    - `NEWS_API_KEY`: Optional, required for geopolitical analysis feature.

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