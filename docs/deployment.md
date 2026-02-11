# Deployment Guide

## Prerequisites

- Python 3.10+
- Pinecone Account (API Key & Index)
- AWS Account (S3 Bucket & Credentials)
- OpenAI API Key
- Turso Database (URL & Token)

## Environment Variables

Create a `.env` file in the root directory:

```ini
# API
JWT_SECRET_KEY=your_secret_key
JWT_EXPIRY_HOURS=24
API_TITLE="LedgerLens API"

# Database (Turso)
TURSO_DATABASE_URL=libsql://your-db.turso.io
TURSO_AUTH_TOKEN=your_token

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Pinecone
PINECONE_API_KEY=pc-...
PINECONE_INDEX_NAME=ledgerlens

# AWS S3
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=secret...
S3_BUCKET_NAME=ledgerlens-documents
AWS_REGION=us-east-1
```

## Local Development

1. **Install Dependencies:**
   ```bash
   pip install -e .
   ```

2. **Run the Application:**
   ```bash
   uvicorn src.main:app --reload
   ```

3. **run Tests:**
   ```bash
   pip install pytest
   pytest
   ```

## Docker Deployment

1. **Build Image:**
   ```bash
   docker build -t ledgerlens .
   ```

2. **Run Container:**
   ```bash
   docker run -d -p 8000:8000 --env-file .env ledgerlens
   ```
