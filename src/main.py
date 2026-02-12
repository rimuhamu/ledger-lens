import os
from fastapi import FastAPI
from src.api.routes import auth, documents, analysis, dashboard

app = FastAPI(
    title="LedgerLens API",
    description="Multi-step reasoning agent for financial report analysis with RAG",
    version="2.0.0"
)

from fastapi.middleware.cors import CORSMiddleware
from src.config.settings import get_settings

settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(analysis.router)
app.include_router(dashboard.router)

@app.get("/")
def read_root():
    return {
        "status": "LedgerLens Analyst is Online",
        "version": "2.0.0",
        "storage": "Pinecone + S3",
        "model": "GPT-4o-mini / LangGraph"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "api": "connected"
    }