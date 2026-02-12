import os
import uuid
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import libsql_client
from pydantic import BaseModel
from dotenv import load_dotenv

from typing import List, Dict, Any

load_dotenv()
logger = logging.getLogger(__name__)

TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL", "")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN", "")

if not TURSO_DATABASE_URL or not TURSO_AUTH_TOKEN:
    raise RuntimeError(
        "TURSO_DATABASE_URL and TURSO_AUTH_TOKEN must be set in the environment"
    )

@dataclass
class User:
    id: str
    email: str
    password: str
    created_at: str

@dataclass
class Document:
    id: str
    user_id: str
    filename: str
    ticker: str
    s3_key: str
    created_at: str
    analysis_status: str  # 'pending', 'completed', 'failed'
    sentiment_score: float = 0.0
    sentiment_label: str = "neutral"  # 'bullish', 'bearish', 'neutral'
    ai_score: float = 0.0
    risk_level: str = "low"
    summary: str = ""

class TursoDB:
    """Thin wrapper around libsql_client for user and document CRUD operations."""

    def __init__(self):
        self._client = libsql_client.create_client_sync(
            url=TURSO_DATABASE_URL,
            auth_token=TURSO_AUTH_TOKEN,
        )
        self._ensure_tables()
        logger.info("Connected to Turso – database ready")

    def _ensure_tables(self):
        # Users table
        self._client.execute(
            "CREATE TABLE IF NOT EXISTS users ("
            "  id TEXT PRIMARY KEY,"
            "  email TEXT UNIQUE NOT NULL,"
            "  password TEXT NOT NULL,"
            "  created_at TEXT NOT NULL"
            ")"
        )
        
        # Documents table
        self._client.execute(
            "CREATE TABLE IF NOT EXISTS documents ("
            "  id TEXT PRIMARY KEY,"
            "  user_id TEXT NOT NULL,"
            "  filename TEXT,"
            "  ticker TEXT,"
            "  s3_key TEXT,"
            "  created_at TEXT NOT NULL,"
            "  analysis_status TEXT DEFAULT 'pending',"
            "  sentiment_score REAL DEFAULT 0.0,"
            "  sentiment_label TEXT DEFAULT 'neutral',"
            "  ai_score REAL DEFAULT 0.0,"
            "  risk_level TEXT DEFAULT 'low',"
            "  summary TEXT DEFAULT ''"
            ")"
        )

    @staticmethod
    def _row_to_user(row) -> User:
        """Convert a libsql result row to a User dataclass."""
        return User(
            id=row[0],
            email=row[1],
            password=row[2],
            created_at=row[3],
        )

    def create_user(self, email: str, hashed_password: str) -> User:
        """Insert a new user and return it."""
        user_id = uuid.uuid4().hex
        now = datetime.utcnow().isoformat()

        self._client.execute(
            "INSERT INTO users (id, email, password, created_at) VALUES (?, ?, ?, ?)",
            [user_id, email, hashed_password, now],
        )

        return User(id=user_id, email=email, password=hashed_password, created_at=now)

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Fetch a user by primary key."""
        rs = self._client.execute(
            "SELECT id, email, password, created_at FROM users WHERE id = ?",
            [user_id],
        )
        if rs.rows:
            return self._row_to_user(rs.rows[0])
        return None

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Fetch a user by email address."""
        rs = self._client.execute(
            "SELECT id, email, password, created_at FROM users WHERE email = ?",
            [email],
        )
        if rs.rows:
            return self._row_to_user(rs.rows[0])
        return None

    # --- Document Methods ---

    def create_document(self, document_id: str, user_id: str, filename: str, ticker: str, s3_key: str) -> Document:
        """Insert a new document record."""
        now = datetime.utcnow().isoformat()
        self._client.execute(
            "INSERT INTO documents (id, user_id, filename, ticker, s3_key, created_at, analysis_status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [document_id, user_id, filename, ticker, s3_key, now, "pending"]
        )
        return Document(
            id=document_id, user_id=user_id, filename=filename, ticker=ticker, s3_key=s3_key, 
            created_at=now, analysis_status="pending"
        )

    def get_document(self, document_id: str) -> Optional[Document]:
        """Fetch a document by ID."""
        rs = self._client.execute(
            "SELECT id, user_id, filename, ticker, s3_key, created_at, analysis_status, "
            "sentiment_score, sentiment_label, ai_score, risk_level, summary "
            "FROM documents WHERE id = ?",
            [document_id]
        )
        if rs.rows:
            row = rs.rows[0]
            return Document(
                id=row[0], user_id=row[1], filename=row[2], ticker=row[3], s3_key=row[4], created_at=row[5],
                analysis_status=row[6], sentiment_score=row[7], sentiment_label=row[8], ai_score=row[9],
                risk_level=row[10], summary=row[11]
            )
        return None

    def update_document_analysis(self, document_id: str, sentiment_score: float, sentiment_label: str, 
                                 ai_score: float, risk_level: str, summary: str):
        """Update document with analysis results."""
        self._client.execute(
            "UPDATE documents SET "
            "analysis_status = 'completed', "
            "sentiment_score = ?, "
            "sentiment_label = ?, "
            "ai_score = ?, "
            "risk_level = ?, "
            "summary = ? "
            "WHERE id = ?",
            [sentiment_score, sentiment_label, ai_score, risk_level, summary, document_id]
        )

    def delete_document(self, document_id: str):
        """Delete a document record."""
        self._client.execute(
            "DELETE FROM documents WHERE id = ?",
            [document_id]
        )

    def list_user_documents(self, user_id: str, limit: int = 50) -> List[Document]:
        """List documents for a user, most recent first."""
        rs = self._client.execute(
            "SELECT id, user_id, filename, ticker, s3_key, created_at, analysis_status, "
            "sentiment_score, sentiment_label, ai_score, risk_level, summary "
            "FROM documents WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            [user_id, limit]
        )
        docs = []
        for row in rs.rows:
            docs.append(Document(
                id=row[0], user_id=row[1], filename=row[2], ticker=row[3], s3_key=row[4], created_at=row[5],
                analysis_status=row[6], sentiment_score=row[7], sentiment_label=row[8], ai_score=row[9],
                risk_level=row[10], summary=row[11]
            ))
        return docs
    
    def get_dashboard_stats(self, user_id: str) -> Dict[str, Any]:
        """Aggregate statistics for the dashboard."""
        
        rs_total = self._client.execute("SELECT COUNT(*) FROM documents WHERE user_id = ?", [user_id])
        total_reports = rs_total.rows[0][0] if rs_total.rows else 0
        
        rs_last = self._client.execute(
            "SELECT created_at FROM documents WHERE user_id = ? AND analysis_status = 'completed' "
            "ORDER BY created_at DESC LIMIT 1", 
            [user_id]
        )
        last_analysis = rs_last.rows[0][0] if rs_last.rows else None
        
        # TODO: implement AI confidence calculation
        # In a real system, AI score might be confidence. We'll use the 'ai_score' column.
        rs_avg_ai = self._client.execute(
            "SELECT AVG(ai_score) FROM documents WHERE user_id = ? AND analysis_status = 'completed'", 
            [user_id]
        )
        avg_ai_score = rs_avg_ai.rows[0][0] if rs_avg_ai.rows and rs_avg_ai.rows[0][0] is not None else 0.0
        
        rs_sentiment = self._client.execute(
            "SELECT sentiment_label, COUNT(*) FROM documents "
            "WHERE user_id = ? AND analysis_status = 'completed' GROUP BY sentiment_label",
            [user_id]
        )
        sentiment_counts = {row[0]: row[1] for row in rs_sentiment.rows}
        
        return {
            "total_reports": total_reports,
            "last_analysis": last_analysis,
            "ai_accuracy_score": avg_ai_score,
            "sentiment_distribution": sentiment_counts
        }

turso_db = TursoDB()

class UserRegister(BaseModel):
    email: str
    password: str


class UserLogin(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    id: str
    email: str
    created_at: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse
