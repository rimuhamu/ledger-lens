import os
import uuid
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import libsql_client
from pydantic import BaseModel
from dotenv import load_dotenv

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


class TursoDB:
    """Thin wrapper around libsql_client for user CRUD operations."""

    def __init__(self):
        self._client = libsql_client.create_client_sync(
            url=TURSO_DATABASE_URL,
            auth_token=TURSO_AUTH_TOKEN,
        )
        self._ensure_tables()
        logger.info("Connected to Turso – user database ready")

    def _ensure_tables(self):
        self._client.execute(
            "CREATE TABLE IF NOT EXISTS users ("
            "  id TEXT PRIMARY KEY,"
            "  email TEXT UNIQUE NOT NULL,"
            "  password TEXT NOT NULL,"
            "  created_at TEXT NOT NULL"
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
