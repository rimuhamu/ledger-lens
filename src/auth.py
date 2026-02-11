import os
from datetime import datetime, timedelta
from typing import Optional

import bcrypt

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from dotenv import load_dotenv

from src.models import User, turso_db

load_dotenv()

SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
ALGORITHM = "HS256"
EXPIRY_HOURS: int = int(os.getenv("JWT_EXPIRY_HOURS", "24"))


def hash_password(plain: str) -> str:
    """Hash a plain-text password using bcrypt."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plain-text password against a bcrypt hash."""
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))

def create_access_token(
    user_id: str,
    email: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a signed JWT containing the user's id and email."""
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=EXPIRY_HOURS))
    payload = {"sub": user_id, "email": email, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Decode and validate a JWT.  Raises ``JWTError`` on failure."""
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

_bearer_scheme = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> User:
    """
    Extract and validate the JWT from ``Authorization: Bearer <token>``.

    Returns a ``User`` dataclass so handlers have full access to user data.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(credentials.credentials)
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = turso_db.get_user_by_id(user_id)
    if user is None:
        raise credentials_exception
    return user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    ),
) -> Optional[User]:
    """
    Same as ``get_current_user`` but returns ``None`` when no token is provided.
    Used for endpoints accessible both authenticated and anonymously.
    """
    if credentials is None:
        return None
    try:
        payload = decode_access_token(credentials.credentials)
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        return turso_db.get_user_by_id(user_id)
    except JWTError:
        return None