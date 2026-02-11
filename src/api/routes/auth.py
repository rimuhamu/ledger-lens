from fastapi import APIRouter, HTTPException, Depends
from src.domain.schemas.auth import UserRegister, UserLogin, TokenResponse, UserResponse
from src.auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
)
from src.models import turso_db, User

router = APIRouter(prefix="/auth", tags=["Auth"])

@router.post("/register", response_model=TokenResponse)
async def register(body: UserRegister):
    """
    Register a new user account.
    """
    if not body.email or not body.password:
        raise HTTPException(status_code=400, detail="Email and password are required")

    if len(body.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    # Check if email already exists
    existing = turso_db.get_user_by_email(body.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    user = turso_db.create_user(body.email, hash_password(body.password))

    token = create_access_token(user.id, user.email)

    return TokenResponse(
        access_token=token,
        user=UserResponse(
            id=user.id,
            email=user.email,
            created_at=user.created_at,
        ),
    )

@router.post("/login", response_model=TokenResponse)
async def login(body: UserLogin):
    """
    Authenticate with email and password.
    """
    user = turso_db.get_user_by_email(body.email)
    if not user or not verify_password(body.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(user.id, user.email)

    return TokenResponse(
        access_token=token,
        user=UserResponse(
            id=user.id,
            email=user.email,
            created_at=user.created_at,
        ),
    )

@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get the currently authenticated user's profile."""
    # Convert legacy User model to Pydantic schema if needed, or if fields match it works automatically
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        created_at=current_user.created_at,
    )
