from fastapi import APIRouter
from pydantic import BaseModel, EmailStr

router = APIRouter()


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str
    organization: str | None = None


@router.post("/register")
async def register(request: RegisterRequest):
    """Register a new user account."""
    raise NotImplementedError


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """Authenticate and receive JWT tokens."""
    raise NotImplementedError


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token():
    """Refresh an expired access token."""
    raise NotImplementedError
