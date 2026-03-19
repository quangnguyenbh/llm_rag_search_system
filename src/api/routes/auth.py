from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr

from src.api.deps import DBSession
from src.core.auth.service import AuthService

router = APIRouter()


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"  # noqa: S105
    expires_in: int


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str
    organization: str | None = None


class RegisterResponse(BaseModel):
    user_id: str
    email: str
    name: str
    role: str
    access_token: str
    refresh_token: str
    token_type: str = "bearer"  # noqa: S105
    expires_in: int


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest, session: DBSession):
    """Register a new user account."""
    svc = AuthService(session)
    try:
        result = await svc.register(
            email=request.email,
            password=request.password,
            name=request.name,
            organization=request.organization,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return RegisterResponse(**result)


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, session: DBSession):
    """Authenticate and receive JWT tokens."""
    svc = AuthService(session)
    try:
        result = await svc.login(email=request.email, password=request.password)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    return TokenResponse(**result)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshRequest, session: DBSession):
    """Refresh an expired access token."""
    svc = AuthService(session)
    try:
        result = await svc.refresh(request.refresh_token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    return TokenResponse(**result)
