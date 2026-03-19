"""Authentication service: user registration, login, token management."""

from datetime import timedelta

from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.core.auth.jwt import create_access_token, verify_token
from src.db.postgres.repositories.user import OrganizationRepository, UserRepository

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Refresh tokens are long-lived (7 days)
REFRESH_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


class AuthService:
    def __init__(self, session: AsyncSession):
        self.user_repo = UserRepository(session)
        self.org_repo = OrganizationRepository(session)

    async def register(
        self,
        email: str,
        password: str,
        name: str,
        organization: str | None = None,
    ) -> dict:
        """Register a new user, optionally creating an organization."""
        existing = await self.user_repo.get_by_email(email)
        if existing:
            raise ValueError("Email already registered")

        org_id: str | None = None
        if organization:
            org = await self.org_repo.create(name=organization)
            org_id = org.id

        hashed = hash_password(password)
        user = await self.user_repo.create(
            email=email,
            name=name,
            hashed_password=hashed,
            role="admin" if organization else "viewer",
            organization_id=org_id,
        )

        access_token = create_access_token(
            data={"sub": user.id, "email": user.email, "role": user.role}
        )
        refresh_token = create_access_token(
            data={"sub": user.id, "type": "refresh"},
            expires_delta=timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES),
        )
        return {
            "user_id": user.id,
            "email": user.email,
            "name": user.name,
            "role": user.role,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": settings.jwt_access_token_expire_minutes * 60,
        }

    async def login(self, email: str, password: str) -> dict:
        """Authenticate user and return tokens."""
        user = await self.user_repo.get_by_email(email)
        if not user or not verify_password(password, user.hashed_password):
            raise ValueError("Invalid credentials")

        access_token = create_access_token(
            data={"sub": user.id, "email": user.email, "role": user.role}
        )
        refresh_token = create_access_token(
            data={"sub": user.id, "type": "refresh"},
            expires_delta=timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES),
        )
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": settings.jwt_access_token_expire_minutes * 60,
        }

    async def refresh(self, refresh_token: str) -> dict:
        """Issue a new access token from a valid refresh token."""
        try:
            payload = verify_token(refresh_token)
        except ValueError as exc:
            raise ValueError("Invalid or expired refresh token") from exc

        if payload.get("type") != "refresh":
            raise ValueError("Not a refresh token")

        user_id: str | None = payload.get("sub")
        if not user_id:
            raise ValueError("Invalid refresh token payload")

        user = await self.user_repo.get_by_id(user_id)
        if not user:
            raise ValueError("User not found")

        access_token = create_access_token(
            data={"sub": user.id, "email": user.email, "role": user.role}
        )
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": settings.jwt_access_token_expire_minutes * 60,
        }

    async def get_current_user_id(self, token: str) -> str:
        """Validate an access token and return the user ID."""
        payload = verify_token(token)
        user_id: str | None = payload.get("sub")
        if not user_id:
            raise ValueError("Invalid token payload")
        return user_id
