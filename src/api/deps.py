"""FastAPI dependencies: auth extraction, DB session, etc."""

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.auth.jwt import verify_token
from src.db.postgres.session import get_session

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/auth/login")


async def get_current_user_id(token: Annotated[str, Depends(oauth2_scheme)]) -> str:
    """Validate the bearer token and return the user_id (sub claim)."""
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = verify_token(token)
    except ValueError as exc:
        raise credentials_exc from exc

    user_id: str | None = payload.get("sub")
    if not user_id:
        raise credentials_exc
    return user_id


async def get_current_user_role(token: Annotated[str, Depends(oauth2_scheme)]) -> str:
    """Return the role claim from the bearer token."""
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = verify_token(token)
    except ValueError as exc:
        raise credentials_exc from exc
    return payload.get("role", "viewer")


def require_role(*allowed_roles: str):
    """Dependency factory: raises 403 if the caller's role is not in allowed_roles."""

    async def _check(role: Annotated[str, Depends(get_current_user_role)]) -> str:
        if role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return role

    return _check


# Shorthand aliases
CurrentUserID = Annotated[str, Depends(get_current_user_id)]
DBSession = Annotated[AsyncSession, Depends(get_session)]
