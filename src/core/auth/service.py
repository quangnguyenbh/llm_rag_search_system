"""Authentication service: user registration, login, token management."""


class AuthService:
    async def register(self, email: str, password: str, name: str) -> dict:
        """Register a new user."""
        raise NotImplementedError

    async def login(self, email: str, password: str) -> dict:
        """Authenticate user and return tokens."""
        raise NotImplementedError

    async def refresh(self, refresh_token: str) -> dict:
        """Refresh an expired access token."""
        raise NotImplementedError
