"""Unit tests for AuthService: register, login, refresh."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.auth.service import AuthService, hash_password, verify_password


class TestPasswordHashing:
    def test_hash_is_not_plain(self):
        hashed = hash_password("secret123")
        assert hashed != "secret123"

    def test_verify_correct_password(self):
        hashed = hash_password("my-pass")
        assert verify_password("my-pass", hashed) is True

    def test_verify_wrong_password(self):
        hashed = hash_password("my-pass")
        assert verify_password("wrong", hashed) is False


class TestAuthServiceRegister:
    def _make_service(self, existing_user=None):
        """Build an AuthService with mocked repositories."""
        session = MagicMock()
        svc = AuthService(session)

        svc.user_repo = MagicMock()
        svc.user_repo.get_by_email = AsyncMock(return_value=existing_user)

        new_user = MagicMock()
        new_user.id = "user-123"
        new_user.email = "test@example.com"
        new_user.name = "Test User"
        new_user.role = "viewer"
        new_user.organization_id = None
        svc.user_repo.create = AsyncMock(return_value=new_user)

        svc.org_repo = MagicMock()
        new_org = MagicMock()
        new_org.id = "org-456"
        svc.org_repo.create = AsyncMock(return_value=new_org)

        return svc

    @pytest.mark.asyncio
    async def test_register_new_user(self):
        svc = self._make_service(existing_user=None)
        result = await svc.register(
            email="test@example.com",
            password="password123",
            name="Test User",
        )
        assert result["user_id"] == "user-123"
        assert result["email"] == "test@example.com"
        assert "access_token" in result
        assert "refresh_token" in result
        assert result["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_register_duplicate_email_raises(self):
        existing = MagicMock()
        svc = self._make_service(existing_user=existing)
        with pytest.raises(ValueError, match="already registered"):
            await svc.register(
                email="test@example.com",
                password="password123",
                name="Test User",
            )

    @pytest.mark.asyncio
    async def test_register_with_organization_creates_org(self):
        svc = self._make_service(existing_user=None)
        # Make user admin when org is provided
        new_user = MagicMock()
        new_user.id = "user-123"
        new_user.email = "admin@example.com"
        new_user.name = "Admin"
        new_user.role = "admin"
        new_user.organization_id = "org-456"
        svc.user_repo.create = AsyncMock(return_value=new_user)

        result = await svc.register(
            email="admin@example.com",
            password="password123",
            name="Admin",
            organization="Acme Corp",
        )
        svc.org_repo.create.assert_called_once_with(name="Acme Corp")
        assert result["user_id"] == "user-123"

    @pytest.mark.asyncio
    async def test_register_stores_hashed_password(self):
        svc = self._make_service(existing_user=None)
        await svc.register(email="test@example.com", password="my-secret", name="Test")
        call_kwargs = svc.user_repo.create.call_args.kwargs
        assert "hashed_password" in call_kwargs
        assert call_kwargs["hashed_password"] != "my-secret"


class TestAuthServiceLogin:
    def _make_service(self, user=None):
        session = MagicMock()
        svc = AuthService(session)
        svc.user_repo = MagicMock()
        svc.user_repo.get_by_email = AsyncMock(return_value=user)
        svc.org_repo = MagicMock()
        return svc

    @pytest.mark.asyncio
    async def test_login_valid_credentials(self):
        user = MagicMock()
        user.id = "user-123"
        user.email = "test@example.com"
        user.role = "viewer"
        user.hashed_password = hash_password("correct-password")
        svc = self._make_service(user=user)

        result = await svc.login("test@example.com", "correct-password")
        assert "access_token" in result
        assert "refresh_token" in result
        assert result["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_login_wrong_password_raises(self):
        user = MagicMock()
        user.id = "user-123"
        user.email = "test@example.com"
        user.role = "viewer"
        user.hashed_password = hash_password("correct-password")
        svc = self._make_service(user=user)

        with pytest.raises(ValueError, match="Invalid credentials"):
            await svc.login("test@example.com", "wrong-password")

    @pytest.mark.asyncio
    async def test_login_unknown_email_raises(self):
        svc = self._make_service(user=None)
        with pytest.raises(ValueError, match="Invalid credentials"):
            await svc.login("unknown@example.com", "password")


class TestAuthServiceRefresh:
    def _make_service(self, user=None):
        session = MagicMock()
        svc = AuthService(session)
        svc.user_repo = MagicMock()
        svc.user_repo.get_by_id = AsyncMock(return_value=user)
        svc.org_repo = MagicMock()
        return svc

    @pytest.mark.asyncio
    async def test_refresh_valid_token(self):
        from datetime import timedelta

        from src.core.auth.jwt import create_access_token

        user = MagicMock()
        user.id = "user-123"
        user.email = "test@example.com"
        user.role = "viewer"
        svc = self._make_service(user=user)

        refresh_token = create_access_token(
            data={"sub": "user-123", "type": "refresh"},
            expires_delta=timedelta(hours=1),
        )
        result = await svc.refresh(refresh_token)
        assert "access_token" in result
        assert result["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_refresh_access_token_raises(self):
        from src.core.auth.jwt import create_access_token

        svc = self._make_service(user=None)
        # Create an access token (not a refresh token)
        access_token = create_access_token({"sub": "user-123", "email": "x@x.com", "role": "viewer"})
        with pytest.raises(ValueError):
            await svc.refresh(access_token)

    @pytest.mark.asyncio
    async def test_refresh_invalid_token_raises(self):
        svc = self._make_service(user=None)
        with pytest.raises(ValueError):
            await svc.refresh("not.a.valid.token")
