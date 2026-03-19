"""Unit tests for DocumentRepository and UserRepository."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.db.postgres.repositories.document import DocumentRepository
from src.db.postgres.repositories.user import OrganizationRepository, UserRepository

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(
    id: str = "doc-1",
    title: str = "Test Manual",
    status: str = "pending",
    chunk_count: int = 0,
    page_count: int = 0,
    organization_id: str | None = None,
):
    doc = MagicMock()
    doc.id = id
    doc.title = title
    doc.format = "pdf"
    doc.storage_path = f"/tmp/{id}.pdf"
    doc.status = status
    doc.chunk_count = chunk_count
    doc.page_count = page_count
    doc.organization_id = organization_id
    doc.metadata_json = {}
    doc.source_url = None
    doc.source_type = "upload"
    return doc


def _make_session(scalar_result=None, scalars_result=None):
    """Build a minimal mock AsyncSession."""
    session = MagicMock()
    # execute() → result
    result = MagicMock()
    result.scalar_one_or_none.return_value = scalar_result
    result.scalar_one.return_value = 0
    result.scalars.return_value.all.return_value = scalars_result or []
    session.execute = AsyncMock(return_value=result)
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.delete = AsyncMock()
    return session


# ---------------------------------------------------------------------------
# DocumentRepository
# ---------------------------------------------------------------------------


class TestDocumentRepositoryGetById:
    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self):
        session = _make_session(scalar_result=None)
        repo = DocumentRepository(session)
        result = await repo.get_by_id("missing-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_document_when_found(self):
        doc = _make_doc("doc-42")
        session = _make_session(scalar_result=doc)
        repo = DocumentRepository(session)
        result = await repo.get_by_id("doc-42")
        assert result is doc


class TestDocumentRepositoryCreate:
    @pytest.mark.asyncio
    async def test_creates_document_and_returns_it(self):
        session = _make_session()
        repo = DocumentRepository(session)

        doc = await repo.create(
            title="My Manual",
            format="pdf",
            storage_path="/tmp/my-manual.pdf",
            source_type="upload",
        )

        session.add.assert_called_once()
        session.commit.assert_awaited_once()
        session.refresh.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_new_document_status_is_pending(self):
        session = _make_session()
        added_docs = []
        session.add.side_effect = lambda obj: added_docs.append(obj)

        repo = DocumentRepository(session)
        await repo.create(title="Manual", format="pdf", storage_path="/tmp/x.pdf")

        assert len(added_docs) == 1
        assert added_docs[0].status == "pending"


class TestDocumentRepositoryListAll:
    @pytest.mark.asyncio
    async def test_returns_all_documents(self):
        docs = [_make_doc("doc-1"), _make_doc("doc-2")]
        session = _make_session(scalars_result=docs)
        repo = DocumentRepository(session)

        result = await repo.list_all()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_empty_list(self):
        session = _make_session(scalars_result=[])
        repo = DocumentRepository(session)
        result = await repo.list_all()
        assert result == []


class TestDocumentRepositoryDelete:
    @pytest.mark.asyncio
    async def test_returns_false_when_not_found(self):
        session = _make_session(scalar_result=None)
        repo = DocumentRepository(session)
        deleted = await repo.delete("nonexistent")
        assert deleted is False
        session.delete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_true_and_deletes(self):
        doc = _make_doc("doc-99")
        session = _make_session(scalar_result=doc)
        repo = DocumentRepository(session)
        deleted = await repo.delete("doc-99")
        assert deleted is True
        session.delete.assert_awaited_once_with(doc)
        session.commit.assert_awaited_once()


# ---------------------------------------------------------------------------
# UserRepository
# ---------------------------------------------------------------------------


class TestUserRepository:
    @pytest.mark.asyncio
    async def test_get_by_email_not_found(self):
        session = _make_session(scalar_result=None)
        repo = UserRepository(session)
        result = await repo.get_by_email("nobody@example.com")
        assert result is None

    @pytest.mark.asyncio
    async def test_create_user(self):
        session = _make_session()
        added = []
        session.add.side_effect = lambda obj: added.append(obj)
        repo = UserRepository(session)

        await repo.create(
            email="alice@example.com",
            name="Alice",
            hashed_password="hashed-pw",
            role="admin",
        )

        assert len(added) == 1
        user = added[0]
        assert user.email == "alice@example.com"
        assert user.name == "Alice"
        assert user.hashed_password == "hashed-pw"
        assert user.role == "admin"

    @pytest.mark.asyncio
    async def test_create_user_generates_uuid(self):
        session = _make_session()
        added = []
        session.add.side_effect = lambda obj: added.append(obj)
        repo = UserRepository(session)

        await repo.create(email="b@b.com", name="B", hashed_password="h")
        assert len(added[0].id) == 36  # UUID4 format


# ---------------------------------------------------------------------------
# OrganizationRepository
# ---------------------------------------------------------------------------


class TestOrganizationRepository:
    @pytest.mark.asyncio
    async def test_create_organization(self):
        session = _make_session()
        added = []
        session.add.side_effect = lambda obj: added.append(obj)
        repo = OrganizationRepository(session)

        await repo.create(name="Acme Corp")

        assert len(added) == 1
        org = added[0]
        assert org.name == "Acme Corp"
        assert org.plan == "free"

    @pytest.mark.asyncio
    async def test_create_with_custom_plan(self):
        session = _make_session()
        added = []
        session.add.side_effect = lambda obj: added.append(obj)
        repo = OrganizationRepository(session)

        await repo.create(name="BigCorp", plan="enterprise")

        assert added[0].plan == "enterprise"
