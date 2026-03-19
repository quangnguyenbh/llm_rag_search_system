"""Document repository — CRUD operations for the Document model."""

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.postgres.models import Document


class DocumentRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, document_id: str) -> Document | None:
        result = await self.session.execute(
            select(Document).where(Document.id == document_id)
        )
        return result.scalar_one_or_none()

    async def list_all(
        self,
        skip: int = 0,
        limit: int = 50,
        organization_id: str | None = None,
    ) -> list[Document]:
        stmt = select(Document)
        if organization_id:
            stmt = stmt.where(Document.organization_id == organization_id)
        stmt = stmt.order_by(Document.created_at.desc()).offset(skip).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def create(
        self,
        title: str,
        format: str,
        storage_path: str,
        organization_id: str | None = None,
        metadata_json: dict | None = None,
        source_url: str | None = None,
        source_type: str | None = None,
    ) -> Document:
        doc = Document(
            id=str(uuid.uuid4()),
            title=title,
            format=format,
            storage_path=storage_path,
            status="pending",
            organization_id=organization_id,
            metadata_json=metadata_json,
            source_url=source_url,
            source_type=source_type,
        )
        self.session.add(doc)
        await self.session.commit()
        await self.session.refresh(doc)
        return doc

    async def update_status(
        self,
        document_id: str,
        status: str,
        chunk_count: int = 0,
        page_count: int = 0,
    ) -> None:
        doc = await self.get_by_id(document_id)
        if doc:
            doc.status = status
            if chunk_count:
                doc.chunk_count = chunk_count
            if page_count:
                doc.page_count = page_count
            await self.session.commit()

    async def delete(self, document_id: str) -> bool:
        doc = await self.get_by_id(document_id)
        if not doc:
            return False
        await self.session.delete(doc)
        await self.session.commit()
        return True

    async def count(self, organization_id: str | None = None) -> int:
        stmt = select(func.count(Document.id))
        if organization_id:
            stmt = stmt.where(Document.organization_id == organization_id)
        result = await self.session.execute(stmt)
        return result.scalar_one()
