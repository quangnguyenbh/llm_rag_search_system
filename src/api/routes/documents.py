import json
import uuid
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from src.api.deps import CurrentUserID, DBSession
from src.config import settings
from src.db.postgres.repositories.document import DocumentRepository
from src.db.vector.qdrant_client import delete_by_document_id, get_qdrant_client
from src.shared.constants import SUPPORTED_EXTENSIONS

logger = structlog.get_logger()
router = APIRouter()


class DocumentMetadata(BaseModel):
    title: str | None = None
    manufacturer: str | None = None
    product_model: str | None = None
    document_type: str | None = None
    language: str = "en"
    tags: list[str] = []


class DocumentResponse(BaseModel):
    id: str
    title: str
    format: str
    status: str
    chunk_count: int
    page_count: int
    created_at: str


def _doc_to_response(doc) -> DocumentResponse:
    return DocumentResponse(
        id=doc.id,
        title=doc.title,
        format=doc.format,
        status=doc.status,
        chunk_count=doc.chunk_count,
        page_count=doc.page_count,
        created_at=doc.created_at.isoformat(),
    )


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    session: DBSession,
    _user_id: CurrentUserID,
    file: Annotated[UploadFile, File()],
    metadata: str = "{}",
):
    """Upload a document for ingestion."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{suffix}'. Allowed: {SUPPORTED_EXTENSIONS}",
        )

    try:
        meta = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid metadata JSON"
        ) from exc

    title = meta.get("title") or Path(file.filename or "untitled").stem

    # Save file locally (local storage backend)
    raw_dir = Path(settings.crawler_data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"{uuid.uuid4()}{suffix}"
    content = await file.read()
    dest.write_bytes(content)

    repo = DocumentRepository(session)
    doc = await repo.create(
        title=title,
        format=suffix.lstrip("."),
        storage_path=str(dest),
        metadata_json=meta,
        source_type="upload",
    )

    # TODO: Enqueue async ingestion via Celery task
    return {"document_id": doc.id, "title": doc.title, "status": doc.status}


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    session: DBSession,
    _user_id: CurrentUserID,
    skip: int = 0,
    limit: int = 50,
):
    """List ingested documents with pagination."""
    repo = DocumentRepository(session)
    docs = await repo.list_all(skip=skip, limit=limit)
    return [_doc_to_response(d) for d in docs]


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    session: DBSession,
    _user_id: CurrentUserID,
):
    """Get document details and ingestion status."""
    repo = DocumentRepository(session)
    doc = await repo.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return _doc_to_response(doc)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    session: DBSession,
    _user_id: CurrentUserID,
):
    """Delete a document and its indexed chunks."""
    repo = DocumentRepository(session)
    deleted = await repo.delete(document_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    # Remove vectors from Qdrant
    try:
        qdrant = get_qdrant_client()
        delete_by_document_id(qdrant, document_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("documents.qdrant_delete_failed", document_id=document_id, error=str(exc))
