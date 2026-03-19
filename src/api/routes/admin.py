import structlog
from fastapi import APIRouter, Depends

from src.api.deps import DBSession, require_role
from src.config import settings
from src.db.postgres.repositories.document import DocumentRepository
from src.db.postgres.repositories.user import UserRepository
from src.db.vector.qdrant_client import get_qdrant_client

logger = structlog.get_logger()
router = APIRouter()

_require_admin = Depends(require_role("admin"))


@router.get("/stats", dependencies=[_require_admin])
async def get_system_stats(session: DBSession):
    """Get system-wide statistics (admin only)."""
    doc_repo = DocumentRepository(session)
    user_repo = UserRepository(session)

    total_docs = await doc_repo.count()
    total_users = await user_repo.count()

    # Try to fetch Qdrant collection info
    vector_count = 0
    try:
        qdrant = get_qdrant_client()
        info = qdrant.get_collection(settings.qdrant_collection)
        vector_count = info.points_count or 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("admin.qdrant_unavailable", error=str(exc))

    return {
        "documents": total_docs,
        "users": total_users,
        "vectors_indexed": vector_count,
    }


@router.get("/ingestion/status", dependencies=[_require_admin])
async def get_ingestion_status(session: DBSession):
    """Get ingestion pipeline status and progress."""
    doc_repo = DocumentRepository(session)

    pending = await _count_by_status(doc_repo, "pending")
    processing = await _count_by_status(doc_repo, "processing")
    indexed = await _count_by_status(doc_repo, "indexed")
    failed = await _count_by_status(doc_repo, "failed")

    return {
        "pending": pending,
        "processing": processing,
        "indexed": indexed,
        "failed": failed,
        "total": pending + processing + indexed + failed,
    }


async def _count_by_status(repo: DocumentRepository, status_val: str) -> int:
    from sqlalchemy import func, select

    from src.db.postgres.models import Document

    result = await repo.session.execute(
        select(func.count(Document.id)).where(Document.status == status_val)
    )
    return result.scalar_one()


@router.post("/crawl/trigger", dependencies=[_require_admin])
async def trigger_crawl(source: str, params: dict | None = None):
    """Trigger a document crawl job."""
    # TODO: Enqueue a Celery task for the given source
    return {"queued": True, "source": source, "params": params or {}}
