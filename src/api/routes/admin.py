from fastapi import APIRouter

router = APIRouter()


@router.get("/stats")
async def get_system_stats():
    """Get system-wide statistics (admin only)."""
    raise NotImplementedError


@router.get("/ingestion/status")
async def get_ingestion_status():
    """Get ingestion pipeline status and progress."""
    raise NotImplementedError


@router.post("/crawl/trigger")
async def trigger_crawl(source: str, params: dict | None = None):
    """Trigger a document crawl job."""
    raise NotImplementedError
