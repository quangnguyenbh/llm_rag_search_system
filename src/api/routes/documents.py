from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

router = APIRouter()


class DocumentMetadata(BaseModel):
    title: str | None = None
    manufacturer: str | None = None
    product_model: str | None = None
    document_type: str | None = None
    language: str = "en"
    tags: list[str] = []


@router.post("/upload")
async def upload_document(file: UploadFile = File(...), metadata: str = "{}"):
    """Upload a document for ingestion."""
    # TODO: Validate file, store in object storage, queue ingestion
    raise NotImplementedError


@router.get("")
async def list_documents(skip: int = 0, limit: int = 50):
    """List ingested documents with pagination."""
    raise NotImplementedError


@router.get("/{document_id}")
async def get_document(document_id: str):
    """Get document details and ingestion status."""
    raise NotImplementedError


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its indexed chunks."""
    raise NotImplementedError
