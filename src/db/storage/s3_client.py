"""Object storage client (S3/GCS/local)."""

from pathlib import Path
import shutil

from src.config import settings


class StorageClient:
    """Abstraction over S3/GCS/local filesystem for document storage."""

    def __init__(self):
        self.backend = settings.storage_backend

    async def upload(self, local_path: Path, remote_key: str) -> str:
        """Upload a file to object storage. Returns the storage path."""
        if self.backend == "local":
            dest = Path(settings.crawler_data_dir) / remote_key
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, dest)
            return str(dest)

        # TODO: S3 upload
        # TODO: GCS upload
        raise NotImplementedError(f"Storage backend not implemented: {self.backend}")

    async def download(self, remote_key: str, local_path: Path) -> Path:
        """Download a file from object storage."""
        if self.backend == "local":
            src = Path(settings.crawler_data_dir) / remote_key
            shutil.copy2(src, local_path)
            return local_path

        raise NotImplementedError(f"Storage backend not implemented: {self.backend}")
