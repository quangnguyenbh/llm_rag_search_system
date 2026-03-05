"""Base classes for document parsing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import uuid


@dataclass
class ParsedPage:
    page_number: int
    text: str
    headings: list[str] = field(default_factory=list)


@dataclass
class ParsedDocument:
    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    pages: list[ParsedPage] = field(default_factory=list)
    raw_text: str = ""
    format: str = ""

    @property
    def full_text(self) -> str:
        if self.pages:
            return "\n\n".join(p.text for p in self.pages)
        return self.raw_text


class DocumentParser(ABC):
    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse a document file and return structured content."""
        ...
