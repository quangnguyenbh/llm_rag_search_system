"""Semantic chunking with section awareness and heading hierarchy.

Strategy:
    PDF → Structure parsing (headings/sections) → Semantic grouping
    → 512-token chunks → 100-token overlap (intra-section only)

Key design decisions (see docs/adr/001-chunking-strategy.md):
    - Split first at heading/section boundaries (topic coherence)
    - Within sections, split at paragraph boundaries
    - Merge small consecutive sections (< min_size) into parent
    - Apply overlap only within the same section (not across topics)
    - Prepend contextual header: "Document: X | Section: Y"
    - Hard cap at max_size to prevent degenerate chunks
"""

import re
import uuid
from dataclasses import dataclass, field

import tiktoken

from src.core.ingestion.parsers.base import ParsedDocument, ParsedPage

# Use cl100k_base which is the tokenizer for text-embedding-3-large
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text, disallowed_special=()))


@dataclass
class Chunk:
    chunk_id: str = ""
    text: str = ""
    document_id: str = ""
    page_number: int | None = None
    section_path: str = ""
    heading_hierarchy: list[str] = field(default_factory=list)
    token_count: int = 0


@dataclass
class _Section:
    """Intermediate representation of a heading-delimited section."""
    heading: str
    page_number: int
    paragraphs: list[str] = field(default_factory=list)
    heading_hierarchy: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(self.paragraphs)

    @property
    def token_count(self) -> int:
        return _count_tokens(self.full_text)


class SemanticChunker:
    def __init__(
        self,
        target_size: int = 512,
        overlap: int = 100,
        min_size: int = 100,
        max_size: int = 1024,
    ):
        self.target_size = target_size
        self.overlap = overlap
        self.min_size = min_size
        self.max_size = max_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, document: ParsedDocument, metadata: dict) -> list[Chunk]:
        """Split a parsed document into semantic chunks.

        Flow:
            1. Build sections from pages (heading-aware)
            2. Merge tiny sections into neighbours
            3. Split oversized sections at paragraph boundaries
            4. Apply intra-section overlap
            5. Prepend contextual header
        """
        title = metadata.get("title", document.title) or "Untitled"

        # Step 1: Parse pages into heading-delimited sections
        sections = self._build_sections(document.pages)

        # Step 2: Merge sections that are too small
        sections = self._merge_small_sections(sections)

        # Step 3+4: Split oversized sections and produce final chunks
        chunks: list[Chunk] = []
        for section in sections:
            section_path = " > ".join(section.heading_hierarchy) if section.heading_hierarchy else section.heading
            section_chunks = self._split_section(section)

            for text in section_chunks:
                # Step 5: Prepend contextual header
                header = self._build_header(title, section_path)
                full_text = f"{header}\n\n{text}"

                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text=full_text,
                    document_id=document.document_id,
                    page_number=section.page_number,
                    section_path=section_path,
                    heading_hierarchy=list(section.heading_hierarchy),
                    token_count=_count_tokens(full_text),
                ))

        return chunks

    # ------------------------------------------------------------------
    # Step 1: Build sections from pages
    # ------------------------------------------------------------------

    def _build_sections(self, pages: list[ParsedPage]) -> list[_Section]:
        """Split page texts at heading boundaries into sections."""
        sections: list[_Section] = []
        current_hierarchy: list[str] = []

        for page in pages:
            if not page.text.strip():
                continue

            headings_on_page = set(page.headings)
            lines = page.text.split("\n")
            current_paragraphs: list[str] = []
            current_heading = current_hierarchy[-1] if current_hierarchy else ""

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue

                # Is this line a detected heading?
                if stripped in headings_on_page:
                    # Flush accumulated paragraphs as a section
                    if current_paragraphs:
                        sections.append(_Section(
                            heading=current_heading or "Introduction",
                            page_number=page.page_number,
                            paragraphs=self._group_into_paragraphs(current_paragraphs),
                            heading_hierarchy=list(current_hierarchy) or ["Introduction"],
                        ))
                        current_paragraphs = []

                    # Update hierarchy (simple: replace last level)
                    current_heading = stripped
                    if current_hierarchy:
                        current_hierarchy[-1] = stripped
                    else:
                        current_hierarchy.append(stripped)
                    continue

                current_paragraphs.append(stripped)

            # Flush remaining content on this page
            if current_paragraphs:
                sections.append(_Section(
                    heading=current_heading or "Introduction",
                    page_number=page.page_number,
                    paragraphs=self._group_into_paragraphs(current_paragraphs),
                    heading_hierarchy=list(current_hierarchy) or ["Introduction"],
                ))

        # If no sections were created (no headings at all), make one big section
        if not sections and pages:
            all_text = "\n".join(p.text for p in pages if p.text.strip())
            sections.append(_Section(
                heading="Content",
                page_number=pages[0].page_number,
                paragraphs=self._group_into_paragraphs(all_text.split("\n")),
                heading_hierarchy=["Content"],
            ))

        return sections

    @staticmethod
    def _group_into_paragraphs(lines: list[str]) -> list[str]:
        """Group consecutive non-empty lines into paragraph strings."""
        paragraphs: list[str] = []
        current: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current:
                    paragraphs.append(" ".join(current))
                    current = []
            else:
                current.append(stripped)

        if current:
            paragraphs.append(" ".join(current))

        return paragraphs

    # ------------------------------------------------------------------
    # Step 2: Merge small sections
    # ------------------------------------------------------------------

    def _merge_small_sections(self, sections: list[_Section]) -> list[_Section]:
        """Merge sections smaller than min_size into the previous section."""
        if not sections:
            return sections

        merged: list[_Section] = [sections[0]]

        for section in sections[1:]:
            if section.token_count < self.min_size and merged:
                # Absorb into previous section
                prev = merged[-1]
                prev.paragraphs.extend(section.paragraphs)
            else:
                merged.append(section)

        return merged

    # ------------------------------------------------------------------
    # Step 3+4: Split section into target-sized chunks with overlap
    # ------------------------------------------------------------------

    def _split_section(self, section: _Section) -> list[str]:
        """Split a section's paragraphs into chunks of ~target_size tokens.

        - If the section fits in target_size, return it as-is.
        - Otherwise, accumulate paragraphs up to target_size, then start a
          new chunk with overlap from the previous chunk.
        - Paragraphs that exceed max_size are force-split by sentence.
        """
        paragraphs = section.paragraphs
        if not paragraphs:
            return []

        # Fast path: entire section fits in target
        if section.token_count <= self.target_size:
            return [section.full_text]

        # First, force-split any individual paragraphs that exceed max_size
        expanded: list[str] = []
        for para in paragraphs:
            if _count_tokens(para) > self.max_size:
                expanded.extend(self._force_split_paragraph(para))
            else:
                expanded.append(para)

        chunks: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for para in expanded:
            para_tokens = _count_tokens(para)

            if current_tokens + para_tokens > self.target_size and current_parts:
                # Emit current chunk
                chunks.append("\n\n".join(current_parts))

                # Build overlap from the tail of current_parts
                overlap_parts = self._extract_overlap(current_parts)
                current_parts = overlap_parts + [para]
                current_tokens = sum(_count_tokens(p) for p in current_parts)
            else:
                current_parts.append(para)
                current_tokens += para_tokens

        # Emit final chunk
        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return chunks

    def _extract_overlap(self, parts: list[str]) -> list[str]:
        """Take paragraphs from the end of parts totalling ~overlap tokens."""
        overlap_parts: list[str] = []
        tokens = 0
        for part in reversed(parts):
            part_tokens = _count_tokens(part)
            if tokens + part_tokens > self.overlap:
                break
            overlap_parts.insert(0, part)
            tokens += part_tokens
        return overlap_parts

    def _force_split_paragraph(self, text: str) -> list[str]:
        """Force-split a very long paragraph by sentence boundaries.

        Falls back to word-level splitting when no sentence boundaries exist.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # If regex didn't actually split (no punctuation), split by words
        if len(sentences) == 1:
            return self._split_by_words(text)

        parts: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            st = _count_tokens(sentence)
            if current_tokens + st > self.target_size and current:
                parts.append(" ".join(current))
                current = [sentence]
                current_tokens = st
            else:
                current.append(sentence)
                current_tokens += st

        if current:
            parts.append(" ".join(current))

        return parts

    def _split_by_words(self, text: str) -> list[str]:
        """Last-resort split: cut at word boundaries to fit target_size."""
        words = text.split()
        parts: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for word in words:
            wt = _count_tokens(word)
            if current_tokens + wt > self.target_size and current:
                parts.append(" ".join(current))
                current = [word]
                current_tokens = wt
            else:
                current.append(word)
                current_tokens += wt

        if current:
            parts.append(" ".join(current))

        return parts

    # ------------------------------------------------------------------
    # Step 5: Contextual header
    # ------------------------------------------------------------------

    @staticmethod
    def _build_header(title: str, section_path: str) -> str:
        """Build a contextual header prepended to each chunk."""
        if section_path and section_path not in ("Content", "Introduction"):
            return f"Document: {title} | Section: {section_path}"
        return f"Document: {title}"
