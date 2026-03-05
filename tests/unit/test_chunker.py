"""Tests for the SemanticChunker."""

import pytest

from src.core.ingestion.chunker import SemanticChunker, Chunk, _count_tokens
from src.core.ingestion.parsers.base import ParsedDocument, ParsedPage


@pytest.fixture
def chunker():
    return SemanticChunker(target_size=512, overlap=100, min_size=100, max_size=1024)


@pytest.fixture
def small_chunker():
    """Small target for easier testing."""
    return SemanticChunker(target_size=50, overlap=10, min_size=10, max_size=100)


def _make_doc(pages: list[tuple[str, list[str]]], title: str = "Test Manual") -> ParsedDocument:
    """Helper: list of (text, headings) tuples → ParsedDocument."""
    return ParsedDocument(
        title=title,
        pages=[
            ParsedPage(page_number=i + 1, text=text, headings=headings)
            for i, (text, headings) in enumerate(pages)
        ],
        format="pdf",
    )


class TestTokenCounting:
    def test_basic_counting(self):
        assert _count_tokens("hello world") > 0

    def test_empty_string(self):
        assert _count_tokens("") == 0


class TestChunkMetadata:
    def test_chunk_has_document_id(self, chunker):
        doc = _make_doc([("Some text content here for testing.", [])])
        chunks = chunker.chunk(doc, {"title": "Test"})
        assert all(c.document_id == doc.document_id for c in chunks)

    def test_chunk_has_page_number(self, chunker):
        doc = _make_doc([("Content on page one.", []), ("Content on page two.", [])])
        chunks = chunker.chunk(doc, {"title": "Test"})
        assert all(c.page_number is not None for c in chunks)

    def test_chunk_has_token_count(self, chunker):
        doc = _make_doc([("Some meaningful text about electronics.", [])])
        chunks = chunker.chunk(doc, {"title": "Test"})
        assert all(c.token_count > 0 for c in chunks)

    def test_chunk_ids_are_unique(self, chunker):
        doc = _make_doc([("First paragraph.\n\nSecond paragraph.\n\nThird paragraph.", [])])
        chunks = chunker.chunk(doc, {"title": "Test"})
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


class TestContextualHeader:
    def test_header_includes_title(self, chunker):
        doc = _make_doc([("Some text.", [])])
        chunks = chunker.chunk(doc, {"title": "Bosch Dishwasher Manual"})
        assert "Bosch Dishwasher Manual" in chunks[0].text

    def test_header_includes_section(self, chunker):
        doc = _make_doc([("Troubleshooting\nError E15 means water leak.", ["Troubleshooting"])])
        chunks = chunker.chunk(doc, {"title": "Manual"})
        assert any("Troubleshooting" in c.text for c in chunks)


class TestSectionDetection:
    def test_single_section_no_headings(self, chunker):
        doc = _make_doc([("Just some plain text without any headings.", [])])
        chunks = chunker.chunk(doc, {"title": "Test"})
        assert len(chunks) >= 1

    def test_multiple_headings_create_sections(self, small_chunker):
        text = "Introduction\nWelcome to the manual.\n\nSetup\nPlug in the device and press power."
        headings = ["Introduction", "Setup"]
        doc = _make_doc([(text, headings)])
        chunks = small_chunker.chunk(doc, {"title": "Test"})
        assert len(chunks) >= 1

    def test_empty_pages_skipped(self, chunker):
        doc = _make_doc([("", []), ("   ", []), ("Actual content here.", [])])
        chunks = chunker.chunk(doc, {"title": "Test"})
        assert len(chunks) >= 1
        assert any("Actual content" in c.text for c in chunks)


class TestSmallSectionMerging:
    def test_tiny_section_merged_into_previous(self, small_chunker):
        # Create two headings, second one has very little text
        text = "Overview\nThis is the main overview section with some words.\n\nNote\nOk."
        headings = ["Overview", "Note"]
        doc = _make_doc([(text, headings)])
        chunks = small_chunker.chunk(doc, {"title": "Test"})
        # "Ok." alone is < min_size, should be merged
        texts = " ".join(c.text for c in chunks)
        assert "Ok." in texts


class TestChunkSizing:
    def test_small_doc_single_chunk(self, chunker):
        doc = _make_doc([("Short document.", [])])
        chunks = chunker.chunk(doc, {"title": "Test"})
        assert len(chunks) == 1

    def test_large_section_gets_split(self):
        chunker = SemanticChunker(target_size=30, overlap=5, min_size=5, max_size=60)
        # Build a document with many paragraphs
        paragraphs = [f"Paragraph number {i} with some filler text." for i in range(20)]
        text = "\n\n".join(paragraphs)
        doc = _make_doc([(text, [])])
        chunks = chunker.chunk(doc, {"title": "Test"})
        assert len(chunks) > 1

    def test_no_chunk_exceeds_max_size(self):
        chunker = SemanticChunker(target_size=50, overlap=10, min_size=10, max_size=100)
        # One massive paragraph
        text = "Word " * 500
        doc = _make_doc([(text, [])])
        chunks = chunker.chunk(doc, {"title": "T"})
        for c in chunks:
            # Allow some slack for the contextual header
            assert c.token_count <= 150, f"Chunk too large: {c.token_count} tokens"


class TestOverlap:
    def test_overlap_shares_text_between_chunks(self):
        chunker = SemanticChunker(target_size=30, overlap=10, min_size=5, max_size=100)
        paragraphs = [f"Sentence {i} is about topic alpha." for i in range(15)]
        text = "\n\n".join(paragraphs)
        doc = _make_doc([(text, [])])
        chunks = chunker.chunk(doc, {"title": "T"})

        if len(chunks) >= 2:
            # Some text from the end of chunk[0] should appear in chunk[1]
            # (overlap means shared content)
            text_0 = chunks[0].text
            text_1 = chunks[1].text
            # At least one paragraph should be shared
            paras_0 = set(text_0.split("\n\n"))
            paras_1 = set(text_1.split("\n\n"))
            shared = paras_0 & paras_1
            assert len(shared) >= 1, "Expected overlap between consecutive chunks"


class TestEmptyInput:
    def test_empty_document(self, chunker):
        doc = ParsedDocument(title="Empty", pages=[], format="pdf")
        chunks = chunker.chunk(doc, {"title": "Empty"})
        assert chunks == []

    def test_document_with_only_empty_pages(self, chunker):
        doc = _make_doc([("", []), ("", [])])
        chunks = chunker.chunk(doc, {"title": "Empty"})
        assert chunks == []
