"""Print all chunks from a single PDF for inspection."""

from pathlib import Path
from src.core.ingestion.parsers.pdf_parser import PdfParser
from src.core.ingestion.chunker import SemanticChunker

parser = PdfParser()
chunker = SemanticChunker()

pdf_path = Path("data/raw/huggingface/kaizen9_finepdfs_en/Cwlwm-Updated-Guidance-Final-30.03.2020.pdf")
doc = parser.parse(pdf_path)

print(f"Title: {doc.title}")
print(f"Pages: {len(doc.pages)}")
for i, page in enumerate(doc.pages):
    print(f"  Page {page.page_number}: {len(page.text)} chars, {len(page.headings)} headings")
    if page.headings:
        for h in page.headings[:10]:
            print(f"    heading: {h!r}")

chunks = chunker.chunk(doc, {"title": doc.title})
print(f"\nTotal chunks: {len(chunks)}\n")

for i, c in enumerate(chunks):
    print(f"{'=' * 70}")
    print(f"CHUNK {i+1}/{len(chunks)}")
    print(f"  chunk_id:    {c.chunk_id}")
    print(f"  page:        {c.page_number}")
    print(f"  section:     {c.section_path}")
    print(f"  hierarchy:   {c.heading_hierarchy}")
    print(f"  tokens:      {c.token_count}")
    print(f"--- TEXT START ---")
    print(c.text)
    print(f"--- TEXT END ---\n")
