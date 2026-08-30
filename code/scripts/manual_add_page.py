"""One-off tool to manually insert a single page into a corpus DB.

Used when the automated scraper is blocked by a bot filter / 502 / WAF and
we paste the page text by hand.

Usage:
    python scripts/manual_add_page.py \
        --db data/richmond_corpus.db \
        --url https://www.example.com/page \
        --title "Page Title" \
        --institution "City of Richmond" \
        --short-name richmond \
        --category "Public Health & Emergency" \
        --label "Some Label" \
        --text-file /tmp/page.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.models import Document, DocumentMetadata
from src.ingest.chunker import SectionAwareChunker
from src.ingest.normalizer import TextNormalizer
from src.ingest.store import CorpusStore


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--institution", required=True)
    p.add_argument("--short-name", required=True)
    p.add_argument("--category", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--text-file", required=True)
    # Unified-shape extras — optional so legacy per-city manual inserts still work.
    p.add_argument("--cell", default="",
                   help="COFOG cell code (e.g. '03'); needed for the per-cell "
                        "Excel sheet split.")
    p.add_argument("--country", default="")
    p.add_argument("--layer", default="")
    args = p.parse_args()

    raw = Path(args.text_file).read_text(encoding="utf-8")
    normalizer = TextNormalizer()
    normalized = normalizer.normalize(raw)
    if not normalized:
        print("ERROR: normalizer returned empty text")
        sys.exit(1)

    doc_id = Document.make_doc_id(args.url)
    metadata = DocumentMetadata.web(source_uri=args.url, title=args.title)
    extra = {
        "institution": args.institution,
        "institution_short_name": args.short_name,
        "category": args.category,
        "label": args.label,
        "manual_insert": True,
    }
    if args.cell:
        extra["cell"] = args.cell
    if args.country:
        extra["country"] = args.country
    if args.layer:
        extra["layer"] = args.layer
    metadata.extra.update(extra)
    metadata.compute_content_hash(normalized)

    chunker = SectionAwareChunker(max_chunk_chars=2000, overlap_chars=200)
    chunks = chunker.chunk(normalized, doc_id)

    doc = Document(
        doc_id=doc_id,
        metadata=metadata,
        full_text=normalized,
        chunks=chunks,
    )

    store = CorpusStore(args.db)
    store.add_documents([doc])
    counts = store.count()
    store.close()
    print(f"Inserted doc_id={doc_id} url={args.url}")
    print(f"  text: {len(normalized):,} chars, {len(chunks)} chunks")
    print(f"  corpus now: {counts['documents']} documents, {counts['chunks']} chunks")


if __name__ == "__main__":
    main()
