"""Export full-page documents from a corpus DB to numbered text files.

Usage:
    python scripts/export_pages.py --db data/hamilton_corpus.db
    # -> exports to data/hamilton_pages/
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.store import CorpusStore


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def title_from_uri(uri: str) -> str:
    return uri.rstrip("/").rsplit("/", 1)[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export full-page documents from a corpus DB.")
    parser.add_argument("--db", required=True, help="Path to the corpus SQLite database")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only export pages whose source URL contains this substring")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: <db_parent>/<stem>_pages/)")
    parser.add_argument("--per-city", action="store_true",
                        help="Place files in <out-dir>/<stem>/ subfolder and drop "
                             "the redundant <stem>_ prefix from filenames")
    args = parser.parse_args()

    db_path = Path(args.db)
    # data/hamilton_corpus.db -> data/hamilton_pages/ (default) or user-supplied --out-dir
    stem = db_path.stem.removesuffix("_corpus")
    base = Path(args.out_dir) if args.out_dir else db_path.parent / f"{stem}_pages"
    out_dir = base / stem if args.per_city else base
    out_dir.mkdir(parents=True, exist_ok=True)

    store = CorpusStore(db_path)
    docs = store.list_documents()
    if args.filter:
        docs = [d for d in docs if args.filter in d["source_uri"]]
    docs.sort(key=lambda d: d["source_uri"])

    for i, doc in enumerate(docs, 1):
        text = store.get_full_text(doc["doc_id"])
        if text is None:
            print(f"SKIP {doc['doc_id']}: no full_text")
            continue
        title = doc["title"] or title_from_uri(doc["source_uri"])
        slug = slugify(title)
        filename = (
            f"{i:02d}_{slug}.txt"
            if args.per_city
            else f"{stem}_{i:02d}_{slug}.txt"
        )
        (out_dir / filename).write_text(text, encoding="utf-8")
        print(f"{filename}  ({len(text):,} chars)")

    store.close()
    print(f"\nExported {len(docs)} files to {out_dir}")


if __name__ == "__main__":
    main()
