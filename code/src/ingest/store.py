"""SQLite-backed corpus store for the immutable chunk corpus.

Provides persistent storage for ``Document`` and ``Chunk`` objects with
full metadata.  Supports querying by source type, document ID, and
individual chunk ID.  Can also export chunks as self-documenting
``.txt`` files for human inspection.

The store uses upsert semantics: re-ingesting a document with the same
``doc_id`` replaces the previous version and its chunks.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from src.core.models import Chunk, Document, DocumentMetadata

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id        TEXT PRIMARY KEY,
    source_type   TEXT NOT NULL,
    source_uri    TEXT NOT NULL,
    title         TEXT,
    content_hash  TEXT,
    fetch_timestamp TEXT,
    full_text     TEXT,
    extra_json    TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   TEXT PRIMARY KEY,
    doc_id     TEXT NOT NULL,
    idx        INTEGER NOT NULL,
    text       TEXT NOT NULL,
    meta_json  TEXT,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_docs_source_type ON documents(source_type);
"""


class CorpusStore:
    """Persistent, queryable corpus store backed by a single SQLite file.

    Attributes:
        db_path: Absolute path to the SQLite database file.
    """

    def __init__(self, db_path: str | Path) -> None:
        """Open (or create) the corpus database.

        The parent directory is created automatically if it does not
        exist.  The schema is applied idempotently on every open.

        Args:
            db_path: File path for the SQLite database.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    # ── write ────────────────────────────────────────────────────────────

    def add_document(self, doc: Document) -> None:
        """Insert or replace a document and all of its chunks.

        If a document with the same ``doc_id`` already exists, it and
        its chunks are fully replaced (upsert semantics).

        Args:
            doc: The ``Document`` to store.
        """
        meta = doc.metadata
        extra = meta.extra

        self._conn.execute(
            """
            INSERT OR REPLACE INTO documents
                (doc_id, source_type, source_uri, title, content_hash,
                 fetch_timestamp, full_text, extra_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc.doc_id,
                meta.source_type,
                meta.source_uri,
                meta.title,
                meta.content_hash,
                meta.fetch_timestamp,
                doc.full_text,
                json.dumps(extra, ensure_ascii=False),
            ),
        )

        # Remove old chunks for this doc then insert new
        self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc.doc_id,))
        for chunk in doc.chunks:
            self._conn.execute(
                """
                INSERT INTO chunks (chunk_id, doc_id, idx, text, meta_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.index,
                    chunk.text,
                    json.dumps(chunk.metadata, ensure_ascii=False),
                ),
            )

        self._conn.commit()

    def add_documents(self, docs: list[Document]) -> None:
        """Insert or replace multiple documents.

        Args:
            docs: List of ``Document`` objects to store.
        """
        for doc in docs:
            self.add_document(doc)
        logger.info("Stored %d documents in corpus", len(docs))

    # ── read ─────────────────────────────────────────────────────────────

    def get_all_chunks(self) -> list[Chunk]:
        """Return every chunk in the corpus, ordered by document then index.

        Returns:
            List of all ``Chunk`` objects.
        """
        rows = self._conn.execute(
            "SELECT chunk_id, doc_id, idx, text, meta_json FROM chunks ORDER BY doc_id, idx"
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def get_chunks_by_doc(self, doc_id: str) -> list[Chunk]:
        """Return chunks belonging to a specific document.

        Args:
            doc_id: The 16-char hex document identifier.

        Returns:
            Ordered list of ``Chunk`` objects for that document.
        """
        rows = self._conn.execute(
            "SELECT chunk_id, doc_id, idx, text, meta_json FROM chunks WHERE doc_id = ? ORDER BY idx",
            (doc_id,),
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def get_chunks_by_source_type(self, source_type: str) -> list[Chunk]:
        """Return all chunks from documents of a given source type.

        Args:
            source_type: Either ``"web"`` or ``"pdf"``.

        Returns:
            List of ``Chunk`` objects from matching documents.
        """
        rows = self._conn.execute(
            """
            SELECT c.chunk_id, c.doc_id, c.idx, c.text, c.meta_json
            FROM chunks c JOIN documents d ON c.doc_id = d.doc_id
            WHERE d.source_type = ?
            ORDER BY c.doc_id, c.idx
            """,
            (source_type,),
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def get_document_metadata(self, doc_id: str) -> dict | None:
        """Return raw document metadata as a dictionary.

        Args:
            doc_id: The 16-char hex document identifier.

        Returns:
            A dict with all document columns, or ``None`` if not found.
        """
        row = self._conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if not row:
            return None
        return dict(row)

    def list_documents(self) -> list[dict]:
        """List all documents with basic metadata (no full text).

        Returns:
            List of dicts with keys ``doc_id``, ``source_type``,
            ``source_uri``, ``title``, ``content_hash``,
            ``fetch_timestamp``, and ``extra_json``.
        """
        rows = self._conn.execute(
            "SELECT doc_id, source_type, source_uri, title, content_hash, fetch_timestamp, extra_json FROM documents"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_chunk_by_id(self, chunk_id: str) -> Chunk | None:
        """Retrieve a single chunk by its ID.

        Args:
            chunk_id: The 16-char hex chunk identifier.

        Returns:
            The matching ``Chunk``, or ``None`` if not found.
        """
        row = self._conn.execute(
            "SELECT chunk_id, doc_id, idx, text, meta_json FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        return self._row_to_chunk(row) if row else None

    def get_full_text(self, doc_id: str) -> str | None:
        """Return the full normalized text of a document.

        Args:
            doc_id: The 16-char hex document identifier.

        Returns:
            The complete text, or ``None`` if the document is not found.
        """
        row = self._conn.execute(
            "SELECT full_text FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return row["full_text"] if row else None

    def count(self) -> dict[str, int]:
        """Return aggregate counts of documents and chunks.

        Returns:
            Dict with keys ``"documents"`` and ``"chunks"``.
        """
        doc_count = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"documents": doc_count, "chunks": chunk_count}

    # ── export ───────────────────────────────────────────────────────────

    def export_chunks_as_txt(self, output_dir: str | Path) -> None:
        """Export every chunk as a self-documenting ``.txt`` file.

        Each file contains a metadata header (chunk ID, document ID,
        source info, page range, section title) followed by the chunk
        text, separated by a ``----`` delimiter.

        Args:
            output_dir: Directory to write chunk files into.  Created
                if it does not exist.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_chunks = self.get_all_chunks()
        for chunk in all_chunks:
            doc_meta = self.get_document_metadata(chunk.doc_id)
            header_lines = [
                f"CHUNK_ID: {chunk.chunk_id}",
                f"DOC_ID: {chunk.doc_id}",
                f"SOURCE_TYPE: {doc_meta['source_type']}" if doc_meta else "",
                f"SOURCE_URI: {doc_meta['source_uri']}" if doc_meta else "",
                f"TITLE: {doc_meta['title']}" if doc_meta else "",
            ]
            if chunk.metadata.get("page_start"):
                header_lines.append(f"PAGE_START: {chunk.metadata['page_start']}")
                header_lines.append(f"PAGE_END: {chunk.metadata['page_end']}")
            if chunk.metadata.get("section_title"):
                header_lines.append(f"SECTION: {chunk.metadata['section_title']}")

            header = "\n".join(line for line in header_lines if line) + "\n----\n"

            file_path = output_dir / f"{chunk.chunk_id}.txt"
            file_path.write_text(header + chunk.text, encoding="utf-8")

        logger.info("Exported %d chunk files to %s", len(all_chunks), output_dir)

    # ── internal ─────────────────────────────────────────────────────────

    def _row_to_chunk(self, row: sqlite3.Row) -> Chunk:
        """Convert a SQLite row into a ``Chunk`` dataclass.

        Args:
            row: A ``sqlite3.Row`` from the ``chunks`` table.

        Returns:
            A ``Chunk`` instance with deserialized metadata.
        """
        return Chunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            text=row["text"],
            index=row["idx"],
            metadata=json.loads(row["meta_json"]) if row["meta_json"] else {},
        )

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
