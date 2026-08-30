"""Data models for the ingestion pipeline.

Defines the core dataclasses used across ingestion, chunking, and storage:
``DocumentMetadata``, ``Chunk``, and ``Document``.  Every identifier is
deterministic (SHA-256-based) so that rebuilding the corpus from the same
sources produces identical IDs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class DocumentMetadata:
    """Metadata envelope shared by every ingested document.

    Source-type-specific fields (e.g. ``status_code`` for web,
    ``total_pages`` for PDF) are stored in the ``extra`` dict rather than
    as subclass attributes so that a single flat dataclass can be
    serialized uniformly.

    Attributes:
        source_type: Origin kind, either ``"web"`` or ``"pdf"``.
        source_uri: Canonical locator -- final URL after redirects, or
            absolute file path for PDFs.
        title: Human-readable document title, if discoverable.
        fetch_timestamp: ISO-8601 UTC timestamp of when the content was
            retrieved.
        content_hash: SHA-256 hex digest of the **normalized** full text.
        extra: Arbitrary key-value pairs for source-type-specific metadata
            (e.g. ``status_code``, ``total_pages``, ``page_sections``).
    """

    source_type: str
    source_uri: str
    title: str = ""
    fetch_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    content_hash: str = ""
    extra: dict = field(default_factory=dict)

    def compute_content_hash(self, text: str) -> None:
        """Compute and store the SHA-256 hash of ``text``.

        Args:
            text: The normalized full text of the document.
        """
        self.content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def web(
        source_uri: str,
        title: str = "",
        status_code: int = 0,
        content_type: str = "",
    ) -> DocumentMetadata:
        """Factory for web-document metadata.

        Args:
            source_uri: Final URL after any redirects.
            title: Page ``<title>`` text, if extracted.
            status_code: HTTP status code of the response.
            content_type: Value of the ``Content-Type`` response header.

        Returns:
            A ``DocumentMetadata`` instance with ``source_type="web"``.
        """
        return DocumentMetadata(
            source_type="web",
            source_uri=source_uri,
            title=title,
            extra={"status_code": status_code, "content_type": content_type},
        )

    @staticmethod
    def pdf(
        source_uri: str,
        title: str = "",
        total_pages: int = 0,
        **extra_fields,
    ) -> DocumentMetadata:
        """Factory for PDF-document metadata.

        Args:
            source_uri: Absolute file-system path to the PDF.
            title: Document title (defaults to the file stem).
            total_pages: Number of pages in the PDF.
            **extra_fields: Additional metadata such as ``subfolder``,
                ``low_text_pages``, or ``page_sections``.

        Returns:
            A ``DocumentMetadata`` instance with ``source_type="pdf"``.
        """
        extra = {"total_pages": total_pages, **extra_fields}
        return DocumentMetadata(
            source_type="pdf",
            source_uri=source_uri,
            title=title,
            extra=extra,
        )


@dataclass
class Chunk:
    """A contiguous text fragment with a stable, content-derived ID.

    Chunks are the atomic units fed to downstream QA generation agents.
    Their IDs are deterministic so that citations remain valid even when
    the corpus is rebuilt from the same sources.

    Attributes:
        chunk_id: 16-char hex string derived from
            ``SHA-256(doc_id::index::text)``.
        doc_id: Identifier of the parent ``Document``.
        text: The chunk's text content.
        index: Zero-based position within the parent document.
        metadata: Optional per-chunk annotations (e.g. ``page_start``,
            ``page_end``, ``section_title`` for PDF chunks).
    """

    chunk_id: str
    doc_id: str
    text: str
    index: int
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def make_chunk_id(doc_id: str, index: int, text: str) -> str:
        """Generate a stable, deterministic chunk ID.

        Args:
            doc_id: Parent document identifier.
            index: Chunk position within the document.
            text: Chunk text content.

        Returns:
            A 16-character hex string uniquely identifying this chunk.
        """
        payload = f"{doc_id}::{index}::{text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class Document:
    """A fully ingested document: metadata, full text, and ordered chunks.

    Attributes:
        doc_id: 16-char hex string derived from ``SHA-256(source_uri)``.
        metadata: Provenance and descriptive metadata.
        full_text: The complete normalized text of the document.
        chunks: Ordered list of ``Chunk`` objects produced by the chunker.
    """

    doc_id: str
    metadata: DocumentMetadata
    full_text: str
    chunks: list[Chunk] = field(default_factory=list)

    @staticmethod
    def make_doc_id(source_uri: str) -> str:
        """Generate a deterministic document ID from its source URI.

        Args:
            source_uri: The canonical locator (URL or file path).

        Returns:
            A 16-character hex string uniquely identifying this document.
        """
        return hashlib.sha256(source_uri.encode("utf-8")).hexdigest()[:16]
