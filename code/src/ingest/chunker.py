"""Text chunker with stable hash-based chunk IDs.

Splits normalized text into overlapping chunks suitable for QA generation.
Chunk boundaries prefer paragraph breaks (``\\n\\n``), falling back to
sentence boundaries for oversized paragraphs.

Two chunker classes are provided:

- ``TextChunker`` — baseline paragraph-aligned chunking.
- ``SectionAwareChunker`` — detects section headings in the text and
  keeps each section together when possible, only splitting within a
  section when it exceeds the character budget.

Overlap between consecutive chunks ensures that facts spanning a
boundary are captured in at least one chunk.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from src.core.models import Chunk

# ── Heading patterns for section detection ────────────────────────────────────

HEADING_PATTERNS: list[re.Pattern[str]] = [
    # Numbered headings: "1.2 Methodology", "3.1.1 Results"
    re.compile(r"^(?:\d+\.)+\s+\S.+$"),
    # ALL-CAPS lines of 4+ chars: "EXECUTIVE SUMMARY"
    re.compile(r"^[A-Z][A-Z0-9 \-&:,/]{3,}$"),
    # Markdown-style headings: "## Section Title"
    re.compile(r"^#{1,4}\s+\S.+$"),
]


class _Section(NamedTuple):
    """A detected section: heading title + body text."""
    title: str
    body: str


# ── TextChunker (baseline) ────────────────────────────────────────────────────

class TextChunker:
    """Split text into overlapping, paragraph-aligned chunks.

    Attributes:
        max_chunk_chars: Soft upper bound on chunk size in characters.
            A chunk may slightly exceed this when a single paragraph
            is added to an empty accumulator.
        overlap_chars: Target number of trailing characters from the
            previous chunk to repeat at the start of the next chunk.
    """

    def __init__(
        self,
        max_chunk_chars: int = 2000,
        overlap_chars: int = 200,
    ) -> None:
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars

    def chunk(self, text: str, doc_id: str) -> list[Chunk]:
        """Split ``text`` into chunks, preferring paragraph boundaries.

        Args:
            text: Normalized document text.
            doc_id: Parent document identifier.

        Returns:
            Ordered list of ``Chunk`` objects with deterministic IDs.
        """
        if not text.strip():
            return []
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return self._chunk_paragraphs(paragraphs, doc_id, start_index=0)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _chunk_paragraphs(
        self,
        paragraphs: list[str],
        doc_id: str,
        start_index: int = 0,
        section_title: str = "",
    ) -> list[Chunk]:
        """Accumulate paragraphs into chunks respecting the char budget.

        This is the core paragraph-accumulation loop, extracted so that
        both ``TextChunker`` and ``SectionAwareChunker`` can reuse it.

        Args:
            paragraphs: Non-empty, stripped paragraph strings.
            doc_id: Parent document identifier.
            start_index: Starting chunk index (for deterministic IDs).
            section_title: Section heading to store in chunk metadata.

        Returns:
            List of chunks with deterministic IDs starting from
            ``start_index``.
        """
        chunks: list[Chunk] = []
        current_parts: list[str] = []
        current_len = 0
        index = start_index

        for para in paragraphs:
            if not para:
                continue

            para_len = len(para)

            # Oversized paragraph → flush + sentence-split
            if para_len > self.max_chunk_chars:
                if current_parts:
                    chunks.append(self._make_chunk(
                        "\n\n".join(current_parts), doc_id, index, section_title,
                    ))
                    index += 1
                    current_parts = []
                    current_len = 0

                for sub in self._split_long_paragraph(para):
                    chunks.append(self._make_chunk(sub, doc_id, index, section_title))
                    index += 1
                continue

            # Would adding this paragraph exceed the budget?
            new_len = current_len + para_len + (2 if current_parts else 0)
            if new_len > self.max_chunk_chars and current_parts:
                chunks.append(self._make_chunk(
                    "\n\n".join(current_parts), doc_id, index, section_title,
                ))
                index += 1

                # Carry over trailing paragraphs as overlap
                overlap_parts: list[str] = []
                overlap_len = 0
                for p in reversed(current_parts):
                    if overlap_len + len(p) + 2 > self.overlap_chars:
                        break
                    overlap_parts.insert(0, p)
                    overlap_len += len(p) + 2

                current_parts = overlap_parts
                current_len = (
                    sum(len(p) for p in current_parts)
                    + 2 * max(0, len(current_parts) - 1)
                )

            current_parts.append(para)
            current_len += para_len + (2 if len(current_parts) > 1 else 0)

        # Flush remaining
        if current_parts:
            chunks.append(self._make_chunk(
                "\n\n".join(current_parts), doc_id, index, section_title,
            ))

        return chunks

    def _split_long_paragraph(self, para: str) -> list[str]:
        """Split an oversized paragraph into sentence-based sub-chunks."""
        sentences = re.split(r'(?<=[.!?])\s+', para)
        parts: list[str] = []
        current = ""
        for sent in sentences:
            if current and len(current) + len(sent) + 1 > self.max_chunk_chars:
                parts.append(current.strip())
                current = sent
            else:
                current = f"{current} {sent}" if current else sent
        if current.strip():
            parts.append(current.strip())
        return parts

    def _make_chunk(
        self, text: str, doc_id: str, index: int, section_title: str = "",
    ) -> Chunk:
        """Create a ``Chunk`` with a deterministic content-based ID."""
        chunk_id = Chunk.make_chunk_id(doc_id, index, text)
        metadata: dict = {}
        if section_title:
            metadata["section_title"] = section_title
        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text,
            index=index,
            metadata=metadata,
        )


# ── SectionAwareChunker ──────────────────────────────────────────────────────

class SectionAwareChunker(TextChunker):
    """Detect section headings and chunk within sections.

    Phase 1 — Segment the document into named sections by scanning for
    heading patterns (numbered headings, ALL-CAPS lines, markdown ``#``).
    Phase 2 — Chunk each section independently using the parent's
    paragraph-accumulation logic.  Sections that fit within
    ``max_chunk_chars`` become a single chunk.

    This preserves topical coherence: the LLM sees complete sections
    rather than arbitrary text fragments that may cut mid-topic.
    """

    @staticmethod
    def _is_heading(line: str) -> bool:
        """Return ``True`` if the line looks like a section heading."""
        stripped = line.strip()
        if not stripped or len(stripped) > 120:
            return False
        return any(pat.match(stripped) for pat in HEADING_PATTERNS)

    def _segment_sections(self, text: str) -> list[_Section]:
        """Split text into sections based on detected headings.

        A line is treated as a heading when it matches a heading pattern
        and is preceded by a blank line (or is the first non-empty line).

        Returns:
            Ordered list of ``_Section(title, body)`` tuples.  The first
            section may have an empty title if the text starts without a
            heading.
        """
        lines = text.split("\n")
        sections: list[_Section] = []
        current_title = ""
        current_lines: list[str] = []
        prev_blank = True  # treat start-of-text as after a blank line

        for line in lines:
            stripped = line.strip()

            if not stripped:
                prev_blank = True
                current_lines.append("")
                continue

            # A heading must follow a blank line (or be at the start)
            if prev_blank and self._is_heading(stripped):
                # Flush the current section
                body = "\n".join(current_lines).strip()
                if body or current_title:
                    sections.append(_Section(title=current_title, body=body))
                current_title = stripped
                current_lines = []
                prev_blank = False
                continue

            prev_blank = False
            current_lines.append(line)

        # Flush final section
        body = "\n".join(current_lines).strip()
        if body or current_title:
            sections.append(_Section(title=current_title, body=body))

        # If nothing was detected as a heading, return the entire text as
        # a single untitled section.
        if not sections:
            sections.append(_Section(title="", body=text.strip()))

        return sections

    def chunk(self, text: str, doc_id: str) -> list[Chunk]:
        """Split text into section-aware chunks.

        1. Segment text into sections via heading detection.
        2. Chunk each section independently (sections that fit within
           ``max_chunk_chars`` become a single chunk).
        3. Carry overlap across section boundaries.
        4. Re-index chunks sequentially.

        Args:
            text: Normalized document text.
            doc_id: Parent document identifier.

        Returns:
            Ordered list of ``Chunk`` objects with deterministic IDs and
            ``section_title`` in metadata.
        """
        if not text.strip():
            return []

        sections = self._segment_sections(text)
        all_chunks: list[Chunk] = []
        index = 0
        carry_over: list[str] = []  # overlap paragraphs from previous section

        for section in sections:
            if not section.body:
                continue

            paragraphs = [p.strip() for p in section.body.split("\n\n") if p.strip()]
            if not paragraphs:
                continue

            # Prepend carry-over from previous section for overlap
            if carry_over:
                paragraphs = carry_over + paragraphs
                carry_over = []

            section_chunks = self._chunk_paragraphs(
                paragraphs, doc_id,
                start_index=index,
                section_title=section.title,
            )

            if section_chunks:
                # Compute carry-over for next section: trailing paragraphs
                # of the last chunk that fit within overlap_chars
                last_text = section_chunks[-1].text
                last_paras = [p.strip() for p in last_text.split("\n\n") if p.strip()]
                overlap_parts: list[str] = []
                overlap_len = 0
                for p in reversed(last_paras):
                    if overlap_len + len(p) + 2 > self.overlap_chars:
                        break
                    overlap_parts.insert(0, p)
                    overlap_len += len(p) + 2
                carry_over = overlap_parts

                all_chunks.extend(section_chunks)
                index += len(section_chunks)

        return all_chunks
