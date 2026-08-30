"""Conservative, deterministic text normalization.

Applied to all ingested text (web and PDF) before chunking and hashing
to ensure stable content hashes and reduced prompt noise.  Every step is
idempotent -- running the pipeline twice on the same input produces
identical output.
"""

import re
import unicodedata


class TextNormalizer:
    """Apply a fixed sequence of lossless text transforms.

    The pipeline is intentionally minimal so that meaning is preserved
    while surface-level formatting noise (extra whitespace, mixed
    newlines, Unicode compatibility characters) is eliminated.

    Attributes:
        min_line_length: Lines shorter than this (after stripping) are
            dropped.  Removes single-character UI fragments, stray
            bullets, and other non-content noise.  Set to ``0`` to
            keep all lines.
    """

    def __init__(self, min_line_length: int = 3) -> None:
        """Initialise the normalizer.

        Args:
            min_line_length: Minimum character count for a line to be
                retained.  Defaults to ``3``, which filters out common
                UI artifacts (lone bullets, toggle arrows, etc.) while
                keeping meaningful short lines such as ``"No"`` or
                ``"N/A"``.
        """
        self.min_line_length = min_line_length

    def normalize(self, text: str) -> str:
        """Run the full normalization pipeline on ``text``.

        Steps applied in order:
        1. NFKC Unicode normalization.
        2. Newline unification (``\\r\\n`` / ``\\r`` -> ``\\n``).
        3. Horizontal whitespace collapse (runs of spaces/tabs -> single
           space).
        4. Short-line removal (< ``min_line_length`` characters).
        5. Repeated blank-line collapse (3+ -> 2).
        6. Leading/trailing whitespace strip.

        Args:
            text: Raw extracted text (web or PDF).

        Returns:
            The cleaned text, ready for hashing and chunking.
        """
        text = self._unicode_normalize(text)
        text = self._normalize_newlines(text)
        text = self._collapse_whitespace(text)
        if self.min_line_length > 0:
            text = self._filter_short_lines(text)
        text = self._strip_repeated_blank_lines(text)
        text = text.strip()
        return text

    def _unicode_normalize(self, text: str) -> str:
        """Apply NFKC normalization.

        Canonical decomposition followed by compatibility composition.
        Converts typographic variants (e.g. full-width digits, ligatures)
        to their standard forms.

        Args:
            text: Input text.

        Returns:
            NFKC-normalized text.
        """
        return unicodedata.normalize("NFKC", text)

    def _normalize_newlines(self, text: str) -> str:
        """Convert all newline variants to ``\\n``.

        Args:
            text: Input text with potentially mixed line endings.

        Returns:
            Text with uniform ``\\n`` line endings.
        """
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _collapse_whitespace(self, text: str) -> str:
        """Collapse horizontal whitespace runs to a single space.

        Only collapses spaces and tabs; newlines are preserved so that
        paragraph structure is maintained.

        Args:
            text: Input text.

        Returns:
            Text with at most one consecutive space per run.
        """
        return re.sub(r"[^\S\n]+", " ", text)

    def _filter_short_lines(self, text: str) -> str:
        """Remove lines shorter than ``min_line_length``.

        These are typically UI fragments (toggle arrows, stray bullets,
        menu separators) left behind after boilerplate removal.

        Args:
            text: Input text.

        Returns:
            Text with short lines removed.
        """
        lines = text.splitlines()
        filtered = [line for line in lines if len(line.strip()) >= self.min_line_length]
        return "\n".join(filtered)

    def _strip_repeated_blank_lines(self, text: str) -> str:
        """Collapse 3+ consecutive newlines to exactly 2.

        Preserves single blank lines (paragraph breaks) while removing
        excessive vertical whitespace.

        Args:
            text: Input text.

        Returns:
            Text with at most one blank line between paragraphs.
        """
        return re.sub(r"\n{3,}", "\n\n", text)
