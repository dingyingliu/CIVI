"""Three-tier SQLite lookup for authoritative source page text.

Used by the Stage 2 diagnostic to fetch the full page text for a
question's source URL so it can be injected into the prompt.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)


def normalize_url(url: str) -> str:
    """Normalize a URL for equality comparison.

    Lowercases the host, strips a leading ``www.`` from the host, strips a
    single trailing slash from the path, upgrades ``http`` → ``https``, and
    trims whitespace.  Applied to both sides of every URL comparison so
    minor format differences do not cause false negatives in the
    grounding-vs-retrieval split.
    """
    if not url:
        return ""
    parsed = urlparse(url.strip())
    scheme = "https" if parsed.scheme in ("http", "https") else parsed.scheme
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path.rstrip("/") if parsed.path != "/" else "/"
    return urlunparse(
        (scheme, netloc, path, parsed.params, parsed.query, parsed.fragment)
    )


class SourceLoader:
    """Look up a page's full text across per-jurisdiction SQLite corpora.

    Attributes:
        data_dir: Directory containing the 3-country COFOG
            ``{country}_{layer}_corpus.db`` SQLite files (e.g.
            ``australia_federal_corpus.db``, ``canada_municipal_corpus.db``).
    """

    def __init__(self, data_dir: Path | str = "data") -> None:
        """Initialise the loader.

        Args:
            data_dir: Directory holding ``*_corpus.db`` files for the 9
                COFOG jurisdictions.
        """
        self.data_dir = Path(data_dir)

    def lookup(self, source_url: str, city: str = "") -> str | None:
        """Return ``full_text`` for ``source_url`` or ``None`` if not found.

        Strategy:
            1. If ``city`` is supplied and ``{city}_corpus.db`` exists,
               try it first.  ``city`` here is the row's per-jurisdiction
               key — typically the institution short name from the COFOG
               config (e.g. ``australian_government``,
               ``canada_municipal``), not a literal city name.
            2. Otherwise (or on miss), scan every ``*_corpus.db`` in
               ``data_dir``.
            3. Return ``None`` if no database contains the URL.

        Args:
            source_url: URL to look up.  Normalized before comparison.
            city: Optional short-name key for the fast path. Legacy
                parameter name; kept for backward compatibility.

        Returns:
            The page's ``full_text`` column, or ``None``.
        """
        target = normalize_url(source_url)
        if not target:
            return None

        tried: set[Path] = set()

        if city:
            city_db = self.data_dir / f"{city.lower()}_corpus.db"
            if city_db.exists():
                text = self._lookup_in_db(city_db, target)
                tried.add(city_db)
                if text is not None:
                    return text

        for db_path in sorted(self.data_dir.glob("*_corpus.db")):
            if db_path in tried:
                continue
            text = self._lookup_in_db(db_path, target)
            if text is not None:
                return text

        return None

    @staticmethod
    def _lookup_in_db(db_path: Path, normalized_target: str) -> str | None:
        """Return ``full_text`` from a DB where the normalized URL matches."""
        try:
            conn = sqlite3.connect(str(db_path))
        except sqlite3.Error as exc:
            logger.warning("Could not open %s: %s", db_path, exc)
            return None

        try:
            rows = conn.execute(
                "SELECT source_uri, full_text FROM documents"
            ).fetchall()
        except sqlite3.Error as exc:
            logger.warning("Query failed on %s: %s", db_path, exc)
            return None
        finally:
            conn.close()

        for uri, text in rows:
            if normalize_url(uri) == normalized_target:
                return text
        return None
