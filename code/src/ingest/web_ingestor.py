"""Web link ingestor using trafilatura for main-body extraction.

Reads URLs from a text file (one per line), fetches each page via
trafilatura's downloader, extracts the main article text (stripping
boilerplate, nav, ads), normalizes, chunks, and returns ``Document``
objects ready for the corpus store.

Trafilatura handles DOM-based content extraction and boilerplate
removal.  On top of that, the ``TextNormalizer`` filters out residual
short-line noise (UI fragments, accessibility links, menu text) that
occasionally survives extraction.

For pages with collapsible / accordion sections (Bootstrap collapse,
Kadence accordions, etc.), trafilatura may miss hidden content.  A
fallback lxml-based extractor is used when trafilatura returns
suspiciously little text relative to the HTML size.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import trafilatura
from lxml import html as lxml_html

from src.core.models import Chunk, Document, DocumentMetadata
from src.ingest.chunker import SectionAwareChunker, TextChunker
from src.ingest.normalizer import TextNormalizer

logger = logging.getLogger(__name__)

# URL patterns that typically point to interactive tools / portals
# rather than informational content worth extracting.
_TOOL_URL_KEYWORDS = frozenset([
    "finder", "tool", "submit", "submission", "application-submission",
])

# Tags to strip when doing lxml fallback extraction.
_STRIP_TAGS = frozenset([
    "script", "style", "nav", "header", "footer", "noscript",
    "iframe", "svg", "form",
])

# XPaths to locate the main content area, tried in order.
_MAIN_CONTENT_XPATHS = [
    "//main",
    "//article",
    '//*[contains(@class, "entry-content")]',
    '//*[contains(@class, "page-content")]',
    '//*[@id="content"]',
    '//*[@role="main"]',
]

# If trafilatura extracts less than this fraction of the lxml text,
# prefer the lxml result.
_FALLBACK_RATIO_THRESHOLD = 0.4


class WebIngestor:
    """Fetch, extract, normalize, and chunk web pages.

    Each URL is processed independently.  Failed URLs are logged and
    skipped so that one bad page does not abort the entire batch.

    Attributes:
        normalizer: Shared ``TextNormalizer`` instance.
        chunker: Shared ``TextChunker`` instance.
        request_delay: Seconds to wait between consecutive HTTP requests
            (politeness throttle).
        timeout: HTTP request timeout in seconds.
        skip_tool_pages: When ``True``, URLs whose path contains any of
            the ``_TOOL_URL_KEYWORDS`` are skipped automatically.
    """

    def __init__(
        self,
        normalizer: TextNormalizer | None = None,
        chunker: TextChunker | None = None,
        request_delay: float = 0.5,
        timeout: int = 30,
        skip_tool_pages: bool = False,
    ) -> None:
        """Initialise the web ingestor.

        Args:
            normalizer: Text normalizer.  A default instance is created
                if ``None``.
            chunker: Text chunker.  A default instance is created if
                ``None``.
            request_delay: Seconds to sleep between requests.
            timeout: Per-request HTTP timeout in seconds.
            skip_tool_pages: Whether to auto-skip URLs that look like
                interactive tools (finders, submission portals) rather
                than informational pages.
        """
        self.normalizer = normalizer or TextNormalizer()
        self.chunker = chunker or SectionAwareChunker()
        self.request_delay = request_delay
        self.timeout = timeout
        self.skip_tool_pages = skip_tool_pages

    def ingest_from_file(self, weblinks_path: str | Path) -> list[Document]:
        """Read URLs from a text file and ingest each one.

        Blank lines and lines starting with ``#`` are ignored.

        Args:
            weblinks_path: Path to a ``.txt`` file with one URL per line.

        Returns:
            List of successfully ingested ``Document`` objects.
        """
        path = Path(weblinks_path)
        urls = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        logger.info("Found %d URLs in %s", len(urls), path)
        return self.ingest_urls(urls)

    def ingest_from_json_config(self, config_path: str | Path) -> list[Document]:
        """Read URLs and metadata from a JSON config file and ingest each one.

        The JSON file should have this structure::

            {
                "institution": "City of Toronto",
                "short_name": "toronto",
                "pages": [
                    {
                        "url": "https://example.com/page1",
                        "category": "Property Tax",
                        "label": "Property Tax Rates and Payments"
                    }
                ]
            }

        The ``institution``, ``short_name``, ``category``, and ``label``
        fields are stored in each document's ``metadata.extra`` dict so
        they persist in the database.

        Args:
            config_path: Path to the ``.json`` config file.

        Returns:
            List of successfully ingested ``Document`` objects.
        """
        path = Path(config_path)
        config = json.loads(path.read_text(encoding="utf-8"))

        institution = config.get("institution", "")
        short_name = config.get("short_name", "")
        pages = config.get("pages", [])
        logger.info(
            "Found %d pages in %s (institution=%s, short_name=%s)",
            len(pages), path, institution, short_name,
        )

        documents: list[Document] = []
        for i, page in enumerate(pages):
            url = page["url"]
            extra_meta = {
                "institution": institution,
                "institution_short_name": short_name,
                "category": page.get("category", ""),
                "label": page.get("label", ""),
            }

            if self.skip_tool_pages and self._is_tool_url(url):
                logger.info("[%d/%d] Skipping tool-like page: %s", i + 1, len(pages), url)
                continue

            logger.info("[%d/%d] Fetching: %s (category=%s, label=%s)",
                        i + 1, len(pages), url,
                        extra_meta["category"], extra_meta["label"])
            try:
                doc = self.ingest_url(url, extra_meta=extra_meta)
                if doc:
                    documents.append(doc)
                    logger.info(
                        "  OK: %s (%d chars, %d chunks)",
                        doc.metadata.title or "(no title)",
                        len(doc.full_text),
                        len(doc.chunks),
                    )
                else:
                    logger.warning("  No content extracted from %s", url)
            except Exception:
                logger.exception("  Failed to ingest %s", url)

            if i < len(pages) - 1:
                time.sleep(self.request_delay)

        logger.info("Ingested %d / %d URLs from JSON config", len(documents), len(pages))
        return documents

    def ingest_urls(self, urls: list[str]) -> list[Document]:
        """Ingest a list of URLs sequentially.

        URLs that fail to download or yield no extractable content are
        logged and skipped.

        Args:
            urls: List of absolute HTTP(S) URLs.

        Returns:
            List of successfully ingested ``Document`` objects.
        """
        documents: list[Document] = []
        for i, url in enumerate(urls):
            if self.skip_tool_pages and self._is_tool_url(url):
                logger.info("[%d/%d] Skipping tool-like page: %s", i + 1, len(urls), url)
                continue

            logger.info("[%d/%d] Fetching: %s", i + 1, len(urls), url)
            try:
                doc = self.ingest_url(url)
                if doc:
                    documents.append(doc)
                    logger.info(
                        "  OK: %s (%d chars, %d chunks)",
                        doc.metadata.title or "(no title)",
                        len(doc.full_text),
                        len(doc.chunks),
                    )
                else:
                    logger.warning("  No content extracted from %s", url)
            except Exception:
                logger.exception("  Failed to ingest %s", url)

            if i < len(urls) - 1:
                time.sleep(self.request_delay)

        logger.info("Ingested %d / %d URLs", len(documents), len(urls))
        return documents

    def ingest_url(
        self,
        url: str,
        extra_meta: dict[str, Any] | None = None,
    ) -> Document | None:
        """Fetch, extract, normalize, and chunk a single URL.

        Uses trafilatura to download HTML and extract the main article
        body.  If trafilatura returns suspiciously little text (common
        with pages that hide content in collapsible sections), falls
        back to lxml-based extraction.

        Args:
            url: An absolute HTTP(S) URL.
            extra_meta: Optional dict of additional metadata fields
                (e.g. ``category``, ``label``, ``institution``) to merge
                into ``DocumentMetadata.extra``.

        Returns:
            A ``Document`` with populated metadata and chunks, or
            ``None`` if the page could not be downloaded or contained
            no extractable content.
        """
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            logger.warning("Failed to download: %s", url)
            return None

        # Primary extraction via trafilatura
        result = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            output_format="txt",
            with_metadata=False,
        )

        # Fallback: if trafilatura got very little text, try lxml
        lxml_text = self._extract_with_lxml(downloaded)
        if lxml_text:
            traf_len = len(result) if result else 0
            if traf_len < len(lxml_text) * _FALLBACK_RATIO_THRESHOLD:
                logger.info(
                    "  Trafilatura got %d chars vs lxml %d chars — "
                    "using lxml fallback (page likely has collapsed sections)",
                    traf_len, len(lxml_text),
                )
                result = lxml_text

        if not result or not result.strip():
            return None

        # Extract title via bare_extraction (single pass)
        title = self._extract_title(downloaded)

        # Normalize (includes short-line filtering for UI noise)
        normalized_text = self.normalizer.normalize(result)
        if not normalized_text:
            return None

        # Build document
        doc_id = Document.make_doc_id(url)
        metadata = DocumentMetadata.web(source_uri=url, title=title)
        if extra_meta:
            metadata.extra.update(extra_meta)
        metadata.compute_content_hash(normalized_text)

        # Chunk
        chunks = self.chunker.chunk(normalized_text, doc_id)

        return Document(
            doc_id=doc_id,
            metadata=metadata,
            full_text=normalized_text,
            chunks=chunks,
        )

    @staticmethod
    def _extract_with_lxml(html: str) -> str | None:
        """Extract text from HTML using lxml, including hidden content.

        Finds the main content area via common XPath selectors, strips
        non-content tags (scripts, styles, nav, forms), and returns the
        text content.  This captures text inside collapsible sections
        that trafilatura may miss.

        Args:
            html: Raw HTML string.

        Returns:
            Extracted text, or ``None`` if no main content area found.
        """
        try:
            tree = lxml_html.fromstring(html)
        except Exception:
            return None

        # Remove non-content elements, preserving tail text so that
        # content following a stripped tag (e.g. text or accordion
        # headers like <button>/<summary> after a <script>) is not lost.
        for tag in list(tree.iter()):
            if tag.tag in _STRIP_TAGS:
                parent = tag.getparent()
                if parent is not None:
                    if tag.tail:
                        prev = tag.getprevious()
                        if prev is not None:
                            prev.tail = (prev.tail or "") + tag.tail
                        else:
                            parent.text = (parent.text or "") + tag.tail
                    parent.remove(tag)

        # Find main content area
        main_el = None
        for xpath in _MAIN_CONTENT_XPATHS:
            found = tree.xpath(xpath)
            if found:
                main_el = found[0]
                break

        if main_el is None:
            return None

        text = main_el.text_content()

        # Basic whitespace cleanup
        text = re.sub(r"\t+", " ", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip() or None

    @staticmethod
    def _extract_title(downloaded: str) -> str:
        """Extract the page title from downloaded HTML.

        Tries ``og:title`` first (usually the cleanest), then falls
        back to the HTML ``<title>`` tag with site-name suffixes
        stripped.

        Args:
            downloaded: Raw HTML string as returned by
                ``trafilatura.fetch_url``.

        Returns:
            The page title, or an empty string if not found.
        """
        try:
            tree = lxml_html.fromstring(downloaded)
        except Exception:
            return ""

        # Prefer og:title — usually the cleanest form
        og = tree.xpath('//meta[@property="og:title"]/@content')
        if og and og[0].strip():
            return og[0].strip()

        # Fall back to <title> tag, strip common site-name suffixes
        title_el = tree.find(".//title")
        if title_el is not None:
            raw = title_el.text_content().strip()
            # "Page Name - Site Name" or "Page Name | Site Name"
            raw = re.split(r"\s*[|\-–—]\s*", raw, maxsplit=1)[0].strip()
            if raw:
                return raw

        return ""

    @staticmethod
    def _is_tool_url(url: str) -> bool:
        """Check whether a URL likely points to an interactive tool.

        Interactive pages (search finders, submission portals) rarely
        contain factual text content useful for QA generation.

        Args:
            url: The URL to check.

        Returns:
            ``True`` if the URL path contains any tool keyword.
        """
        url_lower = url.lower()
        return any(kw in url_lower for kw in _TOOL_URL_KEYWORDS)
