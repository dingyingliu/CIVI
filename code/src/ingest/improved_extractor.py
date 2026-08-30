"""Multi-strategy HTML-to-text extractor for QA-generation corpus building.

The original ``WebIngestor`` pipeline relies on a single trafilatura.extract
call with the default settings and a 40%-of-lxml fallback.  An audit of
572 corpus pages found that pattern under-extracts on 17 pages (5--53% of
real content) due to three systemic issues:

  1. Collapsed accordion / disclosure widgets dropped from output.
  2. Early truncation on certain templates (no accordions, just stops).
  3. Wrong main-element selected when a site-wide alert banner is present.

This extractor addresses all three by:

  * Pre-cleaning known noise out of the HTML before extraction
    (see ``src.ingest.html_cleaner``).  Handles issue 3.
  * Running multiple extractors and choosing the longest non-junk output:
        - trafilatura with ``favor_recall=True`` (looser inclusion)
        - trafilatura with default settings (current behaviour, baseline)
        - readability-lxml's ``Document.summary()``
        - ``WebIngestor._extract_with_lxml`` main-content XPath
    Handles issues 1 and 2 -- whichever extractor doesn't trip on the
    page-specific failure mode wins.
  * Rejecting outputs that look like known junk (bot challenges, tiny
    error banners, etc.) before length-comparison.

The expected fetch path is ``PlaywrightFetcher(expand_disclosures=True)``,
so accordions are already opened in the DOM before this runs.

Usage::

    from src.ingest.improved_extractor import extract_best
    text, strategy = extract_best(html, url)
"""

from __future__ import annotations

import logging

import trafilatura
from lxml import html as lxml_html
from readability import Document

from src.ingest.html_cleaner import clean_html
from src.ingest.web_ingestor import WebIngestor

logger = logging.getLogger(__name__)

# Reject extractor outputs containing any of these substrings (case-
# insensitive). These are the bot-challenge and known-error markers we've
# observed in the corpus to date.
_JUNK_MARKERS = (
    "think you were a bot",
    "pardon our interruption",
    "please enable javascript",
    "access to this page has been denied",
    "request unsuccessful. incapsula",
    "checking your browser before accessing",
)

# Below this many words an extraction is considered too thin to be useful,
# regardless of how it compares to other strategies.
_MIN_WORDS = 30


def _is_junk(text: str) -> bool:
    if not text:
        return True
    if len(text.split()) < _MIN_WORDS:
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in _JUNK_MARKERS)


def _try_trafilatura(html: str, *, favor_recall: bool) -> str:
    try:
        result = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            include_links=False,
            output_format="txt",
            with_metadata=False,
            favor_recall=favor_recall,
        )
        return result or ""
    except Exception:
        logger.exception("trafilatura.extract failed (favor_recall=%s)", favor_recall)
        return ""


def _try_readability(html: str) -> str:
    try:
        summary_html = Document(html).summary()
        if not summary_html:
            return ""
        tree = lxml_html.fromstring(summary_html)
        text = tree.text_content()
        # Collapse runs of whitespace to single spaces between non-newlines,
        # but preserve paragraph breaks (double-newline between blocks).
        return text.strip()
    except Exception:
        logger.exception("readability.Document.summary failed")
        return ""


def _try_lxml_main(html: str) -> str:
    try:
        return WebIngestor._extract_with_lxml(html) or ""
    except Exception:
        logger.exception("WebIngestor._extract_with_lxml failed")
        return ""


def _try_trafilatura_static(url: str) -> str:
    """Fetch the URL with trafilatura's direct HTTP downloader (no JS) and extract.

    This is a rescue strategy for sites whose JS framework strips content
    from the rendered DOM (calgary.ca's accordion widgets manage panel
    state in component memory, so the post-JS DOM has only toggle buttons).
    Trafilatura's downloader sees the pre-JS server response, where the
    panel content is present in static HTML.
    """
    if not url:
        return ""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return ""
        cleaned = clean_html(downloaded)
        result = trafilatura.extract(
            cleaned,
            include_comments=False,
            include_tables=True,
            include_links=False,
            output_format="txt",
            with_metadata=False,
            favor_recall=True,
        )
        return result or ""
    except Exception:
        logger.exception("static fetch+extract failed for %s", url)
        return ""


def extract_best(html: str, url: str = "", *, try_static: bool = True) -> tuple[str, str]:
    """Run multiple extractors and return the best output.

    Args:
        html: Raw HTML string (already rendered if from Playwright).
        url: Source URL, used only for logging context.

    Returns:
        ``(text, strategy_name)`` where ``text`` is the chosen extraction
        and ``strategy_name`` identifies which extractor produced it.
        If all strategies produce junk, returns ``("", "none")``.
    """
    cleaned = clean_html(html)

    candidates: list[tuple[str, str]] = []

    text = _try_trafilatura(cleaned, favor_recall=True)
    if not _is_junk(text):
        candidates.append((text, "trafilatura_recall"))

    text = _try_trafilatura(cleaned, favor_recall=False)
    if not _is_junk(text):
        candidates.append((text, "trafilatura_default"))

    text = _try_readability(cleaned)
    if not _is_junk(text):
        candidates.append((text, "readability"))

    text = _try_lxml_main(cleaned)
    if not _is_junk(text):
        candidates.append((text, "lxml_main"))

    # Rescue strategy: fetch the URL directly without JS rendering. Wins on
    # sites whose JS framework empties accordion panels from the post-render
    # DOM (e.g. calgary.ca) but keeps the content in the original server HTML.
    if try_static and url:
        text = _try_trafilatura_static(url)
        if not _is_junk(text):
            candidates.append((text, "trafilatura_static"))

    if not candidates:
        logger.warning("extract_best: no usable extraction for %s", url)
        return "", "none"

    # Pick longest by character count.  Word count would also work but
    # tables and lists distort word counts vs what's on the page.
    best_text, best_strategy = max(candidates, key=lambda x: len(x[0]))
    if logger.isEnabledFor(logging.DEBUG):
        ranked = sorted(candidates, key=lambda x: len(x[0]), reverse=True)
        logger.debug(
            "extract_best for %s: %s",
            url,
            ", ".join(f"{s}={len(t)}" for t, s in ranked),
        )
    return best_text, best_strategy
