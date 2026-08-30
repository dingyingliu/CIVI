"""HTML pre-cleaning pass to remove noise before main-body extraction.

Trafilatura's main-body algorithm sometimes selects a non-content element
(site-wide alert banners, registration error widgets, modals) as "main" and
discards the actual article body.  Stripping these noise elements before
trafilatura sees them forces it to pick the right element.

Usage::

    from src.ingest.html_cleaner import clean_html
    cleaned = clean_html(raw_html)
"""

from __future__ import annotations

import logging
import re

from lxml import html as lxml_html

logger = logging.getLogger(__name__)

# Always-strip tag names. Restricted to elements that never carry
# article text. We deliberately do NOT strip:
#   * <nav>, <header>, <footer>, <aside> -- trafilatura analyses those
#     structurally, and some templates (e.g., SharePoint-based sites)
#     nest the article body inside a <header>/<aside> ancestor.
#   * <form> -- ASP.NET / SharePoint sites wrap the entire page body in
#     a single <form runat="server">, so stripping it would delete the
#     article. Trafilatura already filters real form fields via the
#     element-level metadata it tracks.
_STRIP_TAGS = (
    "script", "style", "noscript", "iframe", "svg",
)

# Strip elements whose ``class`` or ``id`` contains any of these substrings.
# Each entry is a lowercase substring; matching is case-insensitive.
# Conservative — must be likely-noise across many sites, not just one.
_STRIP_CLASS_OR_ID_SUBSTRINGS = (
    # Cookie / consent banners
    "cookie-banner", "cookieconsent", "cookie-notice", "consent-banner",
    "gdpr",
    # Site-wide notification bars (the kind that appear on every page)
    "notification-bar", "alert-banner", "site-alert", "global-alert",
    "emergency-banner", "announcement-bar",
    # Modal/popup chrome
    "modal-backdrop", "popup-overlay", "lightbox-overlay",
    # Search widgets often pulled in as "main" by readability
    "search-widget", "search-bar-container", "site-search",
    # Social share strips
    "social-share", "share-buttons", "addthis",
    # Skip-link / a11y helper bars (often the first innerText)
    "skip-link", "skip-to-main", "a11y-bar",
    # Newsletter signup blocks
    "newsletter-signup", "subscribe-form",
    # Breadcrumbs (we don't want them counted as content)
    "breadcrumbs", "breadcrumb-trail",
    # Print/share footers
    "page-tools", "page-actions", "print-tools",
)

# Strip elements with these exact role attributes. Note: we intentionally
# do NOT strip role="main" or role="article".
_STRIP_ROLES = (
    "alert", "alertdialog", "dialog", "banner", "complementary",
    "navigation", "search",
)


def _matches_noise(class_attr: str, id_attr: str) -> bool:
    haystack = f"{class_attr} {id_attr}".lower()
    return any(needle in haystack for needle in _STRIP_CLASS_OR_ID_SUBSTRINGS)


def clean_html(html: str) -> str:
    """Return ``html`` with known noise elements removed.

    Drops:
      * <script>, <style>, <nav>, <header>, <footer>, <aside>, <iframe>, etc.
      * Elements whose ``class`` or ``id`` contains a known-noise substring
        (cookie banners, alert bars, modals, share strips, etc.).
      * Elements with role attributes indicating non-content widgets
        (alert, dialog, banner, search, etc.).

    The cleaned HTML is returned as a UTF-8 string suitable for
    ``trafilatura.extract`` or any other extractor.

    If parsing fails (malformed HTML), the original string is returned
    unchanged so the caller can still attempt extraction.
    """
    try:
        tree = lxml_html.fromstring(html)
    except Exception:
        logger.warning("clean_html: lxml parse failed, returning original")
        return html

    # 1. Drop always-strip tags
    for tag in _STRIP_TAGS:
        for el in tree.xpath(f"//{tag}"):
            _drop(el)

    # 2. Drop by class/id substring match
    for el in tree.xpath("//*[@class or @id]"):
        cls = el.get("class") or ""
        eid = el.get("id") or ""
        if _matches_noise(cls, eid):
            _drop(el)

    # 3. Drop by role attribute
    for el in tree.xpath("//*[@role]"):
        role = (el.get("role") or "").strip().lower()
        if role in _STRIP_ROLES:
            _drop(el)

    return lxml_html.tostring(tree, encoding="unicode")


def _drop(el) -> None:
    """Remove an lxml element while preserving any tail text on its parent."""
    parent = el.getparent()
    if parent is None:
        return
    # Preserve trailing whitespace/text following the element
    if el.tail:
        previous = el.getprevious()
        if previous is not None:
            previous.tail = (previous.tail or "") + el.tail
        else:
            parent.text = (parent.text or "") + el.tail
    parent.remove(el)
