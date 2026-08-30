"""Headless-browser HTML fetcher using Playwright.

Purpose: fetch pages that trafilatura's direct HTTP downloader can't get,
typically because the site is behind an anti-bot WAF (Incapsula/Imperva,
Cloudflare JS challenge, Akamai Bot Manager) or uses TLS fingerprint
blocking.

Usage::

    with PlaywrightFetcher() as fetcher:
        html = fetcher.fetch_html("https://example.com/page")
        if html:
            # feed into trafilatura.extract(html, ...) as usual
            ...

Design:
- A single Chromium browser + context is reused across URLs for speed.
- Stealth tweaks applied at the context level (UA, viewport, locale,
  ``navigator.webdriver`` removal) — enough to defeat basic WAF checks
  without a paid stealth plugin.
- Sync API (matches the rest of the ingestion pipeline, which is sync).
"""

from __future__ import annotations

import logging
from types import TracebackType
from typing import Self

from playwright.sync_api import (
    Browser,
    BrowserContext,
    Playwright,
    sync_playwright,
)
from playwright_stealth import stealth_sync

logger = logging.getLogger(__name__)

# Realistic Chrome-on-Windows UA. Kept recent but not bleeding-edge so
# it stays plausible across releases.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

# Strip automation signals that WAFs commonly fingerprint.
_STEALTH_INIT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
window.chrome = { runtime: {} };
"""

# JS that opens collapsed disclosure / accordion widgets so their hidden text
# appears in the rendered DOM. Restricted to obvious content-panel patterns —
# avoids un-hiding modals, dialogs, mobile nav drawers, dropdowns.
_EXPAND_DISCLOSURES_JS = """
(() => {
  // HTML5 details/summary
  document.querySelectorAll('details').forEach(d => d.setAttribute('open', ''));

  // Bootstrap collapse panels (.collapse + .show / .in)
  document.querySelectorAll('.collapse').forEach(el => {
    el.classList.add('show', 'in');
  });

  // Generic ARIA disclosure pattern used by most modern accordions
  document.querySelectorAll('[aria-expanded="false"]').forEach(el => {
    el.setAttribute('aria-expanded', 'true');
  });

  // aria-hidden=true on content panels (carefully scoped — not modals)
  const panelSelectors = (
    '[role="region"], .panel, .panel-body, .panel-collapse, ' +
    '.accordion-body, .accordion-content, .acc-content, ' +
    '.collapse-body, .collapse-content, ' +
    '[class*="accordion"][aria-hidden], [class*="collapse"][aria-hidden]'
  );
  document.querySelectorAll(panelSelectors).forEach(el => {
    if (el.getAttribute('aria-hidden') === 'true') {
      el.setAttribute('aria-hidden', 'false');
    }
    if (el.hasAttribute('hidden')) {
      el.removeAttribute('hidden');
    }
  });
})();
"""


class PlaywrightFetcher:
    """Reusable Playwright-backed HTML fetcher.

    Attributes:
        headless: Whether to run Chromium headless (default ``True``).
        timeout_ms: Per-page navigation timeout in milliseconds.
        settle_ms: Extra wait after DOMContentLoaded for late-loading
            scripts, accordions, and CSR content.
    """

    def __init__(
        self,
        headless: bool = True,
        timeout_ms: int = 30_000,
        settle_ms: int = 1500,
        expand_disclosures: bool = True,
    ) -> None:
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.settle_ms = settle_ms
        self.expand_disclosures = expand_disclosures
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None

    def __enter__(self) -> Self:
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
        )
        self._context = self._browser.new_context(
            user_agent=_USER_AGENT,
            viewport={"width": 1366, "height": 900},
            locale="en-US",
            timezone_id="America/Toronto",
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        self._context.add_init_script(_STEALTH_INIT_SCRIPT)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            if self._context is not None:
                self._context.close()
            if self._browser is not None:
                self._browser.close()
            if self._pw is not None:
                self._pw.stop()
        except Exception:
            logger.exception("Error shutting down Playwright")

    def fetch_html(self, url: str) -> str | None:
        """Fetch a URL and return its fully-rendered HTML.

        Args:
            url: Absolute HTTP(S) URL.

        Returns:
            The page's rendered HTML, or ``None`` if navigation failed
            or the response status was not in the 2xx range.
        """
        if self._context is None:
            raise RuntimeError("PlaywrightFetcher must be used as a context manager")

        page = self._context.new_page()
        # Apply tf-playwright-stealth patches to defeat fingerprint-based WAFs
        # (canvas, WebGL, fonts, codecs, plugins, navigator props, etc.).
        try:
            stealth_sync(page)
        except Exception:
            logger.exception("stealth_sync failed; continuing without it")
        try:
            response = page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=self.timeout_ms,
            )
            if response is None:
                logger.warning("No response object for %s", url)
                return None
            status = response.status
            if status >= 400:
                logger.warning("HTTP %d for %s", status, url)
                return None
            # Give late-loading JS a moment to paint collapsibles / CSR content
            page.wait_for_timeout(self.settle_ms)
            if self.expand_disclosures:
                try:
                    page.evaluate(_EXPAND_DISCLOSURES_JS)
                    page.wait_for_timeout(200)  # allow reflow
                except Exception:
                    logger.exception("expand_disclosures JS failed for %s", url)
            return page.content()
        except Exception as exc:
            logger.warning("Playwright navigation failed for %s: %s", url, exc)
            return None
        finally:
            page.close()
