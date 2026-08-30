"""Audit extraction quality across the corpus.

For every document in every *_corpus.db file, this script:
1. Re-fetches the source URL with stealth Playwright (networkidle wait,
   bounded at 15s with a domcontentloaded fallback for chatty pages)
2. Injects CSS overrides + ARIA flips + custom-accordion clicks
   (.accordion__section__title, .accordion-button, [data-toggle="collapse"])
   so hidden content becomes measurable
3. Measures the rendered innerText length of the main-content region —
   tries semantic selectors first, falls back to body-with-chrome-stripped
4. Compares to ``LENGTH(documents.full_text)`` stored in the DB
5. Flags pages where our extraction captured less than ``--threshold`` of
   what's actually on the page (default 0.5 == 50%)

Output: prints the suspect list directly to stdout, sorted ascending by
ratio (worst first), columns ``# | db | ratio | db_chars | page_chars | url``.
Cap at ``--cap`` rows (default 50) with a tail summary.

Usage::

    python scripts/audit_extraction.py
    python scripts/audit_extraction.py --threshold 0.4 --max-per-city 5
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.playwright_fetcher import PlaywrightFetcher

logger = logging.getLogger("audit_extraction")

# JS injected into each page before measurement.  Three steps in this order
# (CSS overrides -> attribute flips -> programmatic clicks) so that pages
# whose accordion content depends on CSS, ARIA state, OR a JS click handler
# all become measurable.
_PRE_MEASURE_JS = """
(() => {
  // 1. CSS overrides — force hidden accordion content visible regardless of
  //    the page's stylesheets. Uses !important so site CSS can't beat us.
  const style = document.createElement('style');
  style.textContent = `
    .accordion__section__content,
    .accordion-collapse,
    .collapse,
    .collapse.in,
    .collapse.show {
      display: block !important;
      max-height: none !important;
      height: auto !important;
      visibility: visible !important;
    }
  `;
  document.head.appendChild(style);

  // 2. Attribute flips — HTML5 details, ARIA, Bootstrap class state
  document.querySelectorAll('details').forEach(d => d.setAttribute('open', ''));
  document.querySelectorAll('.collapse').forEach(el => el.classList.add('show', 'in'));
  document.querySelectorAll('[aria-expanded="false"]').forEach(el => {
    el.setAttribute('aria-expanded', 'true');
  });
  document.querySelectorAll('[aria-hidden="true"]').forEach(el => {
    el.setAttribute('aria-hidden', 'false');
  });

  // 3. Click custom-accordion headers that need a JS handler to expand
  //    (Edmonton .accordion__section__title, Bootstrap 4/5, etc).
  //    try/catch each click so a bad handler doesn't kill the whole pass.
  const clickSels = [
    '.accordion__section__title',
    '.accordion-button.collapsed',
    '[data-toggle="collapse"]',
    'details:not([open]) > summary',
  ];
  clickSels.forEach(sel => {
    document.querySelectorAll(sel).forEach(el => {
      try { el.click(); } catch (e) {}
    });
  });
})();
"""

# Measures the rendered innerText length after expansion.  Tries semantic
# main-content selectors first; falls back to a *cloned-and-stripped* body
# (chrome removed) so heavy-nav municipal sites without semantic landmarks
# don't get a denominator inflated with menus and footers.
_MEASURE_TEXT_JS = """
(() => {
  const sels = [
    'main', 'article', '[role="main"]', '#content', '#main-content',
    '.entry-content', '.page-content', '.main-content',
  ];
  for (const s of sels) {
    const el = document.querySelector(s);
    if (el && el.innerText && el.innerText.length > 200) {
      return el.innerText.length;
    }
  }
  if (!document.body) return 0;
  // Fallback: clone body, strip nav/header/footer/aside, then measure.
  const clone = document.body.cloneNode(true);
  const stripSels = [
    'nav', 'header', 'footer', 'aside',
    '[role="navigation"]', '[role="banner"]', '[role="contentinfo"]',
    '.nav', '.navigation', '.footer', '.header', '.site-header', '.site-footer',
    'script', 'style', 'noscript',
  ];
  stripSels.forEach(sel => {
    clone.querySelectorAll(sel).forEach(el => el.remove());
  });
  return clone.innerText.length;
})();
"""


def find_corpus_dbs() -> list[Path]:
    return sorted(Path("data").glob("*_corpus.db"))


def collect_docs(db: Path) -> list[tuple[str, str, int]]:
    """Return list of (doc_id, source_uri, current_chars) for all docs in db."""
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("SELECT doc_id, source_uri, LENGTH(full_text) FROM documents")
    rows = cur.fetchall()
    conn.close()
    return rows


def measure_page(fetcher: PlaywrightFetcher, url: str, settle_ms: int = 2500) -> int | None:
    """Return main-content innerText length for url, or None if fetch failed.

    Uses ``networkidle`` (bounded at 15s) so SPA-style pages have a chance
    to render before measurement.  Falls back to ``domcontentloaded`` if
    networkidle times out — a few chatty analytics pages would never
    actually idle, and that fallback keeps any single URL bounded.
    """
    if fetcher._context is None:
        raise RuntimeError("PlaywrightFetcher not entered")
    page = fetcher._context.new_page()
    try:
        from playwright_stealth import stealth_sync
        try:
            stealth_sync(page)
        except Exception:
            pass
        try:
            try:
                response = page.goto(url, wait_until="networkidle", timeout=15000)
            except Exception:
                # networkidle never settled — fall back to domcontentloaded.
                response = page.goto(
                    url, wait_until="domcontentloaded", timeout=fetcher.timeout_ms,
                )
            if response is None or response.status >= 400:
                return None
            page.wait_for_timeout(settle_ms)
            page.evaluate(_PRE_MEASURE_JS)
            page.wait_for_timeout(400)  # let click-triggered reflow settle
            length = page.evaluate(_MEASURE_TEXT_JS)
            return int(length) if length else 0
        except Exception as exc:
            logger.warning("measure failed for %s: %s", url, exc)
            return None
    finally:
        page.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Audit corpus extraction quality")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="ratio under which a page is flagged suspect (default 0.5)")
    parser.add_argument("--max-per-city", type=int, default=0,
                        help="cap docs per city (0 = unlimited; useful for smoke tests)")
    parser.add_argument("--cities", type=str, default=None,
                        help="comma-separated city stems to limit to (default: all)")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="seconds between requests")
    parser.add_argument("--cap", type=int, default=50,
                        help="max suspect rows to print before summarizing the tail")
    args = parser.parse_args()

    target_cities: set[str] | None = None
    if args.cities:
        target_cities = {c.strip() for c in args.cities.split(",") if c.strip()}

    dbs = find_corpus_dbs()
    if target_cities:
        dbs = [d for d in dbs if d.stem.removesuffix("_corpus") in target_cities]
    logger.info("Auditing %d corpora", len(dbs))

    total = sum(min(len(collect_docs(d)), args.max_per_city or 999999) for d in dbs)
    logger.info("Total docs to audit: %d (~%d min @ ~3s each)", total, total * 3 // 60)

    # In-memory results: list of dicts (instead of CSV).
    results: list[dict] = []

    with PlaywrightFetcher() as fetcher:
        audited = 0
        for db in dbs:
            city = db.stem.removesuffix("_corpus")
            docs = collect_docs(db)
            if args.max_per_city:
                docs = docs[: args.max_per_city]
            for i, (doc_id, url, current_chars) in enumerate(docs):
                audited += 1
                page_chars = measure_page(fetcher, url)
                if page_chars is None:
                    ratio = -1.0
                    status = "fetch_failed"
                else:
                    ratio = current_chars / page_chars if page_chars else -1.0
                    status = (
                        "yes" if 0 < ratio < args.threshold
                        else ("empty_page" if page_chars < 100 else "no")
                    )
                results.append({
                    "db": city,
                    "doc_id": doc_id,
                    "url": url,
                    "current_chars": current_chars,
                    "page_chars": page_chars if page_chars is not None else 0,
                    "ratio": ratio,
                    "status": status,
                })
                logger.info(
                    "[%d/%d] %s ratio=%.2f db_chars=%d page_chars=%s status=%s",
                    audited, total, city,
                    ratio if ratio >= 0 else -1,
                    current_chars,
                    page_chars if page_chars is not None else "FAIL",
                    status,
                )
                time.sleep(args.delay)

    # ----- summary printed to stdout (in-chat output) -----
    suspects = [r for r in results if r["status"] in ("yes", "fetch_failed", "empty_page")]
    # Sort ascending by ratio; fetch_failed (ratio=-1) bubbles up first.
    suspects.sort(key=lambda r: (r["ratio"] if r["ratio"] >= 0 else -999, r["db"], r["url"]))

    print()
    print("=" * 78)
    print(f"AUDIT COMPLETE — {len(results)} docs audited, {len(suspects)} flagged")
    print("=" * 78)
    if not suspects:
        print("No suspects.")
        return

    # Status breakdown
    from collections import Counter
    by_status = Counter(r["status"] for r in suspects)
    print(f"By status: {dict(by_status)}")
    by_db = Counter(r["db"] for r in suspects)
    print(f"By db:     {dict(by_db)}")
    print()

    cap = args.cap
    print(f"{'#':>3}  {'db':<32}  {'ratio':>6}  {'db_chars':>9}  {'page_chars':>10}  url")
    print("-" * 130)
    for i, r in enumerate(suspects[:cap], 1):
        ratio_str = f"{r['ratio']:.2f}" if r["ratio"] >= 0 else "FAIL"
        url_short = r["url"] if len(r["url"]) <= 75 else r["url"][:72] + "..."
        print(f"{i:>3}  {r['db']:<32}  {ratio_str:>6}  {r['current_chars']:>9,}  "
              f"{r['page_chars']:>10,}  {url_short}")

    if len(suspects) > cap:
        tail = suspects[cap:]
        tail_by_db = Counter(r["db"] for r in tail)
        print("-" * 130)
        print(f"... + {len(tail)} more suspects (truncated). Tail by db: {dict(tail_by_db)}")


if __name__ == "__main__":
    main()
