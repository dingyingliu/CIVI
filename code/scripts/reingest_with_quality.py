"""Re-ingest pages using the improved extraction pipeline.

Pipeline per URL:

    PlaywrightFetcher(expand_disclosures=True).fetch_html(url)
        -> improved_extractor.extract_best(html, url)
        -> TextNormalizer.normalize(text)
        -> SectionAwareChunker.chunk(text, doc_id)
        -> CorpusStore.add_documents([doc])  (upsert)

Two ways to scope a run:

  * ``--web-config <path>`` -- ingest every URL in the given JSON config.
    Two config shapes are auto-detected:
      - Single-institution (legacy):
          {"institution", "short_name",
           "pages": [{"url","category","label"}, ...]}
        DB path: ``data/{short_name}_corpus.db``.
      - Multi-institution (unified):
          {"country", "layer",
           "institutions": [{"institution","short_name","pages":[...]} , ...]}
        Per-page fields: ``url``, ``cell``, ``cell_name``, ``label``.
        ``cell_name`` is mapped to the ``category`` metadata slot so the
        existing Excel column logic continues to work. ``cell``,
        ``country``, ``layer``, ``institution``, and
        ``institution_short_name`` are stashed in ``metadata.extra``.
        DB path: ``data/{country}_{layer}_corpus.db`` (lowercased,
        spaces -> underscores) -- one DB per file, all institutions in it.
  * ``--urls-csv <path>`` -- ingest a list of URLs from a CSV with at
    least the columns ``city`` and ``url`` (the audit's output works
    directly).  Use ``--suspect-only`` to filter to ``suspect=yes`` rows.

If neither is given, every config under ``data/city_json/`` and
``data/3 country cofog/`` is re-ingested.

Logs go to ``logs/reingest.log`` by default (override with ``--log``).
Per-doc strategy choice is recorded in ``metadata.extra['fetched_via']``
and ``metadata.extra['extraction_strategy']`` for later analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.models import Document, DocumentMetadata
from src.ingest.chunker import SectionAwareChunker
from src.ingest.improved_extractor import extract_best
from src.ingest.normalizer import TextNormalizer
from src.ingest.playwright_fetcher import PlaywrightFetcher
from src.ingest.store import CorpusStore

logger = logging.getLogger("reingest")

# Directories scanned for ingest configs. ``city_json`` holds the legacy
# per-city files; ``3 country cofog`` holds the unified multi-institution
# files (country/layer/institutions[]).
CONFIG_DIRS: tuple[Path, ...] = (
    Path("data/city_json"),
    Path("data/3 country cofog"),
)


def build_document(
    url: str,
    html: str,
    extra_meta: dict[str, Any],
    normalizer: TextNormalizer,
    chunker: SectionAwareChunker,
    *,
    try_static: bool = True,
) -> tuple[Document | None, str]:
    """Run improved extraction + normalize + chunk; return (doc, strategy).

    ``try_static=False`` disables the 5th rescue strategy (static
    ``trafilatura.fetch_url``).  Useful for slow/firewalled servers
    (e.g. AU gov sites) that hang HTTP requests for ~120 s before
    eventually giving up — the playwright-fetched HTML already has
    everything those sites can give us.
    """
    text, strategy = extract_best(html, url, try_static=try_static)
    if not text:
        return None, strategy

    normalized = normalizer.normalize(text)
    if not normalized:
        return None, strategy

    doc_id = Document.make_doc_id(url)
    metadata = DocumentMetadata.web(source_uri=url, title="")  # title set below
    metadata.extra.update(extra_meta)
    metadata.extra["fetched_via"] = "playwright_stealth_expand"
    metadata.extra["extraction_strategy"] = strategy
    metadata.compute_content_hash(normalized)

    # Prefer the curated label from the JSON config; fall back to the title
    # extracted from the page HTML. The HTML helper sometimes returns a
    # short heading fragment (e.g. "On") on pages where the document
    # structure confuses the extractor, so a hand-vetted label wins.
    from src.ingest.web_ingestor import WebIngestor
    metadata.title = (
        extra_meta.get("label")
        or WebIngestor._extract_title(html)
        or ""
    )

    chunks = chunker.chunk(normalized, doc_id)
    return Document(
        doc_id=doc_id,
        metadata=metadata,
        full_text=normalized,
        chunks=chunks,
    ), strategy


def _normalize_path_part(value: str) -> str:
    """Lowercase + collapse spaces to underscores for DB path components."""
    return value.strip().lower().replace(" ", "_")


def _jobs_for_institution_block(
    inst_block: dict,
    db_path: str,
    *,
    country: str = "",
    layer: str = "",
) -> list[tuple[str, dict[str, str], str]]:
    """Flatten one ``{institution, short_name, pages[]}`` block into job tuples."""
    institution = inst_block.get("institution", "")
    short_name = inst_block.get("short_name", "")
    out: list[tuple[str, dict[str, str], str]] = []
    for page in inst_block.get("pages", []):
        # Unified configs use ``cell_name``; legacy use ``category``. Either way
        # the value lands in the ``category`` slot so downstream Excel columns
        # keep working unchanged.
        category = page.get("cell_name") or page.get("category", "")
        meta: dict[str, str] = {
            "institution": institution,
            "institution_short_name": short_name,
            "category": category,
            "label": page.get("label", ""),
        }
        if "cell" in page:
            meta["cell"] = page["cell"]
        if country:
            meta["country"] = country
        if layer:
            meta["layer"] = layer
        out.append((page["url"], meta, db_path))
    return out


def load_jobs_from_config(path: Path) -> list[tuple[str, dict[str, str], str]]:
    """Return ``(url, extra_meta, db_path)`` jobs from one config file.

    Schema is auto-detected: presence of ``institutions[]`` at the root means
    the unified multi-institution shape (one DB per file at
    ``data/{country}_{layer}_corpus.db``); otherwise the legacy
    single-institution shape (``data/{short_name}_corpus.db``).
    """
    config = json.loads(path.read_text(encoding="utf-8"))
    if "institutions" in config:
        country = config.get("country", "")
        layer = config.get("layer", "")
        db_path = (
            f"data/{_normalize_path_part(country)}_"
            f"{_normalize_path_part(layer)}_corpus.db"
        )
        jobs: list[tuple[str, dict[str, str], str]] = []
        for inst_block in config.get("institutions", []):
            jobs.extend(
                _jobs_for_institution_block(
                    inst_block, db_path, country=country, layer=layer
                )
            )
        return jobs
    short_name = config.get("short_name", "")
    db_path = f"data/{short_name}_corpus.db"
    return _jobs_for_institution_block(config, db_path)


def load_urls_from_csv(path: Path, suspect_only: bool) -> list[tuple[str, str]]:
    """Return list of (city, url) from an audit CSV."""
    out = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if suspect_only and row.get("suspect") != "yes":
                continue
            out.append((row["city"], row["url"]))
    return out


def lookup_meta_for_url(url: str, city: str) -> tuple[dict[str, str], str]:
    """Find ``(extra_meta, db_path)`` for a URL by scanning every config dir.

    Walks both ``data/city_json/`` (legacy) and ``data/3 country cofog/``
    (unified) and returns the first match. Falls back to a minimal meta dict
    and ``data/{city}_corpus.db`` if the URL isn't found anywhere.
    """
    for cfg_dir in CONFIG_DIRS:
        if not cfg_dir.exists():
            continue
        for cfg in cfg_dir.glob("*.json"):
            for u, meta, db in load_jobs_from_config(cfg):
                if u == url:
                    return meta, db
    return (
        {"institution": "", "institution_short_name": city,
         "category": "", "label": ""},
        f"data/{city}_corpus.db",
    )


def reingest_pages(
    pages: list[tuple[str, dict[str, str], str]],  # (url, extra_meta, db_path)
    fetcher: PlaywrightFetcher,
    normalizer: TextNormalizer,
    chunker: SectionAwareChunker,
    delay: float,
    *,
    try_static: bool = True,
) -> dict:
    """Ingest a flat list of (url, extra_meta, db_path) tuples.

    Stores upsert into the indicated DB, returns counts.
    """
    by_db: dict[str, CorpusStore] = {}
    counts = {"ok": 0, "fail_fetch": 0, "fail_extract": 0, "by_strategy": defaultdict(int)}

    for i, (url, meta, db_path) in enumerate(pages):
        logger.info("[%d/%d] %s", i + 1, len(pages), url)
        html = fetcher.fetch_html(url)
        if not html:
            logger.warning("  fetch failed")
            counts["fail_fetch"] += 1
            if i < len(pages) - 1:
                time.sleep(delay)
            continue

        try:
            doc, strategy = build_document(
                url, html, meta, normalizer, chunker, try_static=try_static,
            )
        except Exception:
            logger.exception("  build_document raised")
            counts["fail_extract"] += 1
            if i < len(pages) - 1:
                time.sleep(delay)
            continue

        if doc is None:
            logger.warning("  no usable extraction (strategy=%s)", strategy)
            counts["fail_extract"] += 1
        else:
            store = by_db.setdefault(db_path, CorpusStore(db_path))
            store.add_documents([doc])
            counts["ok"] += 1
            counts["by_strategy"][strategy] += 1
            logger.info(
                "  OK [%s] %s (%d chars, %d chunks)",
                strategy,
                doc.metadata.title or "(no title)",
                len(doc.full_text),
                len(doc.chunks),
            )
        if i < len(pages) - 1:
            time.sleep(delay)

    for store in by_db.values():
        store.close()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-ingest pages with the improved extraction pipeline",
    )
    parser.add_argument("--web-config", type=str, default=None,
                        help="Path to a single city JSON config")
    parser.add_argument("--urls-csv", type=str, default=None,
                        help="CSV with 'city' and 'url' columns "
                             "(audit output works directly)")
    parser.add_argument("--suspect-only", action="store_true",
                        help="With --urls-csv: only ingest rows where suspect=yes")
    parser.add_argument("--all-cities", action="store_true",
                        help="Re-ingest every config under data/city_json/ "
                             "and data/3 country cofog/ (schema auto-detected)")
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--delay", type=float, default=0.4,
                        help="seconds between requests")
    parser.add_argument("--no-static-fetch", action="store_true",
                        help="Skip the 5th rescue strategy (static "
                             "trafilatura.fetch_url). Use this for "
                             "slow/firewalled servers (e.g. AU gov sites) "
                             "that hang HTTP requests for ~120s before "
                             "timing out. The playwright HTML already has "
                             "everything those sites can give us.")
    parser.add_argument("--log", type=str, default="logs/reingest.log")
    args = parser.parse_args()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    # Build flat (url, extra_meta, db_path) job list
    jobs: list[tuple[str, dict[str, str], str]] = []
    if args.web_config:
        jobs.extend(load_jobs_from_config(Path(args.web_config)))
    elif args.urls_csv:
        for city, url in load_urls_from_csv(Path(args.urls_csv), args.suspect_only):
            meta, db = lookup_meta_for_url(url, city)
            jobs.append((url, meta, db))
    elif args.all_cities:
        for cfg_dir in CONFIG_DIRS:
            if not cfg_dir.exists():
                continue
            for cfg in sorted(cfg_dir.glob("*.json")):
                jobs.extend(load_jobs_from_config(cfg))
    else:
        parser.error("Must provide --web-config, --urls-csv, or --all-cities")

    logger.info("Starting re-ingest of %d URLs", len(jobs))

    normalizer = TextNormalizer()
    chunker = SectionAwareChunker(
        max_chunk_chars=args.chunk_size,
        overlap_chars=args.overlap,
    )

    with PlaywrightFetcher(expand_disclosures=True) as fetcher:
        counts = reingest_pages(
            jobs, fetcher, normalizer, chunker, args.delay,
            try_static=not args.no_static_fetch,
        )

    logger.info("=== Re-ingest summary ===")
    logger.info("  ok=%d  fail_fetch=%d  fail_extract=%d",
                counts["ok"], counts["fail_fetch"], counts["fail_extract"])
    logger.info("  strategies: %s", dict(counts["by_strategy"]))


if __name__ == "__main__":
    main()
