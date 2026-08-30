"""Run the QA generation pipeline.

Usage:
    # Single jurisdiction corpus DB
    python scripts/run_generator.py \
        --db data/canada_federal_corpus.db \
        --output output/qa_pairs/canada_federal_qa_pairs.xlsx --max-qa 120

    # Fan out across every COFOG config in data/3 country cofog/
    python scripts/run_generator.py --all-layers --output-dir output/qa_pairs

    # Selective regen of one jurisdiction
    python scripts/run_generator.py \
        --db data/australia_municipal_corpus.db \
        --output output/qa_pairs/australia_municipal_qa_pairs.xlsx
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so "src.*" imports resolve
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env file if it exists (for OPENROUTER_API_KEY)
_env_path = _PROJECT_ROOT / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            value = value.strip().strip("'\"")
            os.environ.setdefault(key.strip(), value)

from src.ingest.store import CorpusStore
from src.generate.generator import QAGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate QA pairs from the ingested corpus",
    )
    parser.add_argument(
        "--db", type=str, default="data/canada_federal_corpus.db",
        help="Corpus database path. Expected naming: "
             "data/{country}_{layer}_corpus.db.",
    )
    parser.add_argument(
        "--output", type=str, default="output/qa_pairs/qa_pairs.xlsx",
        help="Output Excel file path (default: output/qa_pairs/qa_pairs.xlsx)",
    )
    parser.add_argument(
        "--prompts-dir", type=str, default="prompts",
        help="Directory containing YAML prompt files (default: prompts)",
    )
    parser.add_argument(
        "--source-type", type=str, choices=["web", "pdf"],
        help="Process only documents of this source type",
    )
    parser.add_argument(
        "--doc-ids", type=str, nargs="+",
        help="Process only these specific document IDs",
    )
    parser.add_argument(
        "--mc-options", type=int, default=8,
        help="Number of MC options per question (default: 8, range 7–10)",
    )
    parser.add_argument(
        "--min-qa-per-doc", type=int, default=3,
        help="Minimum QA pairs per document (default: 3)",
    )
    parser.add_argument(
        "--max-qa-per-doc", type=int, default=10,
        help="Maximum QA pairs per document (default: 10)",
    )
    parser.add_argument(
        "--max-qa", type=int, default=None,
        help="Stop after generating this many QA pairs (default: no limit)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed for reproducibility (default: None = truly random)",
    )
    parser.add_argument(
        "--city", type=str, default="",
        help="City label written to every row of the output Excel. "
             "Optional; with --all-layers, the per-row institution_short_name "
             "is used instead.",
    )
    parser.add_argument(
        "--all-layers", action="store_true",
        help="Run the generator across every unified-shape config in "
             "data/3 country cofog/ — one .xlsx per corpus DB.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/qa_pairs",
        help="With --all-layers: directory for per-layer .xlsx files "
             "(default: output/qa_pairs).",
    )
    args = parser.parse_args()

    if args.all_layers:
        _run_all_layers(args)
        return

    store = CorpusStore(args.db)
    counts = store.count()
    if counts["chunks"] == 0:
        print("Error: corpus is empty. Run the ingester first.")
        sys.exit(1)

    print(f"Corpus loaded: {counts['documents']} documents, {counts['chunks']} chunks")

    generator = QAGenerator(
        store=store,
        prompts_dir=args.prompts_dir,
        mc_num_options=args.mc_options,
        min_qa_per_doc=args.min_qa_per_doc,
        max_qa_per_doc=args.max_qa_per_doc,
        max_qa_pairs=args.max_qa,
        rng_seed=args.seed,
        output_path=args.output,
        city=args.city,
    )

    qa_pairs = generator.run(
        doc_ids=args.doc_ids,
        source_type=args.source_type,
    )

    print(f"\nDone. {len(qa_pairs)} QA pairs written to {args.output}")


def _run_all_layers(args) -> None:
    """Iterate every config in data/3 country cofog/ and run the generator
    once per derived corpus DB."""
    cfg_dir = Path("data/3 country cofog")
    if not cfg_dir.exists():
        print(f"Error: {cfg_dir} not found.")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = sorted(cfg_dir.glob("*.json"))
    if not configs:
        print(f"Error: no .json files in {cfg_dir}")
        sys.exit(1)

    print(f"Found {len(configs)} unified-shape config(s); running generator on each.")

    for cfg in configs:
        config = json.loads(cfg.read_text(encoding="utf-8"))
        country = config.get("country", "").strip().lower().replace(" ", "_")
        layer = config.get("layer", "").strip().lower().replace(" ", "_")
        if not country or not layer:
            print(f"  Skipping {cfg.name}: missing country/layer field")
            continue

        db_path = f"data/{country}_{layer}_corpus.db"
        out_path = out_dir / f"{country}_{layer}_qa_pairs.xlsx"

        if not Path(db_path).exists():
            print(f"  Skipping {cfg.name}: corpus DB {db_path} not found "
                  f"(re-ingest first?)")
            continue

        store = CorpusStore(db_path)
        counts = store.count()
        if counts["chunks"] == 0:
            print(f"  Skipping {db_path}: corpus is empty")
            continue

        print(f"\n=== {country}/{layer} ===")
        print(f"  DB: {db_path} ({counts['documents']} docs, {counts['chunks']} chunks)")
        print(f"  Output: {out_path}")

        generator = QAGenerator(
            store=store,
            prompts_dir=args.prompts_dir,
            mc_num_options=args.mc_options,
            min_qa_per_doc=args.min_qa_per_doc,
            max_qa_per_doc=args.max_qa_per_doc,
            max_qa_pairs=args.max_qa,
            rng_seed=args.seed,
            output_path=str(out_path),
            city=args.city,
        )

        qa_pairs = generator.run(
            doc_ids=args.doc_ids,
            source_type=args.source_type,
        )

        print(f"  Done. {len(qa_pairs)} QA pairs written to {out_path}")


if __name__ == "__main__":
    main()
