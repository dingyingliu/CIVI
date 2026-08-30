"""Run the text-based factual QA benchmark evaluation.

Usage:
    python scripts/run_benchmark.py --input output/qa_pairs/qa_pairs.xlsx
    python scripts/run_benchmark.py --input output/qa_pairs/qa_pairs.xlsx --limit 10
    python scripts/run_benchmark.py --input output/qa_pairs/qa_pairs.xlsx --output output/benchmark_results/bench.xlsx
"""

import argparse
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

from src.benchmark.benchmark_text_qa import BenchmarkConfig, TextQABenchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark LLMs on factual QA pairs",
    )
    parser.add_argument(
        "--input", type=str, default="output/qa_pairs/qa_pairs.xlsx",
        help="Path to QA pairs Excel file (default: output/qa_pairs/qa_pairs.xlsx)",
    )
    parser.add_argument(
        "--output", type=str, default="output/benchmark_results/benchmark_results.xlsx",
        help="Path for benchmark results Excel (default: output/benchmark_results/benchmark_results.xlsx)",
    )
    parser.add_argument(
        "--cache", type=str, default="output/benchmark_results/benchmark_cache.json",
        help="Path for response cache (default: output/benchmark_results/benchmark_cache.json)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit to first N rows (for testing)",
    )
    parser.add_argument(
        "--workers", type=int, default=5,
        help="Max parallel API workers per model (default: 5)",
    )
    parser.add_argument(
        "--institution", type=str, default="",
        help="Institution name for context (uniform across all rows; "
             "use this for federal-tier runs).",
    )
    parser.add_argument(
        "--country-suffix", type=str, default="",
        help="Country to append to per-row state/province name when "
             "--institution is not set. Used at the state tier to "
             "disambiguate state names: e.g. 'Australia' makes the "
             "context 'This question is about Victoria, Australia.'",
    )
    parser.add_argument(
        "--institution-column", type=str, default="",
        help="Per-row institution column. When set, the value of this "
             "column from each row is used directly as the prompt context "
             "(no normalization, no country suffix). Used at the municipal "
             "tier where the 'institution' field holds 'City of Brisbane', "
             "'City of Toronto' etc. Takes precedence over --country-suffix.",
    )
    parser.add_argument(
        "--user-location", type=str, default="",
        help="User location for search context (e.g. 'Sydney, Australia')",
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated model providers to run (e.g. openai,google). Default: all",
    )
    parser.add_argument(
        "--only-models", type=str, default=None,
        help="Comma-separated exact model short names to run (e.g. "
             "gpt-5.4,claude-sonnet-4.6,gemini-3.1-pro). Overrides --models.",
    )
    parser.add_argument(
        "--baseline", action="store_true", default=False,
        help="Disable all per-model interventions (V3 prompt, effort:low, "
             "empty-retry). Reverts to stock pipeline. Cache hits remain "
             "compatible with prior baseline runs.",
    )
    args = parser.parse_args()

    if args.institution and args.country_suffix:
        logging.getLogger().warning(
            "Both --institution and --country-suffix were set. "
            "--institution takes precedence; --country-suffix will be ignored."
        )
    if args.institution and args.institution_column:
        logging.getLogger().warning(
            "Both --institution and --institution-column were set. "
            "--institution takes precedence; --institution-column will be ignored."
        )
    if args.institution_column and args.country_suffix:
        logging.getLogger().warning(
            "Both --institution-column and --country-suffix were set. "
            "--institution-column takes precedence; --country-suffix will be ignored."
        )

    # Filter eval models by exact short name (--only-models) or provider
    # prefix (--models). --only-models takes precedence.
    eval_models = None
    if args.only_models:
        wanted = [s.strip() for s in args.only_models.split(",") if s.strip()]
        from src.benchmark.benchmark_text_qa import DEFAULT_EVAL_MODELS
        eval_models = {
            name: model_id
            for name, model_id in DEFAULT_EVAL_MODELS.items()
            if name in wanted
        }
        missing = [w for w in wanted if w not in eval_models]
        if missing:
            parser.error(
                f"Unknown model short names: {', '.join(missing)}. "
                f"Available: {', '.join(DEFAULT_EVAL_MODELS.keys())}"
            )
    elif args.models:
        providers = [p.strip().lower() for p in args.models.split(",")]
        from src.benchmark.benchmark_text_qa import DEFAULT_EVAL_MODELS
        eval_models = {
            name: model_id
            for name, model_id in DEFAULT_EVAL_MODELS.items()
            if any(model_id.lower().startswith(p + "/") for p in providers)
        }
        if not eval_models:
            parser.error(
                f"No models matched providers: {', '.join(providers)}. "
                f"Available: {', '.join(mid.split('/')[0] for mid in DEFAULT_EVAL_MODELS.values())}"
            )

    config = BenchmarkConfig(
        input_path=Path(args.input),
        output_path=Path(args.output),
        cache_path=Path(args.cache),
        limit_rows=args.limit,
        max_workers=args.workers,
        institution_name=args.institution,
        country_suffix=args.country_suffix,
        institution_column=args.institution_column,
        user_location=args.user_location,
        interventions_enabled=not args.baseline,
        **({"eval_models": eval_models} if eval_models is not None else {}),
    )

    benchmark = TextQABenchmark(config)
    try:
        output_path = benchmark.run()
        print(f"\nBenchmark complete. Results saved to: {output_path}")
    finally:
        benchmark.close()


if __name__ == "__main__":
    main()
