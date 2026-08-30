# CIVI: A Framework for Diagnosing Search Agent Failures in Civic Information

CIVI is a framework for evaluating search agents and diagnosing their failures on authoritative civic information. It combines a 4,097-question evaluation set with ARISE, a diagnostic that attributes incorrect responses to search bypass, retrieval failure, grounding failure, or comprehension failure.

The public dataset is available on [Hugging Face](https://huggingface.co/datasets/dingyingliu/CIVI).

## Scope

CIVI covers English-language civic information from Australia, Canada, and the United States at federal, state or provincial, and municipal levels. Questions span four categories from the UN Classification of the Functions of Government (COFOG) and were constructed from a curated source index of 576 authoritative public-sector pages.

This repository provides code for:

- running the agentic-search evaluation through OpenRouter and Exa
- applying the ARISE failure diagnostic
- generating and validating question-answer pairs
- managing the underlying source corpus.

## Repository structure

| Path | Contents |
| --- | --- |
| `code/src/benchmark/` | Search-agent evaluation and answer extraction. |
| `code/src/diagnostic/` | ARISE diagnostics and source loading. |
| `code/src/generate/` | Question generation, validation, and export. |
| `code/src/ingest/` | Page fetching, text extraction, chunking, and corpus storage. |
| `code/scripts/` | Command-line entry points. |
| `code/prompts/` | Generation prompts and authoritative-domain configuration. |

## Installation

CIVI requires Python 3.11 or later and uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
cd code
uv sync
uv run playwright install chromium   # required only for page ingestion
cp .env.example .env
```

On PowerShell, use `Copy-Item .env.example .env` instead of `cp`. Add the following credentials to `code/.env`:

```text
OPENROUTER_API_KEY=...
EXA_API_KEY=...
```

Running the evaluation invokes external APIs and may incur usage charges.

## Quickstart

The benchmark runner accepts a single-sheet Excel file whose columns follow the released schema (`question_id`, `question`, `answer`, `category`, `institution`, `city`, `source_url`, `page_title`). Download the questions from [Hugging Face](https://huggingface.co/datasets/dingyingliu/CIVI) or generate your own as described below.

If your questions are spread across multiple sheets of one workbook, flatten them first:

```bash
uv run python scripts/concatenate_jurisdiction_sheets.py \
  --input my_questions.xlsx \
  --output output/benchmark_inputs/my_questions_flat.xlsx
```

Run a ten-question evaluation with one model:

```bash
uv run python scripts/run_benchmark.py \
  --input output/benchmark_inputs/my_questions_flat.xlsx \
  --output output/benchmark_results/my_questions_results.xlsx \
  --cache output/benchmark_results/my_questions_cache.json \
  --institution "Australian Government" \
  --only-models gpt-5.4 \
  --limit 10 \
  --workers 2
```

The output contains a summary sheet and row-level records of responses, search behavior, retrieved and cited URLs, extracted answers, and correctness. Run any script with `--help` to view its available options.

## ARISE diagnostics

ARISE analyzes incorrect benchmark responses using search traces and source-injection ablation. Exact reruns require benchmark result workbooks and compatible source-text SQLite corpora under `code/data/`, these corpora are not distributed, but can be rebuilt with the ingestion step below.

```bash
uv run python scripts/run_diagnostic_per_model.py \
  --model qwen-3.6-plus \
  --n 100 \
  --workers 10 \
  --output output/diagnostic_results/qwen_diag_n100.xlsx
```

## Source ingestion

Question generation and ARISE diagnostics both run against a source corpus: page text fetched, extracted, and stored in SQLite. To build one — including for a different set of pages, such as a public authority's own service pages — start from a CSV with `city` and `url` columns, where `city` is any short label used to group pages and name the corpus:

```bash
uv run python scripts/reingest_with_quality.py \
  --urls-csv my_pages.csv \
  --chunk-size 2000 \
  --delay 0.4
```

Fetching uses Playwright with a static-HTTP fallback, extraction selects among several strategies per page. Corpora are written to `data/{city}_corpus.db`. `scripts/audit_extraction.py` reports extraction quality, `scripts/export_pages.py` dumps a corpus to text, and `scripts/manual_add_page.py` adds a page by hand when a site cannot be fetched. The corpora used in the paper are not distributed, and neither are the per-city configs that `--all-cities` expects, so `--urls-csv` is the supported entry point for new page sets.

## Question generation

Generation reads documents from a corpus built as above:

```bash
uv run python scripts/run_generator.py \
  --db data/my_pages_corpus.db \
  --output output/qa_pairs/my_pages_qa_pairs.xlsx \
  --max-qa 120
```

The pipeline generates questions from one source page per model call, performs structural checks, and verifies supporting quotations against the source text. Questions in the released dataset were additionally reviewed by two domain experts, and questions flagged by either reviewer were removed, the automatic checks alone do not reproduce that step.

Ingestion, generation, benchmarking, and diagnosis together form the full pipeline: pages in, attributed failures out.

## Reproducibility and intended use

- Live-search results may vary as search indexes, source pages, models, and APIs change.
- Evaluation and generation use temperature 0, benchmark caches and outputs are stored under `code/output/`.
- The released source-text indexes do not include the full page contents required for exact source-injection reruns.
- CIVI is intended for research evaluation, not for legal, medical, tax, benefits, or other professional advice.

## License

The software is released under the MIT License (`code/LICENSE`). The dataset, distributed separately on Hugging Face, is released under CC BY 4.0 (`code/LICENSE-DATA`).

## Citation

```bibtex
@inproceedings{liu2026civi,
  title     = {{CIVI}: A Framework for Diagnosing Search Agent Failures in Civic Information},
  author    = {Liu, Dingying and Zhong, Yunshun and Zhang, Wentao and Li, Yiyuan},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
  address   = {Budapest, Hungary},
  publisher = {Association for Computational Linguistics}
}
```
