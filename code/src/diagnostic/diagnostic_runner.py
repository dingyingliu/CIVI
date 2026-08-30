"""Stage 2 source-injection diagnostic.

Re-runs each failed MC question with the authoritative source page text
injected into the prompt (web search disabled), then classifies the
failure into one of:

    search_bypass     model didn't search at all in Stage 1
    retrieval         model searched but never visited the source URL
    grounding         model visited the source URL but answered wrong anyway
    comprehension     model still wrong even with source text injected
    skipped_no_source no usable source text could be located
"""

from __future__ import annotations

import logging
import random
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from openpyxl.styles import Font

from src.benchmark.benchmark_text_qa import (
    DEFAULT_EVAL_MODELS,
    TextQABenchmark,
    model_extras,
)
from src.diagnostic.source_loader import SourceLoader, normalize_url
from src.llm.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "Use ONLY the source text below to answer. Do not use any outside knowledge.\n\n"
    "Source page:\n"
    "{source_text}\n\n"
    "Question:\n"
    "{question}\n"
)

FAILURE_MODES: list[str] = [
    "search_bypass",
    "retrieval",
    "grounding",
    "comprehension",
    "skipped_no_source",
]


@dataclass
class DiagnosticConfig:
    """Configuration for one diagnostic run."""

    input_excel: Path
    output_excel: Path
    samples_per_category: int = 38
    max_workers: int = 4
    model_filter: list[str] | None = None
    city_override: str = ""
    rng_seed: int = 42
    data_dir: Path = Path("data")
    skip_log_path: Path = Path("output/diagnostic_skipped.log")


@dataclass
class DiagnosticTask:
    """One (failed question, model) pair to re-run with source injection."""

    question_id: str
    row_idx: int
    model_name: str
    model_id: str
    category: str
    city: str
    question: str
    answer: str
    source_url: str
    original_response: str
    stage1_searched: bool
    stage1_cited_urls: list[str] = field(default_factory=list)


@dataclass
class DiagnosticResult:
    """Outcome of one diagnostic call (or skip)."""

    task: DiagnosticTask
    diagnostic_response: str = ""
    diagnostic_correct: bool = False
    failure_mode: str = ""


class DiagnosticRunner:
    """Orchestrate the source-injection diagnostic for MC failures."""

    def __init__(
        self,
        config: DiagnosticConfig,
        client: OpenRouterClient,
        loader: SourceLoader,
    ) -> None:
        """Initialise the runner.

        Args:
            config: Run configuration.
            client: Shared OpenRouter client (reuses Stage 1 retry/backoff).
            loader: Three-tier source-text lookup.
        """
        self.config = config
        self.client = client
        self.loader = loader
        self.config.skip_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._skip_log_lock = threading.Lock()

    # ── public ────────────────────────────────────────────────────────────

    def run(self) -> Path:
        """Execute the full diagnostic pipeline and write the output Excel."""
        df = self._read_main_run_excel()
        tasks = self._collect_mc_failures(df)
        logger.info("Collected %d failed (MC × model) pairs", len(tasks))

        if self.config.model_filter:
            wanted = set(self.config.model_filter)
            tasks = [t for t in tasks if t.model_name in wanted]
            logger.info(
                "After model filter %s: %d tasks", sorted(wanted), len(tasks)
            )

        sampled = self._stratified_sample(tasks)
        logger.info(
            "After stratified sampling (%d per (model, category)): %d tasks",
            self.config.samples_per_category, len(sampled),
        )

        results = self._execute_concurrent(sampled)
        return self._write_output(results)

    # ── ingest ────────────────────────────────────────────────────────────

    def _read_main_run_excel(self) -> pd.DataFrame:
        return pd.read_excel(self.config.input_excel, sheet_name="Results")

    def _collect_mc_failures(self, df: pd.DataFrame) -> list[DiagnosticTask]:
        """Enumerate every (MC row, model) where ``correct_{m} == 0``."""
        model_cols = [
            c[len("correct_"):] for c in df.columns if c.startswith("correct_")
        ]
        known_models: dict[str, str] = {}
        for m in model_cols:
            if m in DEFAULT_EVAL_MODELS:
                known_models[m] = DEFAULT_EVAL_MODELS[m]
            else:
                logger.warning(
                    "Model '%s' not in DEFAULT_EVAL_MODELS — skipping", m
                )

        tasks: list[DiagnosticTask] = []
        for row_idx, row in df.iterrows():
            for m, slug in known_models.items():
                if not self._is_failure(row.get(f"correct_{m}")):
                    continue
                city_val = (
                    self._coerce_str(row.get("city")) or self.config.city_override
                )
                tasks.append(DiagnosticTask(
                    question_id=self._coerce_str(row.get("question_id")),
                    row_idx=int(row_idx),
                    model_name=m,
                    model_id=slug,
                    category=self._coerce_str(row.get("category")),
                    city=city_val,
                    question=self._coerce_str(row.get("question_text")),
                    answer=self._coerce_str(row.get("gold_answer")),
                    source_url=self._coerce_str(row.get("source_url")),
                    original_response=self._coerce_str(row.get(f"response_{m}")),
                    stage1_searched=self._coerce_bool(row.get(f"searched_{m}")),
                    stage1_cited_urls=self._parse_cited_urls(
                        row.get(f"cited_urls_{m}")
                    ),
                ))
        return tasks

    @staticmethod
    def _is_failure(val) -> bool:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return False
        try:
            return int(val) == 0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _coerce_str(val) -> str:
        if val is None:
            return ""
        if isinstance(val, float) and pd.isna(val):
            return ""
        return str(val).strip()

    @staticmethod
    def _coerce_bool(val) -> bool:
        if val is None:
            return False
        if isinstance(val, bool):
            return val
        if isinstance(val, float):
            if pd.isna(val):
                return False
            return bool(val)
        if isinstance(val, int):
            return bool(val)
        return str(val).strip().lower() in ("true", "1", "yes")

    @staticmethod
    def _parse_cited_urls(raw) -> list[str]:
        if not isinstance(raw, str):
            return []
        s = raw.strip()
        if not s:
            return []
        return [u for u in s.split(" | ") if u]

    # ── sample ────────────────────────────────────────────────────────────

    def _stratified_sample(
        self, tasks: list[DiagnosticTask],
    ) -> list[DiagnosticTask]:
        """Per (model, category) bucket, sample up to ``samples_per_category``."""
        rng = random.Random(self.config.rng_seed)
        buckets: dict[tuple[str, str], list[DiagnosticTask]] = defaultdict(list)
        for t in tasks:
            buckets[(t.model_name, t.category)].append(t)

        sampled: list[DiagnosticTask] = []
        for group in buckets.values():
            if len(group) <= self.config.samples_per_category:
                sampled.extend(group)
            else:
                sampled.extend(
                    rng.sample(group, self.config.samples_per_category)
                )
        return sampled

    # ── execute ───────────────────────────────────────────────────────────

    def _execute_concurrent(
        self, tasks: list[DiagnosticTask],
    ) -> list[DiagnosticResult]:
        results: list[DiagnosticResult] = []
        if not tasks:
            return results
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            futures = {pool.submit(self._run_one, t): t for t in tasks}
            done = 0
            for fut in as_completed(futures):
                results.append(fut.result())
                done += 1
                if done % 25 == 0 or done == len(tasks):
                    logger.info(
                        "  diagnostic progress: %d / %d", done, len(tasks)
                    )
        return results

    def _run_one(self, task: DiagnosticTask) -> DiagnosticResult:
        if not task.source_url:
            self._log_skip(task, "no_source_url_in_excel")
            return DiagnosticResult(task=task, failure_mode="skipped_no_source")

        source_text = self.loader.lookup(task.source_url, task.city)
        if not source_text:
            self._log_skip(task, "not_found_in_any_db")
            return DiagnosticResult(task=task, failure_mode="skipped_no_source")

        prompt = PROMPT_TEMPLATE.format(
            source_text=source_text, question=task.question,
        )
        response_text = self._call_model(task.model_id, prompt)
        diag_correct = self._score_mc(task.question, task.answer, response_text)

        return DiagnosticResult(
            task=task,
            diagnostic_response=response_text,
            diagnostic_correct=diag_correct,
            failure_mode=self._classify(diag_correct, task),
        )

    def _call_model(self, model_id: str, prompt: str) -> str:
        extra_body, max_tokens = model_extras(model_id, "MC")
        result = self.client.chat_eval(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
            stage="diagnostic",
            tools=None,
            extra_body=extra_body,
        )
        return result.get("text", "")

    @staticmethod
    def _score_mc(question: str, answer: str, response: str) -> bool:
        valid_labels = TextQABenchmark._extract_valid_labels(question)
        gold = set(re.findall(r"[A-Z]", str(answer).upper())) & valid_labels
        pred = TextQABenchmark._parse_mc_answer(response, valid_labels)
        return pred == gold

    # ── classify ──────────────────────────────────────────────────────────

    def _classify(self, diag_correct: bool, task: DiagnosticTask) -> str:
        if not diag_correct:
            return "comprehension"
        if not task.stage1_searched:
            return "search_bypass"
        target = normalize_url(task.source_url)
        cited = {normalize_url(u) for u in task.stage1_cited_urls}
        if target and target in cited:
            return "grounding"
        return "retrieval"

    # ── output ────────────────────────────────────────────────────────────

    def _log_skip(self, task: DiagnosticTask, reason: str) -> None:
        line = (
            f"{task.question_id}\t{task.model_name}\t"
            f"{task.source_url}\t{reason}\n"
        )
        with self._skip_log_lock:
            with self.config.skip_log_path.open("a", encoding="utf-8") as f:
                f.write(line)

    def _write_output(self, results: list[DiagnosticResult]) -> Path:
        self.config.output_excel.parent.mkdir(parents=True, exist_ok=True)

        rows: list[dict] = []
        for r in results:
            t = r.task
            rows.append({
                "question_id":          t.question_id,
                "model":                t.model_name,
                "source_url":           t.source_url,
                "category":             t.category,
                "city":                 t.city,
                "original_response":    t.original_response,
                "original_correct":     0,
                "diagnostic_response":  r.diagnostic_response,
                "diagnostic_correct":   1 if r.diagnostic_correct else 0,
                "stage1_searched":      t.stage1_searched,
                "stage1_cited_urls":    " | ".join(t.stage1_cited_urls),
                "n_cited_urls_stage1":  len(t.stage1_cited_urls),
                "failure_mode":         r.failure_mode,
            })
        results_df = pd.DataFrame(rows)

        summary_rows: list[dict] = []
        for model_name in sorted({r.task.model_name for r in results}):
            counts: dict = {"model": model_name}
            total = 0
            for mode in FAILURE_MODES:
                n = sum(
                    1 for r in results
                    if r.task.model_name == model_name and r.failure_mode == mode
                )
                counts[mode] = n
                total += n
            counts["total"] = total
            summary_rows.append(counts)
        summary_df = pd.DataFrame(summary_rows)

        with pd.ExcelWriter(self.config.output_excel, engine="openpyxl") as writer:
            results_df.to_excel(writer, sheet_name="Results", index=False)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            for sheet_name in ("Results", "Summary"):
                ws = writer.sheets[sheet_name]
                for cell in ws[1]:
                    cell.font = Font(bold=True)

        logger.info("Wrote diagnostic results → %s", self.config.output_excel)
        return self.config.output_excel
