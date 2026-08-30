"""QA Generator — top-level orchestrator.

Chains together: FullPageBuilding → Export.
Uses multiple LLMs iteratively by rotating through a configured model list
across documents.

Each document is guaranteed to produce between ``min_qa_per_doc`` and
``max_qa_per_doc`` QA pairs (default 3–10).  The full page text is sent
to the LLM in a single call, and the LLM decides how many questions to
generate based on content density.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.generate.exporter import QAExporter
from src.generate.question_builder_one_call import (
    OneCallBuilder,
    QAPair,
)
from src.generate.quote_verifier import QuoteVerifier
from src.ingest.store import CorpusStore
from src.llm.openrouter_client import OpenRouterClient
from src.llm.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)

# Default model rotation list — users can override
DEFAULT_MODELS: dict[str, str] = {
    "claude-4.6-opus": "anthropic/claude-opus-4.6",
}


# ---------- Checkpoint (de)serialization for resumable runs ---------- #
# Public dataclass fields go in straight; the private ``_*`` attrs the
# generator pins on each pair (institution, cell, label, short_name) need
# explicit handling because dataclasses.asdict() ignores them.

_QA_PRIVATE_ATTRS = ("_institution_short_name", "_cell", "_institution", "_label")


def _qa_to_dict(qa: QAPair) -> dict:
    return {
        "question": qa.question,
        "answer": qa.answer,
        "citation": qa.citation,
        "options": qa.options,
        "quotes": qa.quotes,
        "model_used": qa.model_used,
        "category": qa.category,
        "question_id": qa.question_id,
        **{k: getattr(qa, k, "") for k in _QA_PRIVATE_ATTRS},
    }


def _qa_from_dict(d: dict) -> QAPair:
    qa = QAPair(
        question=d["question"],
        answer=d["answer"],
        citation=d.get("citation", {}),
        options=d.get("options"),
        quotes=d.get("quotes", []),
        model_used=d.get("model_used", ""),
        category=d.get("category", ""),
        question_id=d.get("question_id", ""),
    )
    for k in _QA_PRIVATE_ATTRS:
        if k in d:
            setattr(qa, k, d[k])
    return qa


class QAGenerator:
    """Orchestrate the full multiple-choice QA generation pipeline.

    Iterates over documents in the corpus, retrieves full page text,
    builds MC questions via the one-call builder in a single LLM
    call per document, and exports to Excel.

    LLMs are rotated across documents so that different models
    contribute to the dataset, increasing diversity and reducing
    single-model bias.
    """

    def __init__(
        self,
        store: CorpusStore,
        api_key: str | None = None,
        models: dict[str, str] | None = None,
        prompts_dir: str | Path = "prompts",
        mc_num_options: int = 8,
        min_qa_per_doc: int = 3,
        max_qa_per_doc: int = 10,
        max_qa_pairs: int | None = None,
        rng_seed: int | None = None,
        output_path: str | Path = "output/qa_pairs.xlsx",
        city: str = "",
    ) -> None:
        """Initialise the QA generator.

        Args:
            store: Corpus store with ingested documents.
            api_key: OpenRouter API key (or set ``OPENROUTER_API_KEY``).
            models: Dict of ``{short_name: openrouter_model_id}``.
            prompts_dir: Directory containing YAML prompt files.
            mc_num_options: Number of options per MC question (7–10).
            min_qa_per_doc: Minimum QA pairs per document.
            max_qa_per_doc: Maximum QA pairs per document.
            max_qa_pairs: Global stop limit. ``None`` means no limit.
            rng_seed: Seed for reproducibility.
            output_path: Path for the output Excel file.
        """
        self.store = store
        self.models = models or DEFAULT_MODELS
        self.min_qa_per_doc = min_qa_per_doc
        self.max_qa_per_doc = max_qa_per_doc
        self.max_qa_pairs = max_qa_pairs
        self.rng_seed = rng_seed
        self.output_path = Path(output_path)

        # Shared components
        self.client = OpenRouterClient(api_key=api_key)
        self.prompt_loader = PromptLoader(prompts_dir)

        # Agents
        self.builder = OneCallBuilder(
            self.client, self.prompt_loader,
            mc_num_options=mc_num_options,
        )
        self.verifier = QuoteVerifier()
        self.exporter = QAExporter(output_path, city=city)

        # Model rotation
        self._model_ids = list(self.models.values())
        self._model_index = 0

    def run(
        self,
        doc_ids: list[str] | None = None,
        source_type: str | None = None,
    ) -> list[QAPair]:
        """Run the full QA generation pipeline.

        For each document: retrieve full text → build questions in one
        LLM call → collect.

        Args:
            doc_ids: Specific document IDs to process. ``None`` processes
                all documents (optionally filtered by ``source_type``).
            source_type: Filter by ``"web"`` or ``"pdf"``.

        Returns:
            List of all generated ``QAPair`` objects.
        """
        # Select documents
        if doc_ids:
            documents = [
                d for d in self.store.list_documents()
                if d["doc_id"] in doc_ids
            ]
        elif source_type:
            documents = [
                d for d in self.store.list_documents()
                if d["source_type"] == source_type
            ]
        else:
            documents = self.store.list_documents()

        logger.info("Processing %d documents with %d models", len(documents), len(self._model_ids))

        # Resumable checkpoint: each line is {"doc_id": str, "pairs": [<qa>, ...]}.
        # Loaded on startup so a crashed/interrupted run can resume without
        # re-generating already-completed docs. Cleaned up after successful export.
        checkpoint_path = self.output_path.with_suffix(".checkpoint.jsonl")
        completed_doc_ids: set[str] = set()
        all_qa_pairs: list[QAPair] = []

        if checkpoint_path.exists():
            for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("  bad checkpoint line, skipping: %s", line[:80])
                    continue
                completed_doc_ids.add(entry["doc_id"])
                for qa_dict in entry.get("pairs", []):
                    all_qa_pairs.append(_qa_from_dict(qa_dict))
            logger.info(
                "Resuming from checkpoint: %d docs (%d QA pairs) already completed",
                len(completed_doc_ids), len(all_qa_pairs),
            )

        for doc_meta in documents:
            if self.max_qa_pairs and len(all_qa_pairs) >= self.max_qa_pairs:
                logger.info("Reached target of %d QA pairs, stopping early", self.max_qa_pairs)
                break

            doc_id = doc_meta["doc_id"]
            title = doc_meta.get("title", doc_id)

            if doc_id in completed_doc_ids:
                logger.info("=== Skipping (in checkpoint): %s ===", title)
                continue

            logger.info("\n=== Document: %s ===", title)

            page_text = self.store.get_full_text(doc_id)

            if not page_text:
                logger.warning("  No full text found, skipping")
                continue

            citation = {
                "doc_id": doc_id,
                "source_uri": doc_meta.get("source_uri", ""),
                "title": title,
                "source_type": doc_meta.get("source_type", ""),
            }

            # Extract category from extra_json (populated by JSON config ingest)
            extra = {}
            if doc_meta.get("extra_json"):
                try:
                    extra = json.loads(doc_meta["extra_json"])
                except (json.JSONDecodeError, TypeError):
                    pass
            category = extra.get("category", "")
            institution_short_name = extra.get("institution_short_name", "")
            cell = extra.get("cell", "")
            institution = extra.get("institution", "")
            label = extra.get("label", "")
            country = extra.get("country", "")
            layer = extra.get("layer", "")

            # Determine per-doc limits respecting global cap
            max_for_doc = self.max_qa_per_doc
            if self.max_qa_pairs:
                remaining = self.max_qa_pairs - len(all_qa_pairs)
                max_for_doc = min(max_for_doc, remaining)

            model = self._next_model()

            logger.info(
                "  Building %d–%d questions from full page text (model: %s)",
                self.min_qa_per_doc, max_for_doc, model,
            )

            doc_qa_pairs = self.builder.build_from_page(
                page_text=page_text,
                model=model,
                citation=citation,
                min_questions=self.min_qa_per_doc,
                max_questions=max_for_doc,
                institution=institution,
                country=country,
                layer=layer,
            )

            verified_pairs = self.verifier.verify(doc_qa_pairs, page_text, doc_id)

            for qa in verified_pairs:
                qa.category = category
                qa._institution_short_name = institution_short_name
                qa._cell = cell
                qa._institution = institution
                qa._label = label

            all_qa_pairs.extend(verified_pairs)

            # Append to checkpoint immediately so a crash here loses at most
            # the current doc's work, not everything before it.
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "doc_id": doc_id,
                    "pairs": [_qa_to_dict(qa) for qa in verified_pairs],
                }) + "\n")

            logger.info(
                "  Document done: %d generated, %d after quote verification",
                len(doc_qa_pairs), len(verified_pairs),
            )

        # Assign question IDs: {institution_short_name}_q001, q002, …
        # Group counters by institution short name so each starts at 1.
        id_counters: dict[str, int] = {}
        for qa in all_qa_pairs:
            prefix = getattr(qa, "_institution_short_name", "") or "qa"
            id_counters.setdefault(prefix, 0)
            id_counters[prefix] += 1
            qa.question_id = f"{prefix}_q{id_counters[prefix]:03d}"

        # Export
        if all_qa_pairs:
            self.exporter.export(all_qa_pairs)
            # Successful export — drop the checkpoint so the next run starts fresh.
            try:
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
            except OSError as exc:
                logger.warning("Could not remove checkpoint %s: %s", checkpoint_path, exc)

        logger.info(
            "\nPipeline complete: %d QA pairs from %d documents",
            len(all_qa_pairs), len(documents),
        )
        return all_qa_pairs

    def _next_model(self) -> str:
        """Return the next model ID in the rotation."""
        model = self._model_ids[self._model_index % len(self._model_ids)]
        self._model_index += 1
        return model
