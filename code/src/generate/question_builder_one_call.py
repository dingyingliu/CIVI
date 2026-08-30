"""One-call multiple-choice question builder with inline validation.

Sends a full page directly to the LLM in a single call to generate
multiple MC QA pairs, then validates each pair inline. Validation
requires that option count match ``mc_num_options``, that labels are
unique, and that at least one option is marked correct. The LLM call
is retried up to ``MAX_VALIDATION_RETRIES`` times on failure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.generate.schemas import FULL_PAGE_QA_SCHEMA
from src.llm.openrouter_client import OpenRouterClient
from src.llm.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)

MAX_VALIDATION_RETRIES = 2


@dataclass
class QAPair:
    """A generated multiple-choice question-answer pair.

    Attributes:
        question_id: Unique identifier combining the institution short
            name (from the 3-country COFOG config) and an auto-incrementing
            number (e.g. ``"australian_government_q001"``,
            ``"government_of_canada_q012"``).  Assigned by the generator
            after all pairs are collected.
        question: The question text. Ends with "(Select all that apply)".
        answer: Comma-separated correct labels like ``"A, C"``.
        citation: Source citation metadata.
        options: List of ``{label, text, is_correct}`` dicts.
        quotes: Verbatim quotes supporting the answer.
        model_used: Which LLM generated this pair.
        category: Content category from the COFOG ingest config
            (e.g. ``"Public Order and Safety"``, ``"Health"``).
    """

    question: str
    answer: str
    citation: dict = field(default_factory=dict)
    options: list[dict] | None = None
    quotes: list[str] = field(default_factory=list)
    model_used: str = ""
    category: str = ""
    question_id: str = ""


class OneCallBuilder:
    """Build multiple-choice QA pairs from a full page in a single LLM call.

    ``build_from_page`` sends the full page text and generates multiple
    MC QA pairs in one call.

    Inline validation checks MC option count, label uniqueness, and that
    at least one option is marked correct.  If validation fails, the LLM
    call is retried up to ``MAX_VALIDATION_RETRIES`` times before giving
    up.

    Attributes:
        client: OpenRouter API client.
        prompt_loader: YAML prompt loader.
        mc_num_options: Number of options for MC questions.
    """

    def __init__(
        self,
        client: OpenRouterClient,
        prompt_loader: PromptLoader,
        mc_num_options: int = 8,
    ) -> None:
        self.client = client
        self.prompt_loader = prompt_loader
        self.mc_num_options = max(7, min(10, mc_num_options))

    def build_from_page(
        self,
        page_text: str,
        model: str,
        citation: dict | None = None,
        min_questions: int = 3,
        max_questions: int = 10,
        institution: str = "",
        country: str = "",
        layer: str = "",
    ) -> list[QAPair]:
        """Build multiple MC QA pairs from full page text in a single LLM call.

        The LLM decides how many questions to generate within the
        ``[min_questions, max_questions]`` range based on content density.

        Args:
            page_text: Full text of the source page.
            model: OpenRouter model ID.
            citation: Source citation metadata.
            min_questions: Minimum questions to request.
            max_questions: Maximum questions to request.

        Returns:
            List of validated ``QAPair`` objects (may be shorter than
            requested if some questions fail validation).
        """
        if not page_text or not page_text.strip():
            return []

        # Build a structured header line for institution context, mirroring
        # the cold-eval prompt's prepend pattern. Empty when institution
        # isn't available so we don't render an ugly blank
        # "Institution: (, level)" header.
        if institution:
            institution_header = (
                f"Institution: {institution} ({country}, {layer} level)\n\n"
            )
        else:
            institution_header = ""

        prompt_kwargs = {
            "page_text": page_text,
            "min_questions": str(min_questions),
            "max_questions": str(max_questions),
            "num_options": str(self.mc_num_options),
            "max_label": chr(64 + self.mc_num_options),
            "institution_header": institution_header,
        }

        messages = self.prompt_loader.format_messages("qa_full_page", **prompt_kwargs)

        for attempt in range(1 + MAX_VALIDATION_RETRIES):
            try:
                parsed, record = self.client.chat(
                    model=model,
                    messages=messages,
                    json_schema=FULL_PAGE_QA_SCHEMA,
                    temperature=0,
                    stage="qa_full_page",
                )

                raw_questions = parsed.get("questions", [])
                logger.info(
                    "  Full-page call returned %d questions (%.1fs)",
                    len(raw_questions), record.duration_s,
                )

                qa_pairs: list[QAPair] = []
                for idx, q in enumerate(raw_questions):
                    qa = self._parse_full_page_question(q, model, citation)
                    issues = self._validate(qa)
                    if issues:
                        logger.warning(
                            "  Question %d validation failed: %s",
                            idx + 1, "; ".join(issues),
                        )
                        continue
                    qa_pairs.append(qa)

                if qa_pairs:
                    return qa_pairs

                logger.warning(
                    "  All questions failed validation (attempt %d/%d)",
                    attempt + 1, 1 + MAX_VALIDATION_RETRIES,
                )

            except Exception:
                logger.exception(
                    "  Full-page question build failed (attempt %d/%d)",
                    attempt + 1, 1 + MAX_VALIDATION_RETRIES,
                )

        logger.warning(
            "  Full-page build dropped after %d attempts",
            1 + MAX_VALIDATION_RETRIES,
        )
        return []

    def _parse_full_page_question(
        self,
        parsed: dict,
        model: str,
        citation: dict | None,
    ) -> QAPair:
        """Convert a single MC question from full-page LLM JSON into a QAPair."""
        options = parsed.get("options", [])
        correct_labels = [
            o["label"] for o in options if o.get("is_correct")
        ]
        answer = ", ".join(sorted(correct_labels))
        return QAPair(
            question=parsed["question"],
            answer=answer,
            citation=citation or {},
            options=options,
            quotes=parsed.get("quotes", []),
            model_used=model,
        )

    def _validate(self, qa: QAPair) -> list[str]:
        """Run inline validation checks on an MC pair.

        Returns a list of issue descriptions (empty if valid).
        """
        issues: list[str] = []

        if not qa.question or not qa.question.strip():
            issues.append("empty question")

        if qa.options is not None:
            # Option count must match requested
            if len(qa.options) != self.mc_num_options:
                issues.append(
                    f"expected {self.mc_num_options} options, got {len(qa.options)}"
                )
            # Labels must be unique
            labels = [o["label"] for o in qa.options]
            if len(labels) != len(set(labels)):
                issues.append("duplicate option labels")
            # At least one correct answer
            if not any(o.get("is_correct") for o in qa.options):
                issues.append("no correct option marked")

        return issues
