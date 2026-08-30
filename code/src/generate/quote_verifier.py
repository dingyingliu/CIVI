"""Quote verification for generated QA pairs.

Checks that each QAPair's supporting quotes actually appear in the
source page text.  Both quote and source are normalised (lowercased,
whitespace collapsed, currency/punctuation stripped) before an exact
substring check.  Any QAPair with a quote that cannot be found in the
normalised source is rejected.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from src.generate.question_builder_one_call import QAPair

logger = logging.getLogger(__name__)

# Characters stripped during normalisation (currency symbols, common punctuation
# that LLMs frequently add/remove when quoting).
_STRIP_CHARS = re.compile(r"[$€£¥₹%""\"''`\u2018\u2019\u201c\u201d\u2013\u2014—–-]")
_WHITESPACE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation/currency symbols, collapse whitespace."""
    text = text.lower()
    text = _STRIP_CHARS.sub("", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


class QuoteVerifier:
    """Verify that QAPair quotes are grounded in the source text.

    Parameters:
        reject_log_path: File path for the rejection log (JSON lines).
    """

    def __init__(
        self,
        reject_log_path: str | Path = "output/qa_pairs/rejected_quotes.jsonl",
    ) -> None:
        self.reject_log_path = Path(reject_log_path)
        self.reject_log_path.parent.mkdir(parents=True, exist_ok=True)

    def verify(
        self,
        qa_pairs: list[QAPair],
        page_text: str,
        doc_id: str,
    ) -> list[QAPair]:
        """Return only QAPairs whose quotes all appear in the source text.

        Any QAPair with at least one quote that is not a substring of the
        normalised source is rejected and logged.  QAPairs with no quotes
        pass automatically.

        Args:
            qa_pairs: QA pairs to verify.
            page_text: Source page text to match quotes against.
            doc_id: Document identifier (for logging).

        Returns:
            List of QAPairs that passed verification.
        """
        normalised_source = _normalise(page_text)
        passed: list[QAPair] = []
        pass_count = 0
        fail_count = 0

        for qa in qa_pairs:
            if not qa.quotes:
                self._log_rejection_no_quotes(qa, doc_id)
                fail_count += 1
                continue

            failed_quote = False
            for quote in qa.quotes:
                normalised_quote = _normalise(quote)
                if normalised_quote not in normalised_source:
                    self._log_rejection(qa, quote, doc_id)
                    failed_quote = True
                    break

            if failed_quote:
                fail_count += 1
            else:
                passed.append(qa)
                pass_count += 1

        logger.info(
            "  Quote verification [%s]: %d passed, %d rejected",
            doc_id, pass_count, fail_count,
        )
        return passed

    def _log_rejection_no_quotes(
        self,
        qa: QAPair,
        doc_id: str,
    ) -> None:
        """Append a rejection record for a QAPair with no quotes."""
        record = {
            "doc_id": doc_id,
            "question": qa.question,
            "failing_quote": None,
            "reason": "no quotes provided",
        }
        with open(self.reject_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.warning("  REJECTED (no quotes): %s", qa.question[:120])

    def _log_rejection(
        self,
        qa: QAPair,
        quote: str,
        doc_id: str,
    ) -> None:
        """Append a rejection record to the JSONL log."""
        record = {
            "doc_id": doc_id,
            "question": qa.question,
            "failing_quote": quote,
        }
        with open(self.reject_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.warning(
            "  REJECTED quote: %s",
            quote[:120],
        )
