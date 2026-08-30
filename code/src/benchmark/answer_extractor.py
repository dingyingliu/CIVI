"""Gemini 2.5 Flash answer extractor for MC benchmarks (via OpenRouter).

Single Flash call per model response: reads the response, returns the set of
option letters the model committed to as its final answer (or empty list for
refusals, incoherent output, empty input, or API failure).

Routes through OpenRouter (model id ``google/gemini-2.5-flash``) using the
``OPENROUTER_API_KEY`` so the per-account quota is OpenRouter's pool and is
independent of any direct ``GOOGLE_API_KEY`` daily quota.

Configuration:
    Model:           google/gemini-2.5-flash (via OpenRouter)
    Temperature:     0.0
    API key env var: OPENROUTER_API_KEY
    Cache:           output/benchmark_results/extractor_cache.json
    Cache key:       "<model_name>|<qid>|<prompt_version>"

Behavior on edge cases:
    - Empty/whitespace input          → return [] without an API call
    - JSON parse failure              → return []
    - Letters outside valid_labels    → silently filtered out
    - 429/5xx after retries           → return []
    - All cache failures              → start fresh
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Hardcoded configuration ──────────────────────────────────────────────────

EXTRACTOR_MODEL = "google/gemini-2.5-flash"  # OpenRouter model id
EXTRACTOR_PROMPT_VERSION = "v1"
EXTRACTOR_CACHE_PATH = Path("output/benchmark_results/extractor_cache.json")
MAX_RETRIES = 3
BASE_DELAY_S = 4.0
SAVE_EVERY_N_WRITES = 25  # batched cache flushes for throughput


# ── Prompt and few-shot examples ─────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an answer-extraction tool. Your input is (a) the option labels valid
for a multiple-choice question, and (b) a model's free-form response to that
question. Your output is the set of option letters the model committed to as
its final answer.

Rules:

1. Extract ONLY letters in the valid label set. Never invent letters.
2. Extract what the model COMMITTED to, not what it considered. If the
   response walks through options without committing to any, return [].
3. If the response includes a final answer line ("Answer: A, F",
   "Therefore: F", "The correct option is F"), trust that line over earlier
   deliberation.
4. If the model lists per-option verdicts ("A: True, B: False, C: True..."
   or "A. correct ✓, B. incorrect, ..."), extract the letters marked
   True/Yes/Correct/Selected. If there is no final summary, the per-option
   verdicts ARE the commitment.
5. If the model refuses to answer ("I don't have enough information",
   "I cannot determine"), gives off-topic prose, or commits to no letters,
   return [].
6. Never extract letters that appear only inside the option text being
   quoted back. ("Option A says X" alone is not a commitment to A.)
7. Output ONLY the JSON specified by the schema."""


# Each entry is (user_message_text, expected_letters_list).
# Few-shots are sent as prior turns, not embedded in the system prompt.
FEW_SHOTS: list[tuple[str, list[str]]] = [
    # 1. Clean direct answer with explicit final line
    (
        "Valid labels: A,B,C,D,E,F,G,H\n\n"
        "Response:\n---\n"
        "The 2026 interim tax bill in Brampton has three instalments due "
        "February 18, March 18, and April 22, 2026.\n\n"
        "Answer: F\n"
        "---",
        ["F"],
    ),
    # 2. Verdict-by-option WITH explicit final summary
    (
        "Valid labels: A,B,C,D,E,F,G,H\n\n"
        "Response:\n---\n"
        "A. They are based on 50 percent — TRUE per the page\n"
        "B. They are based on 75 percent — FALSE\n"
        "C. Calculated using new assessment — FALSE\n"
        "D. Based on current year's budget — FALSE\n"
        "E. Adjustments recalculated — TRUE per the page\n"
        "F. For accounts that didn't exist — false in this form\n"
        "G. Based on 50 percent of current year — FALSE\n"
        "H. No adjustments — FALSE\n\n"
        "Final answer: A, E\n"
        "---",
        ["A", "E"],
    ),
    # 3. Verdict-by-option WITHOUT explicit final (the hardest case)
    (
        "Valid labels: A,B,C,D,E,F,G,H\n\n"
        "Response:\n---\n"
        "A. Correct — matches the page\n"
        "B. Incorrect\n"
        "C. Correct — directly stated in the source\n"
        "D. Incorrect\n"
        "E. Incorrect — contradicts the source\n"
        "F. Correct\n"
        "G. Incorrect\n"
        "H. Incorrect — irrelevant\n"
        "---",
        ["A", "C", "F"],
    ),
    # 4. Refusal
    (
        "Valid labels: A,B,C,D,E,F,G,H\n\n"
        "Response:\n---\n"
        "I don't have enough information about the 2026 Brampton tax "
        "schedule to determine the answer with confidence. I'd suggest "
        "checking the city's website directly.\n"
        "---",
        [],
    ),
    # 5. Wandering off-topic / no commit
    (
        "Valid labels: A,B,C,D,E,F,G,H\n\n"
        "Response:\n---\n"
        "The question concerns property taxes in a Canadian municipality. "
        "Property taxes are assessed annually based on property value. "
        "Cities use these revenues to fund services like waste collection "
        "and emergency response.\n"
        "---",
        [],
    ),
]


# ── Response schema (Gemini structured-JSON output) ──────────────────────────

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "letters": {
            "type": "array",
            "description": "Letters the model committed to as its final answer; empty list if none.",
            "items": {
                "type": "string",
                "enum": ["A", "B", "C", "D", "E", "F", "G", "H"],
            },
        },
    },
    "required": ["letters"],
    "propertyOrdering": ["letters"],
}


# ── Extractor class ──────────────────────────────────────────────────────────


class GeminiAnswerExtractor:
    """Single-call Gemini Flash extractor for MC answer letters.

    Thread-safe: an internal lock serializes cache writes so multiple
    extraction threads can share one instance.
    """

    def __init__(
        self,
        cache_path: Path = EXTRACTOR_CACHE_PATH,
        prompt_version: str = EXTRACTOR_PROMPT_VERSION,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.prompt_version = prompt_version
        self.cache: dict[str, list[str]] = self._load_cache()
        self._lock = threading.Lock()
        self._writes_since_save = 0
        self._client = None  # lazy-init in _ensure_client

    # ── Cache plumbing ───────────────────────────────────────────────────────

    def _load_cache(self) -> dict[str, list[str]]:
        if self.cache_path.exists():
            try:
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, OSError):
                logger.warning("Extractor cache at %s is unreadable; starting fresh", self.cache_path)
        return {}

    def _save_cache_locked(self) -> None:
        """Caller must hold ``self._lock``."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self.cache_path)
        self._writes_since_save = 0

    def flush_cache(self) -> None:
        """Force a cache save. Useful at end of a run."""
        with self._lock:
            self._save_cache_locked()

    # ── Lazy SDK init ────────────────────────────────────────────────────────

    def _ensure_client(self) -> None:
        # Double-checked locking so concurrent threads don't each construct
        # a Client instance during their first call.
        if self._client is not None:
            return
        with self._lock:
            if self._client is not None:
                return
            from src.llm.openrouter_client import OpenRouterClient  # noqa: PLC0415

            api_key = os.environ.get("OPENROUTER_API_KEY", "")
            if not api_key:
                raise RuntimeError(
                    "OPENROUTER_API_KEY not set. "
                    "Add it to .env or export it before running."
                )
            self._client = OpenRouterClient(
                api_key=api_key,
                max_retries=MAX_RETRIES,
                base_delay=BASE_DELAY_S,
                timeout=60.0,
            )

    # ── Public API ───────────────────────────────────────────────────────────

    def extract(
        self,
        response_text: str,
        valid_labels: list[str],
        qid: str,
        model_name: str = "",
        cache_key_extra: str = "",
    ) -> list[str]:
        """Extract the set of letters the model committed to.

        Returns sorted list of letters (e.g. ``["A", "F"]``) or empty list.
        Empty list covers: empty input, refusal, incoherent response,
        JSON parse failure, API failure after retries.

        ``cache_key_extra`` is appended to the cache key. The benchmark
        uses this to namespace cached extractions by orchestrator
        configuration (e.g. ``"|exa_r3_u10_s1200"``) so a param change
        invalidates only the affected entries.
        """
        # Empty / whitespace input → skip API entirely
        if not response_text or not response_text.strip():
            return []

        valid_set = {str(L).upper() for L in valid_labels}

        cache_key = f"{model_name}|{qid}|{self.prompt_version}{cache_key_extra}"
        with self._lock:
            cached = self.cache.get(cache_key)
        if cached is not None:
            return [L for L in cached if L in valid_set]

        # Call Flash; on any failure return empty list
        try:
            raw_letters = self._call_flash(response_text, valid_labels)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Extractor failed for %s: %s", cache_key, exc)
            raw_letters = []

        filtered = sorted({str(L).strip().upper() for L in raw_letters if L} & valid_set)

        with self._lock:
            self.cache[cache_key] = filtered
            self._writes_since_save += 1
            if self._writes_since_save >= SAVE_EVERY_N_WRITES:
                self._save_cache_locked()

        return filtered

    # ── Underlying Flash call (via OpenRouter) ───────────────────────────────

    def _call_flash(self, response_text: str, valid_labels: list[str]) -> list[str]:
        self._ensure_client()

        valid_csv = ",".join(valid_labels)
        user_text = (
            f"Valid labels: {valid_csv}\n\n"
            f"Response:\n---\n{response_text}\n---"
        )

        # Build OpenAI-format messages: system + 5 few-shot (user, assistant)
        # pairs + the real query.
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for shot_user, shot_letters in FEW_SHOTS:
            messages.append({"role": "user", "content": shot_user})
            messages.append({
                "role": "assistant",
                "content": json.dumps({"letters": shot_letters}),
            })
        messages.append({"role": "user", "content": user_text})

        # OpenRouterClient.chat retries 429/5xx internally (max_retries set
        # at construction). It returns parsed JSON directly.
        try:
            parsed, _record = self._client.chat(
                model=EXTRACTOR_MODEL,
                messages=messages,
                json_schema=RESPONSE_SCHEMA,
                temperature=0.0,
                stage="extractor",
            )
        except RuntimeError as exc:
            # OpenRouterClient raises RuntimeError after exhausting retries;
            # treat as extraction failure (caller returns empty list).
            logger.warning("Extractor API error after retries: %s", exc)
            raise

        letters = (parsed or {}).get("letters") or []
        return [str(L).strip().upper() for L in letters if L]
