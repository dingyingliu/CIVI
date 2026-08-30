"""OpenRouter Chat Completions client with structured output support.

Wraps the OpenRouter API with retry logic, rate-limit handling,
structured JSON output via ``response_format``, and per-call logging.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class LLMCallRecord:
    """Record of a single LLM API call for observability.

    Attributes:
        model_requested: The model ID sent in the request.
        model_responded: The model ID returned in the response.
        response_id: OpenRouter's response ID.
        prompt_chars: Character count of the prompt.
        usage: Token usage dict from the response.
        duration_s: Wall-clock time for the call.
        stage: Pipeline stage name (e.g. ``"fact_extraction"``).
    """

    model_requested: str = ""
    model_responded: str = ""
    response_id: str = ""
    prompt_chars: int = 0
    usage: dict = field(default_factory=dict)
    duration_s: float = 0.0
    stage: str = ""


class OpenRouterClient:
    """HTTP client for OpenRouter Chat Completions.

    Supports structured JSON output via ``json_schema`` response format,
    exponential backoff on 429/5xx errors, and optional session/trace
    metadata for observability.

    Attributes:
        api_key: OpenRouter API key.
        max_retries: Maximum number of retries on transient errors.
        base_delay: Initial backoff delay in seconds.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        max_retries: int = 3,
        base_delay: float = 2.0,
        timeout: float = 120.0,
    ) -> None:
        """Initialise the client.

        Args:
            api_key: OpenRouter API key.  Falls back to the
                ``OPENROUTER_API_KEY`` environment variable.
            max_retries: Number of retries on 429 / 5xx responses.
            base_delay: Base delay for exponential backoff (seconds).
            timeout: HTTP request timeout in seconds.

        Raises:
            ValueError: If no API key is provided or found in env.
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Pass api_key= or set OPENROUTER_API_KEY."
            )
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        json_schema: dict | None = None,
        temperature: float = 0.3,
        seed: int | None = None,
        stage: str = "",
    ) -> tuple[dict, LLMCallRecord]:
        """Send a chat completion request and return parsed JSON.

        Args:
            model: OpenRouter model ID (e.g. ``"openai/gpt-4o"``).
            messages: Chat messages in OpenAI format.
            json_schema: If provided, enables structured output mode.
                The schema is sent as ``response_format.json_schema``.
            temperature: Sampling temperature.
            seed: Optional seed for deterministic outputs.
            stage: Pipeline stage name for logging.

        Returns:
            A tuple of (parsed_response_dict, LLMCallRecord).

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://example.com/civi-eval",
            "X-Title": "factual-qa-generator-agent",
        }

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if seed is not None:
            body["seed"] = seed
        if json_schema:
            schema_copy = copy.deepcopy(json_schema)
            is_google = model.startswith("google/")
            if is_google:
                self._adapt_schema_for_gemini(schema_copy)
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_copy.pop("title", "response"),
                    "strict": not is_google,
                    "schema": schema_copy,
                },
            }

        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        record = LLMCallRecord(
            model_requested=model,
            prompt_chars=prompt_chars,
            stage=stage,
        )

        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                resp = self._client.post(_BASE_URL, headers=headers, json=body)
                record.duration_s = time.time() - t0

                if resp.status_code == 429 or resp.status_code >= 500:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(
                        "HTTP %d from %s (attempt %d/%d), retrying in %.1fs",
                        resp.status_code, model, attempt + 1,
                        self.max_retries + 1, delay,
                    )
                    time.sleep(delay)
                    continue

                resp.raise_for_status()
                data = resp.json()

                record.response_id = data.get("id", "")
                record.model_responded = data.get("model", model)
                record.usage = data.get("usage", {})

                # Extract content from the first choice
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)

                logger.debug(
                    "LLM call OK: model=%s stage=%s tokens=%s",
                    model, stage, record.usage,
                )
                return parsed, record

            except json.JSONDecodeError:
                # If JSON parsing fails, retry with a nudge
                if attempt < self.max_retries:
                    logger.warning(
                        "JSON parse error from %s (attempt %d), retrying",
                        model, attempt + 1,
                    )
                    time.sleep(self.base_delay)
                    continue
                raise RuntimeError(
                    f"Failed to parse JSON from {model} after {self.max_retries + 1} attempts. "
                    f"Raw content: {content[:500]}"
                )
            except httpx.HTTPStatusError as e:
                if attempt < self.max_retries:
                    logger.warning(
                        "HTTP error %s, retrying. Body: %s",
                        e, e.response.text[:300],
                    )
                    time.sleep(self.base_delay * (2 ** attempt))
                    continue
                raise RuntimeError(
                    f"OpenRouter API error after retries: {e}\n"
                    f"Response body: {e.response.text[:500]}"
                ) from e

        raise RuntimeError(f"All {self.max_retries + 1} attempts failed for {model}")

    @staticmethod
    def _adapt_schema_for_gemini(schema: dict) -> None:
        """Adapt a JSON schema for Gemini's structured output format.

        Gemini does not support ``additionalProperties`` and requires
        ``propertyOrdering`` instead.  It also does not support ``enum``
        on ``number`` types.  This recursively transforms the schema in
        place.
        """
        schema.pop("additionalProperties", None)
        # Remove enum from number types (Gemini doesn't support it)
        if schema.get("type") == "number":
            schema.pop("enum", None)
        props = schema.get("properties")
        if props:
            schema["propertyOrdering"] = list(props.keys())
            for prop_schema in props.values():
                OpenRouterClient._adapt_schema_for_gemini(prop_schema)
        items = schema.get("items")
        if isinstance(items, dict):
            OpenRouterClient._adapt_schema_for_gemini(items)

    def chat_text(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0,
        max_tokens: int = 2048,
        stage: str = "",
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        """Send a chat completion request and return raw text content.

        Unlike ``chat()``, this does not expect JSON output and returns
        the raw string content from the model.  Used for cold evaluation
        where structured output is not needed.

        Args:
            model: OpenRouter model ID.
            messages: Chat messages in OpenAI format.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.
            stage: Pipeline stage name for logging.
            extra_body: Optional dict of additional keys to merge into the
                request body (e.g. provider-specific parameters).

        Returns:
            The raw text content from the model, or empty string on failure.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://example.com/civi-eval",
            "X-Title": "factual-qa-generator-agent",
        }

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra_body:
            body.update(extra_body)

        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                resp = self._client.post(_BASE_URL, headers=headers, json=body)
                duration = time.time() - t0

                if resp.status_code == 429 or resp.status_code >= 500:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(
                        "HTTP %d from %s (attempt %d/%d), retrying in %.1fs",
                        resp.status_code, model, attempt + 1,
                        self.max_retries + 1, delay,
                    )
                    time.sleep(delay)
                    continue

                if resp.status_code >= 400:
                    body_text = resp.text if resp.text else "(empty)"
                    logger.error(
                        "HTTP %d from %s â€” full body: %s",
                        resp.status_code, model, body_text,
                    )

                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"].get("content")

                logger.debug(
                    "LLM text call OK: model=%s stage=%s duration=%.1fs",
                    model, stage, duration,
                )
                return (content or "").strip()

            except Exception as exc:
                logger.warning(
                    "Error calling %s (attempt %d/%d): %s",
                    model, attempt + 1, self.max_retries + 1, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.base_delay * (2 ** attempt))

        return ""

    def chat_eval(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0,
        max_tokens: int = 8192,
        stage: str = "",
        tools: list[dict] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion with optional tools and return rich response.

        Returns a dict with:
            - text: the model's text response
            - searched: whether web search was invoked
            - cited_urls: list of URLs cited in annotations
            - search_count: number of search queries made
            - raw_annotations: the raw annotations array
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://example.com/civi-eval",
            "X-Title": "factual-qa-generator-agent",
        }

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = tools
        if extra_body:
            body.update(extra_body)

        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                resp = self._client.post(_BASE_URL, headers=headers, json=body)
                duration = time.time() - t0

                if resp.status_code == 429 or resp.status_code >= 500:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(
                        "HTTP %d from %s (attempt %d/%d), retrying in %.1fs",
                        resp.status_code, model, attempt + 1,
                        self.max_retries + 1, delay,
                    )
                    time.sleep(delay)
                    continue

                if resp.status_code >= 400:
                    body_text = resp.text if resp.text else "(empty)"
                    logger.error(
                        "HTTP %d from %s â€” full body: %s",
                        resp.status_code, model, body_text,
                    )

                resp.raise_for_status()
                data = resp.json()

                choice = data["choices"][0]
                message = choice["message"]
                content = message.get("content", "")
                annotations = message.get("annotations", []) or []
                usage = data.get("usage", {})

                if "deepseek" in model.lower():
                    finish_reason = choice.get("finish_reason")
                    tool_calls = message.get("tool_calls") or []
                    raw_content = content or ""
                    logger.debug(
                        "DeepSeek raw response: model=%s finish_reason=%s "
                        "tool_calls=%d content_len=%d content[:500]=%r",
                        model, finish_reason, len(tool_calls),
                        len(raw_content), raw_content[:500],
                    )

                if "qwen" in model.lower():
                    finish_reason = choice.get("finish_reason")
                    tool_calls = message.get("tool_calls") or []
                    raw_content = content or ""
                    reasoning_field = message.get("reasoning") or ""
                    reasoning_content_field = message.get("reasoning_content") or ""
                    try:
                        full_message_json = json.dumps(message, ensure_ascii=False)
                    except (TypeError, ValueError):
                        full_message_json = repr(message)
                    logger.info(
                        "Qwen raw response: model=%s finish_reason=%s "
                        "message_keys=%s tool_calls=%d content_len=%d "
                        "reasoning_len=%d reasoning_content_len=%d "
                        "content[:500]=%r reasoning[:500]=%r "
                        "full_message[:3000]=%s",
                        model, finish_reason, sorted(message.keys()),
                        len(tool_calls), len(raw_content),
                        len(reasoning_field), len(reasoning_content_field),
                        raw_content[:500], reasoning_field[:500],
                        full_message_json[:3000],
                    )

                search_count = usage.get("web_search_requests", 0)
                cited_urls = []
                for ann in annotations:
                    if ann.get("type") == "url_citation":
                        url = ann.get("url_citation", {}).get("url", "")
                        if url:
                            cited_urls.append(url)

                searched = search_count > 0 or len(cited_urls) > 0

                # Diagnostic fields â€” captured for every call so post-hoc
                # debugging of empty-response / budget-exhaustion patterns
                # doesn't require a re-run.
                finish_reason = choice.get("finish_reason")
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                completion_details = usage.get("completion_tokens_details") or {}
                reasoning_tokens = completion_details.get("reasoning_tokens")

                logger.debug(
                    "LLM eval call OK: model=%s stage=%s duration=%.1fs "
                    "searched=%s urls=%d finish=%s prompt_tok=%s "
                    "comp_tok=%s reasoning_tok=%s",
                    model, stage, duration, searched, len(cited_urls),
                    finish_reason, prompt_tokens, completion_tokens, reasoning_tokens,
                )
                return {
                    "text":              (content or "").strip(),
                    "searched":          searched,
                    "cited_urls":        cited_urls,
                    "search_count":      search_count,
                    "raw_annotations":   annotations,
                    "finish_reason":     finish_reason,
                    "prompt_tokens":     prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "reasoning_tokens":  reasoning_tokens,
                }

            except Exception as exc:
                logger.warning(
                    "Error calling %s (attempt %d/%d): %s",
                    model, attempt + 1, self.max_retries + 1, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.base_delay * (2 ** attempt))

        return {
            "text":              "",
            "searched":          False,
            "cited_urls":        [],
            "search_count":      0,
            "raw_annotations":   [],
            "finish_reason":     None,
            "prompt_tokens":     None,
            "completion_tokens": None,
            "reasoning_tokens":  None,
        }

    def chat_with_tools(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict],
        temperature: float = 0,
        max_tokens: int = 8192,
        stage: str = "",
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Single chat-completion call exposing the raw assistant message.

        Used by client-side tool-calling loops (e.g. the agentic Exa
        orchestrator in ``benchmark_text_qa``).  Unlike ``chat_eval``,
        this preserves the assistant message verbatim â€” including any
        ``tool_calls`` â€” so the caller can drive the multi-turn loop.

        Returns ``{"message": dict, "finish_reason": str, "usage": dict}``.
        On error after retries, returns a stub with empty content and
        ``finish_reason="error"``.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://example.com/civi-eval",
            "X-Title": "factual-qa-generator-agent",
        }

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": tools,
        }
        if extra_body:
            body.update(extra_body)

        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                resp = self._client.post(_BASE_URL, headers=headers, json=body)
                duration = time.time() - t0

                if resp.status_code == 429 or resp.status_code >= 500:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(
                        "HTTP %d from %s (attempt %d/%d), retrying in %.1fs",
                        resp.status_code, model, attempt + 1,
                        self.max_retries + 1, delay,
                    )
                    time.sleep(delay)
                    continue

                if resp.status_code >= 400:
                    body_text = resp.text if resp.text else "(empty)"
                    logger.error(
                        "HTTP %d from %s â€” full body: %s",
                        resp.status_code, model, body_text,
                    )

                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                logger.debug(
                    "chat_with_tools OK: model=%s stage=%s duration=%.1fs "
                    "finish=%s",
                    model, stage, duration, choice.get("finish_reason"),
                )
                return {
                    "message":       choice["message"],
                    "finish_reason": choice.get("finish_reason"),
                    "usage":         data.get("usage", {}),
                }

            except Exception as exc:
                logger.warning(
                    "chat_with_tools error %s (attempt %d/%d): %s",
                    model, attempt + 1, self.max_retries + 1, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.base_delay * (2 ** attempt))

        return {
            "message":       {"role": "assistant", "content": ""},
            "finish_reason": "error",
            "usage":         {},
        }

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

