"""YAML prompt loader for swappable prompt templates.

Each YAML file contains ``system`` and ``user`` keys with prompt
templates.  Placeholders use Python ``str.format`` syntax
(e.g. ``{chunk_text}``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class PromptLoader:
    """Load and format prompt templates from YAML files.

    Attributes:
        prompts_dir: Directory containing YAML prompt files.
    """

    def __init__(self, prompts_dir: str | Path = "prompts") -> None:
        """Initialise the loader.

        Args:
            prompts_dir: Path to the directory containing ``.yaml``
                prompt files.
        """
        self.prompts_dir = Path(prompts_dir)
        self._cache: dict[str, dict[str, str]] = {}

    def load(self, prompt_name: str) -> dict[str, str]:
        """Load a prompt template by name (without extension).

        Args:
            prompt_name: File stem, e.g. ``"fact_extraction"``.

        Returns:
            Dict with ``"system"`` and ``"user"`` string templates.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
        """
        if prompt_name not in self._cache:
            path = self.prompts_dir / f"{prompt_name}.yaml"
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            self._cache[prompt_name] = {
                "system": raw.get("system", "").strip(),
                "user": raw.get("user", "").strip(),
            }
        return self._cache[prompt_name]

    def format_messages(
        self, prompt_name: str, **kwargs: Any
    ) -> list[dict[str, str]]:
        """Load a prompt and format both system and user templates.

        Args:
            prompt_name: File stem of the YAML prompt.
            **kwargs: Values to substitute into the templates.

        Returns:
            A list of message dicts ready for the OpenRouter client.
        """
        templates = self.load(prompt_name)
        messages = []
        if templates["system"]:
            messages.append({
                "role": "system",
                "content": templates["system"].format(**kwargs),
            })
        if templates["user"]:
            messages.append({
                "role": "user",
                "content": templates["user"].format(**kwargs),
            })
        return messages
