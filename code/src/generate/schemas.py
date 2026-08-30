"""JSON schemas for OpenRouter structured output mode.

Each schema defines the exact shape of the JSON response expected from
the LLM at each pipeline stage.  These are sent via the
``response_format.json_schema`` parameter.
"""

# ── QA builder schema (multiple choice only) ──

ONE_CALL_MC_SCHEMA: dict = {
    "title": "one_call_mc",
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "options": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "text": {"type": "string"},
                    "is_correct": {"type": "boolean"},
                },
                "required": ["label", "text", "is_correct"],
                "additionalProperties": False,
            },
        },
        "quotes": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["question", "options", "quotes"],
    "additionalProperties": False,
}

# ── Full-page schema (multiple MC questions in one call) ──

_FULL_PAGE_QUESTION: dict = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "options": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "text": {"type": "string"},
                    "is_correct": {"type": "boolean"},
                },
                "required": ["label", "text", "is_correct"],
                "additionalProperties": False,
            },
        },
        "quotes": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["question", "options", "quotes"],
    "additionalProperties": False,
}

FULL_PAGE_QA_SCHEMA: dict = {
    "title": "full_page_qa",
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": _FULL_PAGE_QUESTION,
        },
    },
    "required": ["questions"],
    "additionalProperties": False,
}
