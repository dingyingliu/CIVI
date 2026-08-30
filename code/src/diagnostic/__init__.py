"""Stage 2 source-injection diagnostic.

Classifies main-run failures into four modes — search_bypass, retrieval,
grounding, comprehension — by re-running each failed question with the
authoritative source page text injected into the prompt.
"""
