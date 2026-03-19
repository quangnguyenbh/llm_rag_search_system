"""Query analysis: intent classification, entity extraction, and complexity scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


@dataclass
class QueryAnalysis:
    intent: str  # factual | procedural | comparative | troubleshoot
    entities: dict = field(default_factory=dict)
    complexity: float = 0.5
    complexity_score: float = 0.5  # alias kept for API compatibility
    filters: dict = field(default_factory=dict)
    rewritten_query: str = ""

    def __post_init__(self) -> None:
        # Keep complexity_score in sync with complexity
        if self.complexity != self.complexity_score:
            self.complexity_score = self.complexity


# ---------------------------------------------------------------------------
# Intent keywords
# ---------------------------------------------------------------------------
_PROCEDURAL_KEYWORDS = {"how to", "steps", "procedure", "install", "assemble", "setup", "configure"}
_TROUBLESHOOT_KEYWORDS = {
    "error",
    "problem",
    "issue",
    "not working",
    "fix",
    "troubleshoot",
    "fails",
    "failure",
    "broken",
}
_COMPARATIVE_KEYWORDS = {"compare", "difference", "vs", "versus", "better", "which one", "vs."}

# ---------------------------------------------------------------------------
# Entity extraction patterns
# ---------------------------------------------------------------------------
# Model / part numbers: alphanumeric sequences with optional dashes, at least
# two segments (e.g. "ABC-1234", "XR-500", "A4-B2-C3")
_MODEL_NUMBER_RE = re.compile(r"\b[A-Z]{1,5}[-][A-Z0-9]{2,}(?:[-][A-Z0-9]+)*\b")

# Manufacturer names: capitalised words followed by "Inc", "Corp", "Ltd", or
# known brand patterns (simple heuristic)
_MANUFACTURER_RE = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:Inc|Corp|Ltd|LLC|Co|Group)\b"
)

# Part numbers: patterns like "P/N: 12345" or standalone alphanumeric codes
_PART_NUMBER_RE = re.compile(r"\b(?:P/?N[:\s]*)?([A-Z]{0,3}\d{4,}[A-Z]{0,3})\b")

# ---------------------------------------------------------------------------
# Abbreviation expansion map
# ---------------------------------------------------------------------------
_ABBREVIATIONS: dict[str, str] = {
    r"\bmfr\b": "manufacturer",
    r"\bspec\b": "specification",
    r"\bspecs\b": "specifications",
    r"\bpn\b": "part number",
    r"\bp/n\b": "part number",
    r"\bsn\b": "serial number",
    r"\bs/n\b": "serial number",
    r"\bmax\b": "maximum",
    r"\bmin\b": "minimum",
    r"\bapprox\b": "approximately",
    r"\bconfig\b": "configuration",
    r"\binstall\b": "installation",
    r"\bdiag\b": "diagnostic",
    r"\bdiags\b": "diagnostics",
    r"\bdoc\b": "document",
    r"\bdocs\b": "documents",
    r"\bref\b": "reference",
    r"\bvol\b": "volume",
    r"\bchap\b": "chapter",
    r"\bsec\b": "section",
    r"\btbl\b": "table",
    r"\bfig\b": "figure",
}


def _rewrite_query(text: str) -> str:
    """Expand common abbreviations in *text*."""
    result = text
    for pattern, replacement in _ABBREVIATIONS.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


class QueryAnalyzer:
    async def analyze(self, question: str) -> QueryAnalysis:
        """Analyse a user query for intent, entities, complexity, and filters."""
        lower = question.lower()

        # ----------------------------------------------------------------
        # Intent classification via keyword matching
        # ----------------------------------------------------------------
        intent = "factual"
        for kw in _PROCEDURAL_KEYWORDS:
            if kw in lower:
                intent = "procedural"
                break
        for kw in _TROUBLESHOOT_KEYWORDS:
            if kw in lower:
                intent = "troubleshoot"
                break
        for kw in _COMPARATIVE_KEYWORDS:
            if kw in lower:
                intent = "comparative"
                break

        # ----------------------------------------------------------------
        # Entity extraction
        # ----------------------------------------------------------------
        model_numbers = _MODEL_NUMBER_RE.findall(question)
        manufacturers = _MANUFACTURER_RE.findall(question)
        part_numbers = _PART_NUMBER_RE.findall(question)

        entities: dict = {}
        if model_numbers:
            entities["model_numbers"] = model_numbers
        if manufacturers:
            entities["manufacturers"] = manufacturers
        if part_numbers:
            entities["part_numbers"] = part_numbers

        # ----------------------------------------------------------------
        # Complexity score
        # ----------------------------------------------------------------
        word_count = len(question.split())
        entity_bonus = min(len(entities) * 0.1, 0.3)
        question_words = sum(
            1 for w in question.lower().split()
            if w in {"what", "why", "how", "when", "where", "who", "which"}
        )
        question_bonus = min(question_words * 0.05, 0.2)

        if word_count < 8:
            base = 0.2
        elif word_count < 20:
            base = 0.5
        else:
            base = 0.7

        complexity = round(min(base + entity_bonus + question_bonus, 1.0), 2)

        # ----------------------------------------------------------------
        # Filter generation from entities
        # ----------------------------------------------------------------
        filters: dict = {}
        if model_numbers:
            filters["model_numbers"] = model_numbers
        if part_numbers:
            filters["part_numbers"] = part_numbers

        # ----------------------------------------------------------------
        # Query rewrite
        # ----------------------------------------------------------------
        rewritten = _rewrite_query(question)

        return QueryAnalysis(
            intent=intent,
            entities=entities,
            complexity=complexity,
            complexity_score=complexity,
            filters=filters,
            rewritten_query=rewritten,
        )
