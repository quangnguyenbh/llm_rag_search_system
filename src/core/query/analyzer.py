"""Query analysis: intent classification and complexity scoring."""

from dataclasses import dataclass, field


@dataclass
class QueryAnalysis:
    intent: str  # factual, procedural, comparative, troubleshoot
    entities: dict = field(default_factory=dict)
    complexity: float = 0.5
    rewritten_query: str = ""


# Intent keywords
_PROCEDURAL_KEYWORDS = {"how to", "steps", "procedure", "install", "assemble", "setup", "configure"}
_TROUBLESHOOT_KEYWORDS = {"error", "problem", "issue", "not working", "fix", "troubleshoot", "fails"}
_COMPARATIVE_KEYWORDS = {"compare", "difference", "vs", "versus", "better", "which one"}


class QueryAnalyzer:
    async def analyze(self, question: str) -> QueryAnalysis:
        """Analyze a user query for intent and complexity."""
        lower = question.lower()

        # Intent classification via keyword matching
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

        # Complexity: longer / multi-part questions are harder
        word_count = len(question.split())
        if word_count < 8:
            complexity = 0.2
        elif word_count < 20:
            complexity = 0.5
        else:
            complexity = 0.8

        return QueryAnalysis(
            intent=intent,
            entities={},
            complexity=complexity,
            rewritten_query=question,
        )
