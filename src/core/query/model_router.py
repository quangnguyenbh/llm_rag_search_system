"""Route queries to the appropriate LLM based on complexity and type."""

from src.core.query.analyzer import QueryAnalysis


# Model tiers
TIER_FAST = "fast"    # Haiku / GPT-4o-mini
TIER_STANDARD = "standard"  # Sonnet / GPT-4o
TIER_HEAVY = "heavy"  # Opus / GPT-4o (complex reasoning)

MODEL_MAP = {
    TIER_FAST: "claude-3-5-haiku-20241022",
    TIER_STANDARD: "claude-4-sonnet-20250514",
    TIER_HEAVY: "claude-4-sonnet-20250514",
}


class ModelRouter:
    def __init__(self, model_map: dict[str, str] | None = None):
        self.model_map = model_map or MODEL_MAP

    def select(self, analysis: QueryAnalysis) -> str:
        """Select a model based on query analysis."""
        if analysis.complexity < 0.3:
            tier = TIER_FAST
        elif analysis.complexity < 0.7:
            tier = TIER_STANDARD
        else:
            tier = TIER_HEAVY

        # Override for table-heavy queries
        if analysis.intent == "comparative":
            tier = TIER_STANDARD

        return self.model_map.get(tier, self.model_map[TIER_STANDARD])
