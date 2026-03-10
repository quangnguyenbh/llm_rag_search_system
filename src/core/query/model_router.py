"""Route queries to the appropriate Claude model on Bedrock based on complexity."""

from src.core.query.analyzer import QueryAnalysis


# Model tiers — using Bedrock cross-region inference model IDs
TIER_FAST = "fast"
TIER_STANDARD = "standard"
TIER_HEAVY = "heavy"

MODEL_MAP = {
    TIER_FAST: "us.amazon.nova-micro-v1:0",
    TIER_STANDARD: "us.amazon.nova-2-lite-v1:0",
    TIER_HEAVY: "us.amazon.nova-2-lite-v1:0",
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

        if analysis.intent == "comparative":
            tier = TIER_STANDARD

        return self.model_map.get(tier, self.model_map[TIER_STANDARD])
