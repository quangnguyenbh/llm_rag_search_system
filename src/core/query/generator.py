"""LLM generation with grounding prompt and streaming support."""

from dataclasses import dataclass


SYSTEM_PROMPT = """You are ManualAI, a technical documentation assistant. You answer questions \
based ONLY on the provided document context.

Rules:
1. Only use information from the provided context to answer.
2. Cite every factual claim with [Source N] matching the context sources.
3. If the context does not contain enough information, say: \
"I don't have enough information in the available documents to answer this question."
4. When referencing tables, reproduce relevant data accurately.
5. Never fabricate part numbers, specifications, or procedures.
6. Be precise and concise. Prefer bullet points for multi-step procedures.
"""


@dataclass
class GenerationResult:
    answer: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Generator:
    async def generate(self, question: str, context: str, model: str) -> GenerationResult:
        """Generate a grounded answer using the specified LLM."""
        # TODO: Implement OpenAI / Anthropic / open model calls
        # TODO: Implement streaming SSE variant
        return GenerationResult(answer="", model=model)
