"""LLM generation via Amazon Nova on Bedrock with grounding prompt and streaming support."""

import json
import asyncio
from dataclasses import dataclass

import boto3
import structlog

from src.config import settings

logger = structlog.get_logger()


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
    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=settings.aws_bedrock_region,
            )
        return self._client

    def _build_body(self, user_message: str) -> str:
        """Build Nova-format request body."""
        return json.dumps({
            "system": [{"text": SYSTEM_PROMPT}],
            "messages": [
                {"role": "user", "content": [{"text": user_message}]}
            ],
            "inferenceConfig": {"maxTokens": 4096},
        })

    def _build_user_message(self, question: str, context: str) -> str:
        return f"""Context from retrieved documents:

{context}

---

Question: {question}

Please answer the question using ONLY the context above. Cite sources using [Source N] notation."""

    async def generate(self, question: str, context: str, model: str) -> GenerationResult:
        """Generate a grounded answer using Nova on Bedrock."""
        if not context:
            return GenerationResult(
                answer="I don't have enough information in the available documents to answer this question.",
                model=model,
            )

        body = self._build_body(self._build_user_message(question, context))
        client = self._get_client()
        loop = asyncio.get_event_loop()

        response = await loop.run_in_executor(
            None,
            lambda: client.invoke_model(
                modelId=model,
                body=body,
                contentType="application/json",
                accept="application/json",
            ),
        )

        result = json.loads(response["body"].read())
        answer = result["output"]["message"]["content"][0]["text"]
        usage = result.get("usage", {})

        logger.info(
            "generator.complete",
            model=model,
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
        )

        return GenerationResult(
            answer=answer,
            model=model,
            prompt_tokens=usage.get("inputTokens", 0),
            completion_tokens=usage.get("outputTokens", 0),
        )

    async def generate_stream(self, question: str, context: str, model: str):
        """Generate a grounded answer with streaming via Bedrock."""
        if not context:
            yield "I don't have enough information in the available documents to answer this question."
            return

        body = self._build_body(self._build_user_message(question, context))
        client = self._get_client()
        loop = asyncio.get_event_loop()

        response = await loop.run_in_executor(
            None,
            lambda: client.invoke_model_with_response_stream(
                modelId=model,
                body=body,
                contentType="application/json",
                accept="application/json",
            ),
        )

        stream = response.get("body")
        for event in stream:
            chunk = event.get("chunk")
            if chunk:
                data = json.loads(chunk["bytes"])
                if data.get("type") == "contentBlockDelta":
                    text = data.get("delta", {}).get("text", "")
                    if text:
                        yield text
