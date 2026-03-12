"""Phase 3 — Generator Orchestrator. See phases/phase_3_generation.md §3.5"""
from __future__ import annotations

import asyncio
import logging

from src.generation.llm_client import GeminiClient
from src.generation.prompt_builder import build_prompt
from src.generation.response_parser import FALLBACK_RESPONSE, parse_response
from src.models import BotResponse, ChatSession, RetrievalResult

logger = logging.getLogger(__name__)

_RETRY_INSTRUCTION = (
    "\n\nYour previous response was not valid JSON. Respond only with JSON."
)


class Generator:
    def __init__(self) -> None:
        self._client = GeminiClient()

    async def generate_response(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        session: ChatSession,
    ) -> BotResponse:
        """Full generation pipeline: prompt → LLM → parse → guardrails.

        Guardrails applied here:
          - 0 retrieved chunks → return FALLBACK immediately
          - max_similarity < 0.4 → cap confidence and force escalation
        """
        # Guardrail: no context → escalate immediately
        if not retrieval_result.chunks:
            logger.warning("No chunks retrieved — returning fallback.")
            return FALLBACK_RESPONSE

        prompt = build_prompt(
            query=query,
            retrieved_chunks=retrieval_result.chunks,
            history=session.turns,
        )

        bot_response = await self._generate_with_retry(prompt)

        # Confidence override: low retrieval similarity caps LLM confidence
        if retrieval_result.max_similarity < 0.4:
            bot_response.confidence = min(bot_response.confidence, 0.4)
            bot_response.escalate_to_human = True
            logger.info(
                "Low max_similarity (%.3f) — confidence capped, escalating.",
                retrieval_result.max_similarity,
            )

        return bot_response

    async def _generate_with_retry(self, prompt: str) -> BotResponse:
        """Call LLM, retry once with JSON reminder on parse failure."""
        try:
            raw = await self._client.generate(prompt)
        except asyncio.TimeoutError:
            logger.error("LLM timeout after retry — returning fallback.")
            return FALLBACK_RESPONSE

        result = parse_response(raw)
        if result is not None:
            return result

        # Retry with JSON reminder
        logger.warning("Parse failed — retrying with JSON reminder.")
        try:
            raw = await self._client.generate(prompt + _RETRY_INSTRUCTION)
        except asyncio.TimeoutError:
            logger.error("LLM timeout on retry — returning fallback.")
            return FALLBACK_RESPONSE

        result = parse_response(raw)
        if result is not None:
            return result

        logger.error("Parse failed on retry — returning fallback.")
        return FALLBACK_RESPONSE
