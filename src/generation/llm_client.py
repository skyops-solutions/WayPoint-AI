"""Phase 3 — LLM Client. See phases/phase_3_generation.md §3.2

Async wrapper around google-genai >= 1.0.0.
- model: gemini-2.5-flash, temperature: 0.2
- response_mime_type: application/json (structured output)
- Timeout: 10s with one retry before raising
- Logs token usage per call
"""
from __future__ import annotations

import asyncio
import logging

from google import genai
from google.genai import types

from src.config import settings

logger = logging.getLogger(__name__)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.google_api_key)
    return _client


class GeminiClient:
    def __init__(self) -> None:
        self.model = settings.gemini_model

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
    ) -> str:
        """Call Gemini and return raw JSON string.

        Retries once on timeout. Raises on second failure.
        """
        for attempt in range(2):
            try:
                raw = await asyncio.wait_for(
                    self._call(prompt, temperature),
                    timeout=settings.llm_timeout,
                )
                return raw
            except asyncio.TimeoutError:
                if attempt == 0:
                    logger.warning("Gemini timeout (attempt 1) — retrying...")
                else:
                    logger.error("Gemini timeout on retry — giving up.")
                    raise

        raise RuntimeError("unreachable")

    async def _call(self, prompt: str, temperature: float) -> str:
        client = _get_client()
        response = await client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
        self._log_usage(response)
        return response.text

    @staticmethod
    def _log_usage(response: object) -> None:
        try:
            meta = response.usage_metadata  # type: ignore[attr-defined]
            logger.debug(
                "Gemini tokens — prompt: %d, candidates: %d, total: %d",
                meta.prompt_token_count,
                meta.candidates_token_count,
                meta.total_token_count,
            )
        except Exception:
            pass  # usage metadata not always present
