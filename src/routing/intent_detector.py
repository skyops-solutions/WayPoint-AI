"""Phase 4 — Intent Detector. See phases/phase_4_routing.md §4.1"""
from __future__ import annotations

import logging

from google import genai

from src.config import settings

logger = logging.getLogger(__name__)

INTENTS = [
    "booking_inquiry",
    "cancellation",
    "itinerary_change",
    "destination_info",
    "policy_question",
    "complaint",
    "refund_request",
    "legal_dispute",
    "accessibility_need",
    "human_request",
    "general",
]

_INTENT_SET = set(INTENTS)

_PROMPT_TEMPLATE = (
    "Classify the following customer message into exactly one of these intents: "
    "{intents}.\n"
    "Respond with only the intent label, nothing else.\n\n"
    "Message: {message}"
)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.google_api_key)
    return _client


async def detect_intent(message: str) -> str:
    """Classify a user message into one of the supported intent labels.

    Returns a valid intent string. Falls back to 'general' on any error.
    """
    prompt = _PROMPT_TEMPLATE.format(
        intents=", ".join(INTENTS),
        message=message,
    )
    try:
        client = _get_client()
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
        )
        label = response.text.strip().lower()
        if label in _INTENT_SET:
            return label
        # Gemini sometimes returns extra words — try to find a match
        for intent in INTENTS:
            if intent in label:
                return intent
        logger.warning("Unrecognised intent label '%s' — defaulting to 'general'", label)
        return "general"
    except Exception as exc:
        logger.error("Intent detection failed: %s — defaulting to 'general'", exc)
        return "general"
