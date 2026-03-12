"""Phase 3 — Response Parser. See phases/phase_3_generation.md §3.3 & §3.4

Parses the LLM JSON output, validates against schema, applies guardrails,
and returns a BotResponse. Falls back to FALLBACK_RESPONSE on any failure.
"""
from __future__ import annotations

import json
import logging
import re

import jsonschema

from src.models import BotResponse, Source

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema (§3.3)
# ---------------------------------------------------------------------------

RESPONSE_SCHEMA: dict = {
    "type": "object",
    "required": ["answer", "confidence", "escalate_to_human"],
    "properties": {
        "answer": {"type": "string"},
        "booking_link": {"type": ["string", "null"]},
        "related_services": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 3,
        },
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "doc": {"type": "string"},
                    "page": {"type": "integer"},
                },
            },
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "escalate_to_human": {"type": "boolean"},
    },
}

# ---------------------------------------------------------------------------
# Fallback (§3.4)
# ---------------------------------------------------------------------------

FALLBACK_RESPONSE = BotResponse(
    answer=(
        "I'm having trouble processing your request right now. "
        "Let me connect you with our support team."
    ),
    booking_link=None,
    related_services=[],
    sources=[],
    confidence=0.0,
    escalate_to_human=True,
)

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_response(raw: str) -> BotResponse | None:
    """Parse and validate a raw JSON string from the LLM.

    Returns a BotResponse on success, or None if parsing/validation fails.
    Applies guardrails before returning.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("JSON decode error: %s — raw: %.200s", exc, raw)
        return None

    try:
        jsonschema.validate(data, RESPONSE_SCHEMA)
    except jsonschema.ValidationError as exc:
        logger.warning("Schema validation error: %s", exc.message)
        return None

    return _apply_guardrails(data)


def _apply_guardrails(data: dict) -> BotResponse:
    answer = data.get("answer", "").strip()
    booking_link = data.get("booking_link")
    related_services = data.get("related_services", [])[:3]
    raw_sources = data.get("sources", [])
    confidence = float(data.get("confidence", 0.0))
    escalate = bool(data.get("escalate_to_human", False))

    # Guardrail: booking link must be a valid URL
    if booking_link and not _URL_RE.match(str(booking_link)):
        logger.warning("Booking link failed URL validation — nullifying: %s", booking_link)
        booking_link = None

    # Guardrail: source grounding — non-trivial answer with no sources → cap confidence
    sources = [
        Source(doc=s["doc"], page=int(s["page"]))
        for s in raw_sources
        if isinstance(s, dict) and "doc" in s and "page" in s
    ]
    if not sources and len(answer) > 80:
        logger.warning("Non-trivial answer with no sources — potential hallucination, capping confidence.")
        confidence = min(confidence, 0.5)

    return BotResponse(
        answer=answer,
        booking_link=booking_link,
        related_services=related_services,
        sources=sources,
        confidence=confidence,
        escalate_to_human=escalate,
    )
