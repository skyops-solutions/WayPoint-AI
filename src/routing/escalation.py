"""Phase 4 — Escalation Logic & Webhook. See phases/phase_4_routing.md §4.2 & §4.3"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from src.config import settings
from src.models import BotResponse, ChatSession, Turn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent sets
# ---------------------------------------------------------------------------

ALWAYS_ESCALATE_INTENTS = {"complaint", "legal_dispute", "human_request"}
CONDITIONAL_ESCALATE_INTENTS = {"refund_request", "accessibility_need"}
CONDITIONAL_CONFIDENCE_THRESHOLD = 0.75

_HUMAN_KEYWORDS = {"speak to a person", "talk to a human", "human agent", "real person", "speak to someone"}

# How many consecutive low-confidence bot turns trigger escalation
_CONSECUTIVE_LOW_CONF_TURNS = 3

# Message appended to answer on escalation (§4.5)
_ESCALATION_SUFFIX = (
    "\n\nI'll connect you with one of our travel specialists who can assist you further. "
    "You'll be reached shortly. Reference ID: {ref}"
)


# ---------------------------------------------------------------------------
# Escalation decision (§4.2)
# ---------------------------------------------------------------------------

def should_escalate(
    intent: str,
    confidence: float,
    session: ChatSession,
    confidence_threshold: float | None = None,
) -> tuple[bool, str]:
    """Evaluate escalation rules in priority order.

    Returns (should_escalate, reason).
    reason: 'intent' | 'user_request' | 'low_confidence' | 'repeated_low_confidence'
    """
    threshold = confidence_threshold if confidence_threshold is not None else settings.confidence_threshold

    # Rule 1: always-escalate intents
    if intent in ALWAYS_ESCALATE_INTENTS:
        return True, "intent"

    # Rule 2: explicit human request keywords in session's last user turn
    # (intent == human_request already handled above, but cover keyword variants)
    last_user_msg = _last_user_message(session)
    if last_user_msg and _contains_human_request(last_user_msg):
        return True, "user_request"

    # Rule 3: current confidence below threshold
    if confidence < threshold:
        return True, "low_confidence"

    # Rule 4: 3 consecutive bot turns with low confidence
    if _consecutive_low_confidence(session, threshold, n=_CONSECUTIVE_LOW_CONF_TURNS):
        return True, "repeated_low_confidence"

    # Rule 5: conditional intents with moderate confidence
    if intent in CONDITIONAL_ESCALATE_INTENTS and confidence < CONDITIONAL_CONFIDENCE_THRESHOLD:
        return True, "intent"

    return False, ""


def apply_escalation_message(response: BotResponse, session_id: str) -> BotResponse:
    """Append the escalation suffix to the answer (§4.5)."""
    suffix = _ESCALATION_SUFFIX.format(ref=session_id[:8])
    if suffix not in response.answer:
        response.answer = response.answer.rstrip() + suffix
    return response


# ---------------------------------------------------------------------------
# Webhook (§4.3)
# ---------------------------------------------------------------------------

async def fire_escalation(session: ChatSession, reason: str, intent: str) -> None:
    """Fire-and-forget POST to HUMAN_SUPPORT_WEBHOOK with full transcript."""
    webhook_url = settings.human_support_webhook
    if not webhook_url:
        logger.warning("HUMAN_SUPPORT_WEBHOOK not configured — skipping escalation webhook.")
        return

    payload = {
        "session_id": session.session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "detected_intent": intent,
        "transcript": [
            {
                "role": t.role,
                "content": t.content,
                "timestamp": t.timestamp.isoformat(),
            }
            for t in session.turns
        ],
        "last_user_message": _last_user_message(session) or "",
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(webhook_url, json=payload)
            if resp.status_code >= 400:
                logger.error(
                    "Escalation webhook returned %d for session %s",
                    resp.status_code,
                    session.session_id,
                )
            else:
                logger.info("Escalation webhook fired (reason=%s, session=%s)", reason, session.session_id)
    except Exception as exc:
        logger.error("Escalation webhook failed for session %s: %s", session.session_id, exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _last_user_message(session: ChatSession) -> str | None:
    for turn in reversed(session.turns):
        if turn.role == "user":
            return turn.content
    return None


def _contains_human_request(message: str) -> bool:
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in _HUMAN_KEYWORDS)


def _consecutive_low_confidence(
    session: ChatSession, threshold: float, n: int
) -> bool:
    """Return True if the last n assistant turns all had confidence < threshold."""
    assistant_turns = [t for t in session.turns if t.role == "assistant"]
    if len(assistant_turns) < n:
        return False
    return all(
        (t.confidence is not None and t.confidence < threshold)
        for t in assistant_turns[-n:]
    )
