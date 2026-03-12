"""Phase 4 — Router Orchestrator. See phases/phase_4_routing.md §4.4"""
from __future__ import annotations

import logging

from src.models import BotResponse, ChatSession
from src.routing.escalation import (
    apply_escalation_message,
    fire_escalation,
    should_escalate,
)
from src.routing.intent_detector import detect_intent

logger = logging.getLogger(__name__)


class Router:
    async def route(
        self,
        message: str,
        bot_response: BotResponse,
        session: ChatSession,
    ) -> BotResponse:
        """Detect intent, apply escalation rules, fire webhook if escalating.

        The router NEVER downgrades escalate_to_human from True to False.
        If the LLM already set it True, it stays True regardless of routing rules.
        """
        intent = await detect_intent(message)
        logger.info("Detected intent: %s", intent)

        escalate, reason = should_escalate(
            intent=intent,
            confidence=bot_response.confidence,
            session=session,
        )

        # Upward-only override: never downgrade True → False
        if bot_response.escalate_to_human:
            escalate = True
            reason = reason or "llm_escalated"

        if escalate:
            bot_response.escalate_to_human = True
            bot_response = apply_escalation_message(bot_response, session.session_id)
            # Fire-and-forget: await with suppressed exceptions so webhook never blocks response
            try:
                await fire_escalation(session, reason, intent)
            except Exception as exc:
                logger.error("Escalation webhook error (suppressed): %s", exc)
            logger.info("Escalating session %s (reason=%s)", session.session_id, reason)

        return bot_response
