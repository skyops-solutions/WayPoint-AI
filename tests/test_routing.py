"""Tests for Phase 4 — routing and escalation logic."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from src.models import BotResponse, ChatSession, Turn
from src.routing.escalation import (
    ALWAYS_ESCALATE_INTENTS,
    CONDITIONAL_ESCALATE_INTENTS,
    _consecutive_low_confidence,
    _contains_human_request,
    apply_escalation_message,
    should_escalate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session(turns: list[Turn] | None = None) -> ChatSession:
    return ChatSession(
        session_id="abcd1234-efgh",
        created_at=datetime.utcnow(),
        last_active=datetime.utcnow(),
        turns=turns or [],
    )


def _turn(role: str = "assistant", confidence: float | None = 0.8) -> Turn:
    return Turn(
        role=role,
        content="some message",
        timestamp=datetime.utcnow(),
        confidence=confidence,
    )


def _response(confidence: float = 0.8, escalate: bool = False) -> BotResponse:
    return BotResponse(
        answer="Here is your answer.",
        confidence=confidence,
        escalate_to_human=escalate,
        booking_link=None,
        related_services=[],
        sources=[],
    )


# ---------------------------------------------------------------------------
# Escalation rules (§4.2)
# ---------------------------------------------------------------------------

class TestShouldEscalate:
    # Rule 1: always-escalate intents
    @pytest.mark.parametrize("intent", sorted(ALWAYS_ESCALATE_INTENTS))
    def test_always_escalate_intents(self, intent: str):
        escalate, reason = should_escalate(intent, confidence=0.95, session=_session())
        assert escalate is True
        assert reason == "intent"

    # Rule 2: explicit human request keyword in last user turn
    def test_human_keyword_in_message_escalates(self):
        turns = [Turn(role="user", content="I want to speak to a person", timestamp=datetime.utcnow())]
        escalate, reason = should_escalate("general", confidence=0.9, session=_session(turns))
        assert escalate is True
        assert reason == "user_request"

    def test_no_human_keyword_no_escalation(self):
        turns = [Turn(role="user", content="What is your refund timeline?", timestamp=datetime.utcnow())]
        escalate, _ = should_escalate("general", confidence=0.9, session=_session(turns))
        assert escalate is False

    # Rule 3: low confidence
    def test_low_confidence_escalates(self):
        escalate, reason = should_escalate("general", confidence=0.3, session=_session())
        assert escalate is True
        assert reason == "low_confidence"

    def test_confidence_at_threshold_does_not_escalate(self):
        escalate, _ = should_escalate("general", confidence=0.6, session=_session(), confidence_threshold=0.6)
        assert escalate is False

    def test_confidence_just_below_threshold_escalates(self):
        escalate, reason = should_escalate("general", confidence=0.59, session=_session(), confidence_threshold=0.6)
        assert escalate is True
        assert reason == "low_confidence"

    # Rule 4: 3 consecutive low-confidence assistant turns
    def test_three_consecutive_low_confidence_escalates(self):
        turns = [_turn("assistant", confidence=0.3) for _ in range(3)]
        escalate, reason = should_escalate("general", confidence=0.9, session=_session(turns))
        assert escalate is True
        assert reason == "repeated_low_confidence"

    def test_two_consecutive_low_confidence_does_not_escalate(self):
        turns = [_turn("assistant", confidence=0.3) for _ in range(2)]
        escalate, _ = should_escalate("general", confidence=0.9, session=_session(turns))
        assert escalate is False

    def test_three_turns_but_last_high_confidence_does_not_trigger(self):
        turns = [
            _turn("assistant", confidence=0.3),
            _turn("assistant", confidence=0.3),
            _turn("assistant", confidence=0.9),
        ]
        escalate, _ = should_escalate("general", confidence=0.9, session=_session(turns))
        assert escalate is False

    # Rule 5: conditional intents
    @pytest.mark.parametrize("intent", sorted(CONDITIONAL_ESCALATE_INTENTS))
    def test_conditional_intent_below_threshold_escalates(self, intent: str):
        # confidence=0.65: above general threshold (0.6) so Rule 3 won't fire,
        # but below conditional threshold (0.75) so Rule 5 fires with reason='intent'
        escalate, reason = should_escalate(intent, confidence=0.65, session=_session())
        assert escalate is True
        assert reason == "intent"

    @pytest.mark.parametrize("intent", sorted(CONDITIONAL_ESCALATE_INTENTS))
    def test_conditional_intent_above_threshold_does_not_escalate(self, intent: str):
        escalate, _ = should_escalate(intent, confidence=0.9, session=_session())
        assert escalate is False


# ---------------------------------------------------------------------------
# Escalation message (§4.5)
# ---------------------------------------------------------------------------

class TestApplyEscalationMessage:
    def test_suffix_appended(self):
        resp = _response()
        result = apply_escalation_message(resp, "abcd1234")
        assert "travel specialists" in result.answer
        assert "abcd" in result.answer  # first 8 chars of session_id

    def test_suffix_not_duplicated_on_double_call(self):
        resp = _response()
        apply_escalation_message(resp, "abcd1234")
        apply_escalation_message(resp, "abcd1234")
        assert resp.answer.count("travel specialists") == 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_contains_human_request_matches_keywords(self):
        assert _contains_human_request("I want to speak to a person")
        assert _contains_human_request("Can I talk to a human?")
        assert _contains_human_request("I need a human agent please")

    def test_contains_human_request_no_match(self):
        assert not _contains_human_request("How do I cancel my booking?")

    def test_consecutive_low_confidence_exact(self):
        turns = [_turn("assistant", 0.3), _turn("assistant", 0.3), _turn("assistant", 0.3)]
        assert _consecutive_low_confidence(_session(turns), threshold=0.6, n=3) is True

    def test_consecutive_low_confidence_not_enough_turns(self):
        turns = [_turn("assistant", 0.3), _turn("assistant", 0.3)]
        assert _consecutive_low_confidence(_session(turns), threshold=0.6, n=3) is False

    def test_consecutive_ignores_user_turns(self):
        turns = [
            _turn("user"),
            _turn("assistant", 0.3),
            _turn("user"),
            _turn("assistant", 0.3),
            _turn("user"),
            _turn("assistant", 0.3),
        ]
        assert _consecutive_low_confidence(_session(turns), threshold=0.6, n=3) is True


# ---------------------------------------------------------------------------
# Router (§4.4)
# ---------------------------------------------------------------------------

class TestRouter:
    @pytest.mark.anyio
    async def test_always_escalate_intent_sets_true(self):
        from src.routing.router import Router

        router = Router()
        resp = _response(confidence=0.95, escalate=False)
        session = _session()

        with patch("src.routing.router.detect_intent", new=AsyncMock(return_value="complaint")), \
             patch("src.routing.router.fire_escalation", new=AsyncMock()):
            result = await router.route("I am unhappy", resp, session)

        assert result.escalate_to_human is True
        assert "travel specialists" in result.answer

    @pytest.mark.anyio
    async def test_never_downgrades_escalate_true(self):
        from src.routing.router import Router

        router = Router()
        # LLM already set escalate=True, intent says no escalation needed
        resp = _response(confidence=0.95, escalate=True)
        session = _session()

        with patch("src.routing.router.detect_intent", new=AsyncMock(return_value="general")), \
             patch("src.routing.router.fire_escalation", new=AsyncMock()):
            result = await router.route("basic question", resp, session)

        assert result.escalate_to_human is True

    @pytest.mark.anyio
    async def test_no_escalation_for_normal_query(self):
        from src.routing.router import Router

        router = Router()
        resp = _response(confidence=0.9, escalate=False)
        session = _session()

        with patch("src.routing.router.detect_intent", new=AsyncMock(return_value="booking_inquiry")), \
             patch("src.routing.router.fire_escalation", new=AsyncMock()) as mock_webhook:
            result = await router.route("book a flight to Rome", resp, session)

        assert result.escalate_to_human is False
        mock_webhook.assert_not_called()

    @pytest.mark.anyio
    async def test_webhook_failure_does_not_affect_response(self):
        from src.routing.router import Router

        router = Router()
        resp = _response(confidence=0.2, escalate=False)
        session = _session()

        async def failing_webhook(*args, **kwargs):
            raise RuntimeError("webhook down")

        with patch("src.routing.router.detect_intent", new=AsyncMock(return_value="general")), \
             patch("src.routing.router.fire_escalation", new=AsyncMock(side_effect=failing_webhook)):
            # Should not raise even if webhook fails
            result = await router.route("help", resp, session)

        assert result.escalate_to_human is True

    @pytest.mark.anyio
    async def test_low_confidence_triggers_escalation(self):
        from src.routing.router import Router

        router = Router()
        resp = _response(confidence=0.3, escalate=False)
        session = _session()

        with patch("src.routing.router.detect_intent", new=AsyncMock(return_value="general")), \
             patch("src.routing.router.fire_escalation", new=AsyncMock()):
            result = await router.route("vague question", resp, session)

        assert result.escalate_to_human is True
