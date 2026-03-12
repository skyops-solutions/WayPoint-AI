"""Tests for Phase 3 — LLM response generation."""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.generation.prompt_builder import build_prompt
from src.generation.response_parser import (
    FALLBACK_RESPONSE,
    parse_response,
    _apply_guardrails,
)
from src.models import (
    BotResponse,
    ChatSession,
    ChunkMetadata,
    DocumentChunk,
    RetrievalResult,
    Turn,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(content: str = "policy details", doc_type: str = "policy") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=str(uuid.uuid4()),
        doc_id="doc_a",
        content=content,
        metadata=ChunkMetadata(source="policy_cancel.pdf", doc_type=doc_type, page=2, section="Refunds"),
        embedding=[],
    )


def _make_session(turns: list[Turn] | None = None) -> ChatSession:
    return ChatSession(
        session_id=str(uuid.uuid4()),
        created_at=datetime.utcnow(),
        last_active=datetime.utcnow(),
        turns=turns or [],
    )


def _make_retrieval(chunks: list[DocumentChunk], max_sim: float = 0.85) -> RetrievalResult:
    return RetrievalResult(chunks=chunks, max_similarity=max_sim, query_embedding=[])


def _valid_json(**overrides) -> str:
    base = {
        "answer": "You can cancel within 24 hours for a full refund.",
        "booking_link": "https://example.com/cancel",
        "related_services": ["Travel Insurance"],
        "sources": [{"doc": "policy_cancel.pdf", "page": 2}],
        "confidence": 0.9,
        "escalate_to_human": False,
    }
    base.update(overrides)
    return json.dumps(base)


# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_contains_system_prompt(self):
        prompt = build_prompt("test", [], [])
        assert "travel agency assistant" in prompt.lower()

    def test_contains_context_chunks(self):
        chunk = _make_chunk("Cancellation is free within 24h.")
        prompt = build_prompt("cancel?", [chunk], [])
        assert "Cancellation is free within 24h." in prompt
        assert "policy_cancel.pdf" in prompt
        assert "Page 2" in prompt

    def test_no_chunks_says_no_documents(self):
        prompt = build_prompt("cancel?", [], [])
        assert "No relevant documents" in prompt

    def test_history_included(self):
        turns = [
            Turn(role="user", content="Hello", timestamp=datetime.utcnow()),
            Turn(role="assistant", content="Hi there!", timestamp=datetime.utcnow()),
        ]
        prompt = build_prompt("follow up?", [], turns)
        assert "Hello" in prompt
        assert "Hi there!" in prompt

    def test_history_truncated_to_max(self):
        turns = [
            Turn(role="user", content=f"msg{i}", timestamp=datetime.utcnow())
            for i in range(20)
        ]
        prompt = build_prompt("query", [], turns, max_history_turns=3)
        # Only last 6 items (3 pairs) should appear — earliest should be gone
        assert "msg0" not in prompt

    def test_query_in_prompt(self):
        prompt = build_prompt("What is the refund policy?", [], [])
        assert "What is the refund policy?" in prompt


# ---------------------------------------------------------------------------
# Response Parser
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_valid_response_parses(self):
        result = parse_response(_valid_json())
        assert result is not None
        assert result.answer == "You can cancel within 24 hours for a full refund."
        assert result.confidence == 0.9
        assert result.escalate_to_human is False
        assert result.booking_link == "https://example.com/cancel"
        assert len(result.sources) == 1

    def test_malformed_json_returns_none(self):
        assert parse_response("not json {{{") is None

    def test_missing_required_field_returns_none(self):
        data = {"answer": "hello"}  # missing confidence and escalate_to_human
        assert parse_response(json.dumps(data)) is None

    def test_null_booking_link_allowed(self):
        result = parse_response(_valid_json(booking_link=None))
        assert result is not None
        assert result.booking_link is None

    def test_empty_json_string_returns_none(self):
        assert parse_response("") is None


class TestGuardrails:
    def test_invalid_booking_link_nullified(self):
        result = parse_response(_valid_json(booking_link="not-a-url"))
        assert result is not None
        assert result.booking_link is None

    def test_valid_http_link_kept(self):
        result = parse_response(_valid_json(booking_link="http://example.com/book"))
        assert result is not None
        assert result.booking_link == "http://example.com/book"

    def test_no_sources_long_answer_caps_confidence(self):
        long_answer = "A" * 100
        result = parse_response(_valid_json(answer=long_answer, sources=[]))
        assert result is not None
        assert result.confidence <= 0.5

    def test_no_sources_short_answer_confidence_unchanged(self):
        result = parse_response(_valid_json(answer="OK", sources=[], confidence=0.9))
        assert result is not None
        assert result.confidence == 0.9

    def test_related_services_capped_at_3(self):
        result = _apply_guardrails({
            "answer": "ok",
            "related_services": ["a", "b", "c", "d", "e"],
            "sources": [],
            "confidence": 0.8,
            "escalate_to_human": False,
        })
        assert len(result.related_services) <= 3


# ---------------------------------------------------------------------------
# Generator (LLM client patched)
# ---------------------------------------------------------------------------

class TestGenerator:
    @pytest.mark.anyio
    async def test_valid_llm_response_returns_bot_response(self):
        from src.generation.generator import Generator

        gen = Generator()
        chunks = [_make_chunk()]
        retrieval = _make_retrieval(chunks, max_sim=0.85)
        session = _make_session()

        with patch.object(gen._client, "generate", new=AsyncMock(return_value=_valid_json())):
            result = await gen.generate_response("cancel my trip?", retrieval, session)

        assert isinstance(result, BotResponse)
        assert result.confidence == 0.9
        assert result.escalate_to_human is False

    @pytest.mark.anyio
    async def test_empty_retrieval_returns_fallback(self):
        from src.generation.generator import Generator

        gen = Generator()
        retrieval = _make_retrieval([], max_sim=0.0)
        session = _make_session()

        result = await gen.generate_response("cancel?", retrieval, session)

        assert result.escalate_to_human is True
        assert result.confidence == 0.0

    @pytest.mark.anyio
    async def test_low_similarity_caps_confidence_and_escalates(self):
        from src.generation.generator import Generator

        gen = Generator()
        chunks = [_make_chunk()]
        retrieval = _make_retrieval(chunks, max_sim=0.3)  # below 0.4
        session = _make_session()

        with patch.object(gen._client, "generate", new=AsyncMock(return_value=_valid_json(confidence=0.9))):
            result = await gen.generate_response("query", retrieval, session)

        assert result.confidence <= 0.4
        assert result.escalate_to_human is True

    @pytest.mark.anyio
    async def test_malformed_json_retries_then_returns_fallback(self):
        from src.generation.generator import Generator

        gen = Generator()
        chunks = [_make_chunk()]
        retrieval = _make_retrieval(chunks)
        session = _make_session()

        with patch.object(gen._client, "generate", new=AsyncMock(return_value="BAD JSON")):
            result = await gen.generate_response("query", retrieval, session)

        assert result.escalate_to_human is True
        assert result.confidence == 0.0

    @pytest.mark.anyio
    async def test_timeout_returns_fallback(self):
        from src.generation.generator import Generator

        gen = Generator()
        chunks = [_make_chunk()]
        retrieval = _make_retrieval(chunks)
        session = _make_session()

        with patch.object(gen._client, "generate", new=AsyncMock(side_effect=asyncio.TimeoutError)):
            result = await gen.generate_response("query", retrieval, session)

        assert result is FALLBACK_RESPONSE

    @pytest.mark.anyio
    async def test_first_parse_fails_second_succeeds(self):
        from src.generation.generator import Generator

        gen = Generator()
        chunks = [_make_chunk()]
        retrieval = _make_retrieval(chunks)
        session = _make_session()

        # First call returns bad JSON, second returns valid
        mock = AsyncMock(side_effect=["BAD JSON", _valid_json()])
        with patch.object(gen._client, "generate", new=mock):
            result = await gen.generate_response("query", retrieval, session)

        assert result.confidence == 0.9
        assert mock.call_count == 2
