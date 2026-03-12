"""Tests for Phase 5 — FastAPI routes."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.models import BotResponse, ChatSession, RetrievalResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_bot_response(escalate: bool = False) -> BotResponse:
    return BotResponse(
        answer="You can cancel within 24 hours.",
        booking_link="https://example.com/cancel",
        related_services=["Travel Insurance"],
        sources=[],
        confidence=0.9,
        escalate_to_human=escalate,
    )


def _make_retrieval() -> RetrievalResult:
    return RetrievalResult(chunks=[], max_similarity=0.85, query_embedding=[])


@pytest.fixture
def mock_pipeline(monkeypatch):
    """Patch all pipeline components on app.state before each test."""
    from src.api.main import app

    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=_make_retrieval())

    generator = MagicMock()
    generator.generate_response = AsyncMock(return_value=_make_bot_response())

    router = MagicMock()
    router.route = AsyncMock(return_value=_make_bot_response())

    conv_logger = MagicMock()
    conv_logger.log_turn = AsyncMock()
    conv_logger.init_db = AsyncMock()

    app.state.retriever = retriever
    app.state.generator = generator
    app.state.router = router
    app.state.conv_logger = conv_logger
    app.state.index_loaded = True

    return {
        "retriever": retriever,
        "generator": generator,
        "router": router,
        "conv_logger": conv_logger,
    }


@pytest.fixture
async def client(mock_pipeline):
    from src.api.main import app
    from src.chat.session_manager import SessionManager

    app.state.session_manager = SessionManager()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    @pytest.mark.anyio
    async def test_health_ok(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "index_loaded" in data


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class TestSession:
    @pytest.mark.anyio
    async def test_create_session_returns_id(self, client):
        resp = await client.post("/chat/session")
        assert resp.status_code == 201
        data = resp.json()
        assert "session_id" in data
        assert len(data["session_id"]) > 0

    @pytest.mark.anyio
    async def test_get_session_history_empty(self, client):
        create = await client.post("/chat/session")
        sid = create.json()["session_id"]
        resp = await client.get(f"/chat/session/{sid}")
        assert resp.status_code == 200
        assert resp.json()["turns"] == []

    @pytest.mark.anyio
    async def test_get_session_not_found(self, client):
        resp = await client.get("/chat/session/nonexistent-id")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------

class TestChat:
    @pytest.mark.anyio
    async def test_chat_returns_valid_response(self, client, mock_pipeline):
        create = await client.post("/chat/session")
        sid = create.json()["session_id"]

        resp = await client.post("/chat", json={"session_id": sid, "message": "Can I cancel?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "You can cancel within 24 hours."
        assert data["session_id"] == sid
        assert "confidence" in data
        assert "escalate_to_human" in data

    @pytest.mark.anyio
    async def test_chat_session_not_found(self, client):
        resp = await client.post(
            "/chat", json={"session_id": "bad-id", "message": "hello"}
        )
        assert resp.status_code == 404

    @pytest.mark.anyio
    async def test_chat_message_too_long(self, client):
        create = await client.post("/chat/session")
        sid = create.json()["session_id"]
        resp = await client.post(
            "/chat", json={"session_id": sid, "message": "x" * 2001}
        )
        assert resp.status_code == 422  # Pydantic validation

    @pytest.mark.anyio
    async def test_chat_empty_message_rejected(self, client):
        create = await client.post("/chat/session")
        sid = create.json()["session_id"]
        resp = await client.post("/chat", json={"session_id": sid, "message": ""})
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_chat_wires_full_pipeline(self, client, mock_pipeline):
        create = await client.post("/chat/session")
        sid = create.json()["session_id"]
        await client.post("/chat", json={"session_id": sid, "message": "hello"})

        mock_pipeline["retriever"].retrieve.assert_called_once()
        mock_pipeline["generator"].generate_response.assert_called_once()
        mock_pipeline["router"].route.assert_called_once()
        mock_pipeline["conv_logger"].log_turn.assert_called_once()

    @pytest.mark.anyio
    async def test_chat_history_populated_after_turn(self, client, mock_pipeline):
        create = await client.post("/chat/session")
        sid = create.json()["session_id"]
        await client.post("/chat", json={"session_id": sid, "message": "hi"})

        hist = await client.get(f"/chat/session/{sid}")
        turns = hist.json()["turns"]
        assert len(turns) == 2  # user + assistant
        assert turns[0]["role"] == "user"
        assert turns[1]["role"] == "assistant"

    @pytest.mark.anyio
    async def test_chat_escalation_flag_passed_through(self, client, mock_pipeline):
        mock_pipeline["router"].route = AsyncMock(
            return_value=_make_bot_response(escalate=True)
        )
        create = await client.post("/chat/session")
        sid = create.json()["session_id"]
        resp = await client.post("/chat", json={"session_id": sid, "message": "complaint"})
        assert resp.json()["escalate_to_human"] is True


# ---------------------------------------------------------------------------
# POST /ingest (admin)
# ---------------------------------------------------------------------------

class TestIngest:
    @pytest.mark.anyio
    async def test_ingest_requires_token(self, client):
        resp = await client.post("/ingest")
        assert resp.status_code == 401  # no credentials → HTTPBearer returns 401

    @pytest.mark.anyio
    async def test_ingest_wrong_token_returns_401(self, client):
        resp = await client.post(
            "/ingest", headers={"Authorization": "Bearer wrong-token"}
        )
        assert resp.status_code == 401

    @pytest.mark.anyio
    async def test_ingest_valid_token_accepted(self, client, tmp_path):
        from src.config import settings
        from src.api.main import app
        from src.ingestion.indexer import FAISSIndex
        from src.retrieval.vector_search import VectorSearch
        from src.retrieval.retriever import Retriever

        # Patch retriever with a real FAISSIndex (no docs → returns no_documents)
        index = FAISSIndex()
        retriever = MagicMock()
        retriever._search = MagicMock()
        retriever._search._index = index
        app.state.retriever = retriever

        with patch.object(settings, "docs_dir", str(tmp_path)):
            resp = await client.post(
                "/ingest",
                headers={"Authorization": f"Bearer {settings.admin_token}"},
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "no_documents"
