"""Phase 5 — Conversation Logger. See phases/phase_5_interface.md §5.3"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from src.config import settings
from src.models import BotResponse, ChatSession, RetrievalResult

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS conversation_logs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    timestamp    TEXT NOT NULL,
    user_message TEXT NOT NULL,
    bot_answer   TEXT NOT NULL,
    confidence   REAL,
    escalated    INTEGER,
    chunk_ids    TEXT,
    intent       TEXT
);
"""


class ConversationLogger:
    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or settings.db_path

    async def init_db(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(_CREATE_TABLE)
            await db.commit()
        logger.info("Conversation DB ready at %s", self._db_path)

    async def log_turn(
        self,
        session: ChatSession,
        user_message: str,
        response: BotResponse,
        retrieval: RetrievalResult,
    ) -> None:
        """Write one turn to the DB. Errors are suppressed — never block response."""
        chunk_ids = json.dumps([c.chunk_id for c in retrieval.chunks])
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    """
                    INSERT INTO conversation_logs
                        (session_id, timestamp, user_message, bot_answer,
                         confidence, escalated, chunk_ids, intent)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session.session_id,
                        datetime.now(timezone.utc).isoformat(),
                        user_message,
                        response.answer,
                        response.confidence,
                        int(response.escalate_to_human),
                        chunk_ids,
                        None,  # intent stored by router in Phase 4; placeholder here
                    ),
                )
                await db.commit()
        except Exception as exc:
            logger.error("Failed to log conversation turn: %s", exc)
