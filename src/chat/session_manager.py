"""Phase 5 — Session Manager. See phases/phase_5_interface.md §5.2"""
import uuid
from datetime import datetime, timedelta

from src.models import BotResponse, ChatSession, Turn


SESSION_TTL_MINUTES = 30
MAX_TURNS = 10


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}

    def create(self) -> ChatSession:
        session = ChatSession(session_id=str(uuid.uuid4()))
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> ChatSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        session.last_active = datetime.utcnow()
        return session

    def add_turn(
        self, session: ChatSession, user_msg: str, response: BotResponse
    ) -> None:
        session.turns.append(Turn(role="user", content=user_msg))
        session.turns.append(
            Turn(
                role="assistant",
                content=response.answer,
                retrieved_chunk_ids=None,
                confidence=response.confidence,
            )
        )
        self._prune(session)

    def _prune(self, session: ChatSession) -> None:
        if len(session.turns) > MAX_TURNS * 2:
            session.turns = session.turns[-(MAX_TURNS * 2):]

    def expire_stale(self) -> None:
        cutoff = datetime.utcnow() - timedelta(minutes=SESSION_TTL_MINUTES)
        stale = [sid for sid, s in self._sessions.items() if s.last_active < cutoff]
        for sid in stale:
            del self._sessions[sid]
