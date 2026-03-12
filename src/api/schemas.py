from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(..., min_length=1, max_length=2000)


class Source(BaseModel):
    doc: str
    page: int


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    booking_link: str | None = None
    related_services: list[str] = []
    sources: list[Source] = []
    confidence: float
    escalate_to_human: bool


class SessionResponse(BaseModel):
    session_id: str


class TurnOut(BaseModel):
    role: str
    content: str
    confidence: float | None = None


class SessionHistoryResponse(BaseModel):
    session_id: str
    turns: list[TurnOut]


class IngestResponse(BaseModel):
    status: str
    docs_indexed: int
    chunks_indexed: int


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------

class ConversationSummary(BaseModel):
    session_id: str
    first_seen: str
    last_seen: str
    turn_count: int
    escalated: bool
    last_message: str


class AdminTurn(BaseModel):
    timestamp: str
    user_message: str
    bot_answer: str
    confidence: float | None = None
    escalated: bool


class ConversationDetail(BaseModel):
    session_id: str
    turns: list[AdminTurn]


class AdminStats(BaseModel):
    total_conversations: int
    total_turns: int
    escalated_count: int
    avg_confidence: float


class ConversationsResponse(BaseModel):
    items: list[ConversationSummary]
    total: int
    page: int
    page_size: int
