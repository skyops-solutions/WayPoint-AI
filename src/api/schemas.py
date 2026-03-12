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
