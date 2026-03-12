from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


# ---------------------------------------------------------------------------
# Document / Ingestion
# ---------------------------------------------------------------------------

@dataclass
class ChunkMetadata:
    source: str
    doc_type: str           # faq | policy | guide | destination
    page: int
    section: str | None = None
    language: str = "en"


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    content: str
    metadata: ChunkMetadata
    embedding: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    chunks: list[DocumentChunk]
    max_similarity: float
    query_embedding: list[float]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@dataclass
class Source:
    doc: str
    page: int


@dataclass
class BotResponse:
    answer: str
    confidence: float
    escalate_to_human: bool
    booking_link: str | None = None
    related_services: list[str] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)

    def model_dump(self) -> dict:
        return {
            "answer": self.answer,
            "booking_link": self.booking_link,
            "related_services": self.related_services,
            "sources": [{"doc": s.doc, "page": s.page} for s in self.sources],
            "confidence": self.confidence,
            "escalate_to_human": self.escalate_to_human,
        }


# ---------------------------------------------------------------------------
# Chat / Session
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    retrieved_chunk_ids: list[str] | None = None
    confidence: float | None = None


@dataclass
class ChatSession:
    session_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)
    turns: list[Turn] = field(default_factory=list)
    escalated: bool = False
