from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    IngestResponse,
    SessionHistoryResponse,
    SessionResponse,
    TurnOut,
)
from src.chat.logger import ConversationLogger
from src.chat.session_manager import SessionManager
from src.config import settings
from src.generation.generator import Generator
from src.ingestion.indexer import FAISSIndex
from src.retrieval.retriever import Retriever
from src.routing.router import Router

logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)
bearer = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    app.state.index_loaded = False

    # --- Initialise pipeline components ---
    index = FAISSIndex()
    index_path = Path(settings.index_dir)
    if (index_path / "index.faiss").exists():
        try:
            index.load(index_path)
            app.state.index_loaded = True
            logger.info("FAISS index loaded (%d vectors)", index.total_vectors)
        except Exception as exc:
            logger.warning("Could not load index: %s — starting with empty index", exc)
    else:
        logger.warning("No index found at %s — ingest documents first", index_path)

    app.state.retriever = Retriever(index)
    app.state.generator = Generator()
    app.state.router = Router()
    app.state.session_manager = SessionManager()
    app.state.conv_logger = ConversationLogger()
    await app.state.conv_logger.init_db()

    # Background task: expire stale sessions every 5 minutes
    async def _expire_loop() -> None:
        while True:
            await asyncio.sleep(300)
            app.state.session_manager.expire_stale()

    expire_task = asyncio.create_task(_expire_loop())
    logger.info("Travel Agency Chatbot started.")

    yield

    expire_task.cancel()
    logger.info("Shutting down.")


app = FastAPI(title="Travel Agency AI Chatbot", version="0.1.0", lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def verify_admin(credentials: HTTPAuthorizationCredentials = Security(bearer)) -> None:
    if credentials.credentials != settings.admin_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health(request: Request) -> dict:
    return {"status": "ok", "index_loaded": request.app.state.index_loaded}


@app.post("/chat/session", response_model=SessionResponse, status_code=201)
@limiter.limit("30/minute")
async def create_session(request: Request) -> SessionResponse:
    session = request.app.state.session_manager.create()
    return SessionResponse(session_id=session.session_id)


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    sm: SessionManager = request.app.state.session_manager
    try:
        session = sm.get(body.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    retrieval = await request.app.state.retriever.retrieve(body.message)
    response = await request.app.state.generator.generate_response(body.message, retrieval, session)
    response = await request.app.state.router.route(body.message, response, session)
    sm.add_turn(session, body.message, response)
    await request.app.state.conv_logger.log_turn(session, body.message, response, retrieval)

    return ChatResponse(
        session_id=body.session_id,
        answer=response.answer,
        booking_link=response.booking_link,
        related_services=response.related_services,
        sources=[{"doc": s.doc, "page": s.page} for s in response.sources],
        confidence=response.confidence,
        escalate_to_human=response.escalate_to_human,
    )


@app.get("/chat/session/{session_id}", response_model=SessionHistoryResponse)
async def get_session(session_id: str, request: Request) -> SessionHistoryResponse:
    sm: SessionManager = request.app.state.session_manager
    try:
        session = sm.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionHistoryResponse(
        session_id=session_id,
        turns=[
            TurnOut(role=t.role, content=t.content, confidence=t.confidence)
            for t in session.turns
        ],
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: Request, _: None = Depends(verify_admin)) -> IngestResponse:
    """Trigger document re-ingestion (admin only). Runs synchronously for MVP."""
    from src.ingestion.chunker import chunk_document, infer_doc_type
    from src.ingestion.embedder import embed_chunks
    from src.ingestion.parser import parse_document

    docs_dir = Path(settings.docs_dir)
    index_dir = Path(settings.index_dir)
    supported = {".pdf", ".md", ".txt"}

    doc_files = [p for p in docs_dir.rglob("*") if p.suffix.lower() in supported]
    if not doc_files:
        return IngestResponse(status="no_documents", docs_indexed=0, chunks_indexed=0)

    index: FAISSIndex = request.app.state.retriever._search._index
    all_chunks = []
    docs_indexed = 0

    for doc_path in doc_files:
        if doc_path.stem in index.indexed_doc_ids:
            continue
        try:
            raw = parse_document(doc_path)
            chunks = chunk_document(raw, infer_doc_type(doc_path.name))
            all_chunks.extend(chunks)
            docs_indexed += 1
        except Exception as exc:
            logger.error("Ingest error for %s: %s", doc_path.name, exc)

    if all_chunks:
        await embed_chunks(all_chunks)
        index.add(all_chunks)
        index.save(index_dir)
        request.app.state.index_loaded = True

    return IngestResponse(
        status="ok",
        docs_indexed=docs_indexed,
        chunks_indexed=len(all_chunks),
    )


# ---------------------------------------------------------------------------
# Admin routes
# ---------------------------------------------------------------------------

@app.get("/admin/stats")
async def admin_stats(
    request: Request,
    _: None = Depends(verify_admin),
) -> dict:
    import aiosqlite
    db_path = request.app.state.conv_logger._db_path
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT COUNT(DISTINCT session_id), COUNT(*), SUM(escalated), AVG(confidence) FROM conversation_logs"
        ) as cur:
            row = await cur.fetchone()
    total_conv, total_turns, escalated, avg_conf = row
    return {
        "total_conversations": total_conv or 0,
        "total_turns": total_turns or 0,
        "escalated_count": int(escalated or 0),
        "avg_confidence": round(float(avg_conf or 0), 3),
    }


@app.get("/admin/conversations")
async def admin_conversations(
    request: Request,
    _: None = Depends(verify_admin),
    page: int = 1,
    page_size: int = 20,
    escalated_only: bool = False,
    search: str = "",
) -> dict:
    import aiosqlite
    db_path = request.app.state.conv_logger._db_path
    where = "WHERE escalated=1" if escalated_only else "WHERE 1=1"
    params: list = []
    if search:
        where += " AND (user_message LIKE ? OR bot_answer LIKE ?)"
        params += [f"%{search}%", f"%{search}%"]

    offset = (page - 1) * page_size

    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            f"SELECT COUNT(DISTINCT session_id) FROM conversation_logs {where}", params
        ) as cur:
            total = (await cur.fetchone())[0]

        async with db.execute(
            f"""
            SELECT session_id,
                   MIN(timestamp) as first_seen,
                   MAX(timestamp) as last_seen,
                   COUNT(*) as turn_count,
                   MAX(escalated) as escalated,
                   (SELECT user_message FROM conversation_logs c2
                    WHERE c2.session_id = c1.session_id
                    ORDER BY timestamp DESC LIMIT 1) as last_message
            FROM conversation_logs c1
            {where}
            GROUP BY session_id
            ORDER BY last_seen DESC
            LIMIT ? OFFSET ?
            """,
            params + [page_size, offset],
        ) as cur:
            rows = await cur.fetchall()

    items = [
        {
            "session_id": r[0],
            "first_seen": r[1],
            "last_seen": r[2],
            "turn_count": r[3],
            "escalated": bool(r[4]),
            "last_message": (r[5] or "")[:120],
        }
        for r in rows
    ]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@app.get("/admin/conversations/{session_id}")
async def admin_conversation_detail(
    session_id: str,
    request: Request,
    _: None = Depends(verify_admin),
) -> dict:
    import aiosqlite
    db_path = request.app.state.conv_logger._db_path
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT timestamp, user_message, bot_answer, confidence, escalated "
            "FROM conversation_logs WHERE session_id=? ORDER BY timestamp ASC",
            (session_id,),
        ) as cur:
            rows = await cur.fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="Session not found")
    turns = [
        {
            "timestamp": r[0],
            "user_message": r[1],
            "bot_answer": r[2],
            "confidence": r[3],
            "escalated": bool(r[4]),
        }
        for r in rows
    ]
    return {"session_id": session_id, "turns": turns}
