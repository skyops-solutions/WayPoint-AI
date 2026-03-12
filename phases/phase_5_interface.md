# Phase 5 — Chat Interface & API

## Goal
Build the FastAPI backend exposing the full RAG pipeline and a React chat widget for customer interaction. Include session management and conversation logging.

## Deliverables
- `src/api/main.py` — FastAPI app + routes
- `src/api/schemas.py` — Pydantic request/response models
- `src/chat/session_manager.py` — in-memory session store
- `src/chat/logger.py` — conversation logging to SQLite
- `frontend/` — React chat widget
- `tests/test_api.py`

## Tasks

### 5.1 FastAPI Backend

#### Routes
```python
POST /chat/session         → { session_id: str }
POST /chat                 → BotResponse
GET  /chat/session/{id}    → { turns: list[Turn] }
POST /ingest               → { status, docs_indexed, chunks_indexed }  # admin, token-protected
GET  /health               → { status: "ok", index_loaded: bool }
```

#### POST /chat request/response
```python
class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(..., min_length=1, max_length=2000)

class ChatResponse(BaseModel):
    answer: str
    booking_link: str | None
    related_services: list[str]
    sources: list[Source]
    confidence: float
    escalate_to_human: bool
    session_id: str
```

#### Pipeline integration
```python
@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    session = session_manager.get(request.session_id)
    retrieval = await retriever.retrieve(request.message)
    response = await generator.generate_response(request.message, retrieval, session)
    response = await router.route(request.message, response, session)
    session_manager.add_turn(session, request.message, response)
    await logger.log_turn(session, request.message, response, retrieval)
    return ChatResponse(**response.model_dump(), session_id=request.session_id)
```

### 5.2 Session Manager
```python
class SessionManager:
    _sessions: dict[str, ChatSession] = {}   # in-memory, MVP

    def create(self) -> ChatSession
    def get(self, session_id: str) -> ChatSession          # raises 404 if not found
    def add_turn(self, session: ChatSession, user_msg: str, response: BotResponse) -> None
    def prune_history(self, session: ChatSession, max_turns: int = 10) -> None
```

Session TTL: 30 minutes of inactivity → auto-expire (background task).

### 5.3 Conversation Logger
Log to SQLite table `conversation_logs`:

```sql
CREATE TABLE conversation_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    user_message TEXT NOT NULL,
    bot_answer  TEXT NOT NULL,
    confidence  REAL,
    escalated   INTEGER,          -- 0 or 1
    chunk_ids   TEXT,             -- JSON array of chunk_ids retrieved
    intent      TEXT
);
```

```python
class ConversationLogger:
    async def log_turn(
        self,
        session: ChatSession,
        user_message: str,
        response: BotResponse,
        retrieval: RetrievalResult
    ) -> None:
```

Log writes are fire-and-forget (don't block response).

### 5.4 React Chat Widget

**Features:**
- Message input + send button (Enter to send)
- Chat history display with user/bot message bubbles
- Typing indicator while awaiting response
- Structured bot response rendering:
  - Answer text
  - "Book Now" button if `booking_link` present
  - Related services chips/tags
  - Source attribution (collapsible)
  - "Speaking with agent" banner if `escalate_to_human=true`
- Auto-scroll to latest message
- Session ID persisted in `sessionStorage` (lost on tab close = new session)

**Tech stack:** React + TypeScript, Tailwind CSS, no heavy UI framework (keep it embeddable).

**File structure:**
```
frontend/
├── src/
│   ├── components/
│   │   ├── ChatWidget.tsx      # root component
│   │   ├── MessageList.tsx
│   │   ├── MessageBubble.tsx
│   │   ├── BotResponse.tsx     # structured response renderer
│   │   └── InputBar.tsx
│   ├── hooks/
│   │   └── useChat.ts          # API calls, session state
│   ├── types.ts
│   └── main.tsx
├── index.html
├── package.json
└── vite.config.ts
```

**API base URL** configurable via `VITE_API_URL` env var.

### 5.5 CORS & Security
- Allow origins: configurable via `ALLOWED_ORIGINS` env var
- `/ingest` endpoint protected by `Authorization: Bearer {ADMIN_TOKEN}` header
- Rate limiting: 30 requests/minute per IP (use `slowapi`)
- Max message length: 2000 chars (validated in Pydantic schema)

## Acceptance Criteria
- [ ] `POST /chat` returns valid `ChatResponse` in < 3s p95
- [ ] Session history returned correctly by `GET /chat/session/{id}`
- [ ] `/ingest` returns 401 without valid token
- [ ] Frontend renders answer, booking link button, related services, and escalation banner
- [ ] Conversation turns logged to SQLite after each message
- [ ] Session expires after 30 min inactivity
- [ ] Unit tests cover all API routes (happy path + error cases)

## Dependencies
```
# Backend
fastapi>=0.111
uvicorn[standard]>=0.29
pydantic>=2.0
slowapi>=0.1
aiosqlite>=0.20

# Frontend
react 18
typescript
tailwindcss
vite
```
