# Project Specifications — Travel Agency AI Chatbot

## Goal
Build an MVP conversational assistant for a travel agency that answers customer questions using internal documentation, generates structured responses, and routes complex cases to human support.

---

## Functional Requirements

### FR-1: Question Answering
- Answer queries about: bookings, cancellations, itinerary changes, destinations, travel services, policies
- Responses must be grounded in retrieved internal documents — no hallucinated facts
- Response format: `answer` + `booking_link` + `related_services` (see Response Schema)

### FR-2: Document Knowledge Base
Supported source types:
| Type | Format | Examples |
|------|--------|---------|
| FAQs | PDF, Markdown | General Q&A |
| Service guides | PDF | Visa requirements, insurance |
| Policy docs | PDF, DOCX | Cancellation, refund policies |
| Destination guides | PDF, Markdown | Country/city information |

### FR-3: Structured Responses
Every response must include:
```json
{
  "answer": "string — clear, concise answer",
  "booking_link": "string | null — relevant action URL",
  "related_services": ["string"] — 0-3 upsell/related suggestions",
  "sources": [{"doc": "string", "page": int}],
  "confidence": float,
  "escalate_to_human": boolean
}
```

### FR-4: Human Escalation
Trigger escalation when:
- Confidence score < 0.6
- Intent classified as: `complaint`, `legal_dispute`, `refund_request` (complex), `accessibility_need`
- User explicitly requests human agent
- 3+ consecutive low-confidence turns in the same session

Escalation action: fire webhook to `HUMAN_SUPPORT_WEBHOOK` with session ID, full transcript, and detected intent.

### FR-5: Conversational Context
- Maintain context for up to 10 turns per session
- Support follow-up questions referencing prior turns ("What about the cancellation fee for that?")
- Session state stored in-memory (MVP); Redis for production

### FR-6: Conversation Logging
Log per message:
- `session_id`, `timestamp`, `user_message`, `bot_response`, `retrieved_chunks`, `confidence`, `escalated`
- Store in SQLite (MVP) → PostgreSQL (production)

---

## Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| Response latency | < 3s p95 |
| Retrieval accuracy | > 80% top-5 recall on internal eval set |
| Uptime | 99.5% |
| Concurrent users | 50 (MVP), 500 (production) |
| Document coverage | All provided PDFs and FAQs indexed |

---

## Technical Architecture

### Component Overview
```
┌─────────────┐    ┌──────────────────────────────────────────┐
│  Chat UI    │───▶│              FastAPI Backend              │
│  (React)    │◀───│                                           │
└─────────────┘    │  ┌──────────┐  ┌────────┐  ┌─────────┐  │
                   │  │ Session  │  │  RAG   │  │ Router  │  │
                   │  │ Manager  │  │Pipeline│  │ /Escalat│  │
                   │  └──────────┘  └────────┘  └─────────┘  │
                   └────────────────────┬─────────────────────┘
                                        │
                   ┌────────────────────▼─────────────────────┐
                   │              RAG Pipeline                 │
                   │                                           │
                   │  ┌──────────┐  ┌──────────┐  ┌───────┐  │
                   │  │ Embedder │  │  Vector  │  │  LLM  │  │
                   │  │ (Gemini) │  │   DB     │  │Gemini │  │
                   │  └──────────┘  │  (FAISS) │  │2.5-fl.│  │
                   │                └──────────┘  └───────┘  │
                   └───────────────────────────────────────────┘
```

### RAG Pipeline Detail
1. **Ingestion** (offline)
   - Parse PDFs/Markdown → extract text + metadata
   - Chunk: 512 tokens, 64-token overlap, sentence-boundary aware
   - Embed chunks via `google-genai` embeddings model
   - Store vectors + metadata in FAISS index (persisted to disk)

2. **Retrieval** (online, per query)
   - Embed user query
   - ANN search → top-20 candidates
   - Rerank by relevance score → top-5 context chunks
   - Filter by metadata if intent is domain-specific (e.g., only cancellation docs)

3. **Generation** (online)
   - Build prompt: system instructions + retrieved context + conversation history + user query
   - Call `gemini-2.5-flash` with structured output schema
   - Parse and validate response JSON
   - If parsing fails → retry once, then return fallback escalation response

---

## Data Model

### Document Chunk
```python
@dataclass
class DocumentChunk:
    chunk_id: str           # uuid
    doc_id: str             # source document identifier
    content: str            # text content
    embedding: list[float]  # vector
    metadata: ChunkMetadata

@dataclass
class ChunkMetadata:
    source: str             # filename
    doc_type: str           # faq | policy | guide | destination
    page: int
    section: str | None
    language: str           # "en" default
```

### Chat Session
```python
@dataclass
class ChatSession:
    session_id: str
    created_at: datetime
    turns: list[Turn]       # max 10 kept in context
    escalated: bool

@dataclass
class Turn:
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime
    retrieved_chunks: list[str] | None   # chunk_ids, assistant turns only
    confidence: float | None
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Send message, get structured response |
| `POST` | `/chat/session` | Create new session |
| `GET`  | `/chat/session/{id}` | Get session history |
| `POST` | `/ingest` | Trigger document re-ingestion (admin) |
| `GET`  | `/health` | Health check |

### POST /chat
**Request:**
```json
{
  "session_id": "string",
  "message": "string"
}
```
**Response:** See FR-3 Response Schema above.

---

## Prompt Design

### System Prompt
```
You are a helpful travel agency assistant. Answer customer questions using ONLY the provided context documents.
If the context does not contain enough information to answer confidently, say so — do not fabricate details.
Always respond in the following JSON format: { answer, booking_link, related_services, sources, confidence, escalate_to_human }.
Confidence should reflect how well the retrieved context supports the answer (0.0 = no support, 1.0 = fully supported).
```

### Context Template
```
[CONTEXT]
{retrieved_chunks_with_sources}

[CONVERSATION HISTORY]
{last_n_turns}

[USER QUESTION]
{user_message}
```

---

## Evaluation Criteria (MVP Acceptance)
- [ ] All provided documents are indexed and retrievable
- [ ] 10 sample Q&A pairs answered correctly (manually verified)
- [ ] Escalation fires correctly on low-confidence and flagged intents
- [ ] Conversation context maintained across 5-turn test dialogue
- [ ] Response latency < 3s on p95 locally
- [ ] Structured JSON response schema validated on 100% of responses
