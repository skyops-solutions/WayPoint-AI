# WayPoint AI — Intelligent Travel Agency Chatbot

A production-ready RAG-powered chatbot that lets travel agency customers instantly get answers about tours, bookings, cancellations, and destinations — grounded in the agency's own internal documents.

Built as a full-stack MVP: FastAPI backend + React chat widget, deployable as a standalone service or embeddable widget.

---

## What It Does

- Answers customer questions using the agency's real documents (PDFs, Markdown, TXT)
- Returns structured responses: answer + confidence score + booking link + related services
- Automatically escalates complex cases (complaints, legal questions, low confidence) to a human agent
- Maintains conversation history per session
- Admin endpoint to re-index documents without restarting the server

---

## Architecture

```
User Message
    │
    ▼
FastAPI Backend
    │
    ├── Retriever ──► FAISS Vector Index ──► Top-K relevant chunks
    │                  (Gemini Embeddings)
    │
    ├── Generator ──► Gemini 2.5 Flash ──► Structured JSON response
    │                  (RAG prompt with context)
    │
    └── Router ────► Intent detection ──► Escalate or respond
                      Escalation rules
```

**Stack:**
- **LLM:** Google Gemini 2.5 Flash (`google-genai`)
- **Embeddings:** Gemini Embedding 001 (3072-dim vectors)
- **Vector DB:** FAISS `IndexFlatIP` with L2-normalisation (cosine similarity)
- **Backend:** FastAPI, Pydantic v2, slowapi rate limiting, SQLite conversation log
- **Frontend:** React 18 + TypeScript + Tailwind CSS + Vite
- **Tests:** 137 tests across all 5 phases (pytest + anyio)

---

## Project Structure

```
├── src/
│   ├── ingestion/        # Document parsing, chunking, embedding, FAISS indexing
│   ├── retrieval/        # Vector search, reranking, query embedding
│   ├── generation/       # Prompt builder, Gemini client, response parser
│   ├── routing/          # Intent detection, escalation rules, webhook
│   ├── chat/             # Session manager, SQLite conversation logger
│   ├── api/              # FastAPI app, routes, schemas
│   └── config.py         # Pydantic settings (env-based)
├── frontend/             # React + TypeScript chat widget
├── docs/                 # Travel agency knowledge base (your documents go here)
├── scripts/
│   └── ingest.py         # CLI to index documents into FAISS
├── tests/                # 137 tests, all passing
└── data/                 # Generated: FAISS index, embedding cache, SQLite DB
```

---

## Key Features

**RAG Pipeline**
- Recursive character text splitting (512 tokens, 64 overlap)
- Batch embedding with exponential backoff on rate limits
- SHA-256 disk cache — unchanged documents are never re-embedded
- FAISS vector search with similarity threshold filtering
- Reranker: vector similarity (70%) + recency/doc-type boost (30%), Jaccard deduplication

**Response Generation**
- Forced JSON output via `response_mime_type="application/json"` + jsonschema validation
- Guardrails: invalid booking links nullified, low-context confidence capped, `related_services` capped at 3
- Fallback response with `escalate_to_human: true` on parse failure

**Escalation Engine (5 rules)**
1. Always escalate: complaints, legal disputes, explicit human requests
2. Human keywords detected in user message
3. Confidence below threshold (< 0.6)
4. Repeated low-confidence turns (3 in a row)
5. Conditional: refund / accessibility queries with confidence < 0.75

**Chat Widget**
- Session persistence via `sessionStorage`
- Typing indicator, auto-scroll, Enter to send / Shift+Enter for newline
- Collapsible sources, booking button, escalation banner, related service chips

---

## Quick Start

**1. Clone and install**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Configure**
```bash
cp .env.example .env
# Add your GOOGLE_API_KEY and ADMIN_TOKEN
```

**3. Add documents and index**

Place `.pdf`, `.md`, or `.txt` files in `docs/`. File name prefix determines document type:
- `faq_*` → FAQ
- `policy_*` → Policy
- `dest_*` → Destination
- `guide_*` → Travel guide

```bash
PYTHONPATH=. python scripts/ingest.py
```

**4. Run backend**
```bash
PYTHONPATH=. uvicorn src.api.main:app --reload
# http://localhost:8000
```

**5. Run frontend**
```bash
cd frontend
npm install
npm run dev
# http://localhost:5173
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat/session` | Create a new chat session |
| `POST` | `/chat` | Send a message, get AI response |
| `GET` | `/chat/session/{id}` | Fetch conversation history |
| `POST` | `/ingest` | Re-index documents (admin token required) |
| `GET` | `/health` | Health check + index status |

**Chat request:**
```json
POST /chat
{
  "session_id": "uuid",
  "message": "What is your cancellation policy?"
}
```

**Chat response:**
```json
{
  "session_id": "uuid",
  "answer": "You can cancel up to 60 days before...",
  "booking_link": "https://...",
  "related_services": ["Travel Insurance"],
  "sources": [{"doc": "policy_cancellation.md", "page": 2}],
  "confidence": 0.92,
  "escalate_to_human": false
}
```

---

## Running Tests

```bash
PYTHONPATH=. pytest tests/ -v
# 137 passed
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes | Google AI Studio API key |
| `ADMIN_TOKEN` | Yes | Bearer token for `/ingest` endpoint |
| `ALLOWED_ORIGINS` | No | CORS origins (default: `http://localhost:5173`) |
| `DOCS_DIR` | No | Path to documents folder (default: `./docs`) |
| `INDEX_DIR` | No | Path to FAISS index (default: `./data/index`) |
| `DB_PATH` | No | SQLite database path (default: `./data/conversations.db`) |
| `HUMAN_SUPPORT_WEBHOOK` | No | Webhook URL for escalation notifications |
