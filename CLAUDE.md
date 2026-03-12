# Travel Agency AI Chatbot — Claude Code Guidelines

## Project Overview
RAG-based conversational AI chatbot for a travel agency. Provides structured answers to customer inquiries using internal documentation (PDFs, FAQs, service guides). Built with Google Gemini (`gemini-2.5-flash`) and a vector database for semantic retrieval.

## Architecture
```
User → Chat Interface → RAG Pipeline → LLM Response
                           ↓
               Vector DB (FAISS/Pinecone) ← Document Ingestion
```

**Stack:**
- LLM: `google-genai` (`gemini-2.5-flash`)
- Embeddings: Google Generative AI Embeddings
- Vector DB: FAISS (local MVP) → Pinecone (production)
- Backend: Python (FastAPI)
- Frontend: React chat widget
- Document parsing: `pypdf`, `unstructured`

## Key Conventions

### Python
- Use `async/await` throughout — FastAPI + async Gemini client
- Type hints required on all function signatures
- Config via `.env` + `pydantic-settings` — never hardcode API keys
- Raise domain-specific exceptions, not bare `Exception`

### RAG Pipeline
- Chunk size: 512 tokens, overlap: 64 tokens
- Every chunk must carry metadata: `source`, `page`, `section`, `doc_type`
- Retrieval returns top-K=5 chunks; rerank before passing to LLM
- Always include source citations in generated responses

### Response Format
Every LLM response must follow this structure (enforced via prompt template):
```json
{
  "answer": "...",
  "booking_link": "...",
  "related_services": ["...", "..."],
  "confidence": 0.0-1.0,
  "escalate_to_human": false
}
```

### Escalation Logic
- If `confidence < 0.6` → set `escalate_to_human: true`
- If intent = `complaint` or `legal` → always escalate
- Log all escalations with full conversation context

## File Structure
```
/
├── CLAUDE.md
├── project_specs.md
├── phases/
│   ├── phase_1_ingestion.md
│   ├── phase_2_retrieval.md
│   ├── phase_3_generation.md
│   ├── phase_4_routing.md
│   └── phase_5_interface.md
├── src/
│   ├── ingestion/       # Document parsing, chunking, indexing
│   ├── retrieval/       # Embedding, vector search, reranking
│   ├── generation/      # Prompt templates, LLM client, response parsing
│   ├── routing/         # Intent detection, escalation logic
│   ├── api/             # FastAPI routes
│   └── chat/            # Conversation state management
├── frontend/            # React chat widget
├── docs/                # Internal travel agency documents (source data)
├── tests/
└── scripts/             # Ingestion runner, eval scripts
```

## Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run document ingestion
python scripts/ingest.py --docs-dir ./docs

# Start API server
uvicorn src.api.main:app --reload

# Run tests
pytest tests/ -v

# Run frontend
cd frontend && npm run dev
```

## Environment Variables
```
GOOGLE_API_KEY=
PINECONE_API_KEY=         # optional for MVP, use FAISS locally
PINECONE_INDEX_NAME=
HUMAN_SUPPORT_WEBHOOK=    # endpoint to notify human agents
LOG_LEVEL=INFO
```

## What NOT to Do
- Never pass raw user input directly to the LLM without retrieval context
- Never skip metadata when indexing chunks — it's required for citations
- Never hardcode document paths — use config
- Do not add mock/stub retrieval in tests; use a test FAISS index with real embeddings
