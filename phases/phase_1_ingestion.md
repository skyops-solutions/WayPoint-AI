# Phase 1 — Document Ingestion Pipeline

## Goal
Build an offline pipeline that parses travel agency documents, chunks them, generates embeddings, and stores them in a vector index ready for retrieval.

## Deliverables
- `src/ingestion/parser.py` — PDF/Markdown text extraction
- `src/ingestion/chunker.py` — text splitting with metadata
- `src/ingestion/embedder.py` — Google Generative AI embeddings
- `src/ingestion/indexer.py` — FAISS index builder/loader
- `scripts/ingest.py` — CLI runner
- `tests/test_ingestion.py`

## Tasks

### 1.1 Document Parser
- Use `pypdf` for PDF text extraction
- Use `python-docx` for DOCX (if needed)
- Preserve page numbers in metadata
- Extract section headings where detectable
- Normalize whitespace, strip headers/footers

```python
# Expected interface
def parse_document(path: Path) -> list[RawChunk]:
    """Returns list of (text, metadata) pairs, one per page/section."""
```

### 1.2 Text Chunker
- Chunk strategy: recursive character split, 512 tokens, 64-token overlap
- Respect sentence boundaries — do not cut mid-sentence
- Assign `chunk_id` (UUID), propagate source metadata
- Tag `doc_type` based on filename convention:
  - `faq_*.pdf` → `faq`
  - `policy_*.pdf` → `policy`
  - `guide_*.pdf` → `guide`
  - `dest_*.pdf` → `destination`

```python
def chunk_document(raw_chunks: list[RawChunk], doc_type: str) -> list[DocumentChunk]:
```

### 1.3 Embedder
- Model: `models/text-embedding-004` via `google-genai`
- Batch embed: 100 chunks per API call
- Retry with exponential backoff on rate limits
- Cache embeddings to disk to avoid re-embedding unchanged docs

```python
async def embed_chunks(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """Fills chunk.embedding field in place."""
```

### 1.4 FAISS Indexer
- Use `faiss.IndexFlatIP` (inner product / cosine after L2-norm)
- Store index + metadata JSON side-by-side: `index.faiss`, `index_meta.json`
- Support incremental updates (add new docs without full rebuild)
- Expose `save()` / `load()` for persistence

```python
class FAISSIndex:
    def add(self, chunks: list[DocumentChunk]) -> None
    def save(self, path: Path) -> None
    def load(self, path: Path) -> None
```

### 1.5 Ingestion CLI
```bash
python scripts/ingest.py --docs-dir ./docs --index-dir ./data/index --force-rebuild
```
- `--force-rebuild` drops and rebuilds index from scratch
- Default: incremental (skip already-indexed doc IDs)
- Print summary: N docs parsed, M chunks indexed

## Acceptance Criteria
- [ ] All PDFs in `./docs` are parsed without errors
- [ ] Chunk metadata includes: `source`, `doc_type`, `page`, `section`, `chunk_id`
- [ ] Embeddings generated and stored in FAISS index
- [ ] `scripts/ingest.py` runs end-to-end in < 2 min for 20 documents
- [ ] Unit tests pass for parser, chunker, and indexer

## Dependencies
```
pypdf>=4.0
langchain-text-splitters  # or tiktoken for token-aware splitting
google-genai>=1.0.0
faiss-cpu>=1.8
```
