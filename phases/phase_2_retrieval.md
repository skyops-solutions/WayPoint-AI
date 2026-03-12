# Phase 2 — Semantic Retrieval Layer

## Goal
Build the online retrieval component that takes a user query, finds the most relevant document chunks from the FAISS index, and returns ranked context ready for the LLM.

## Deliverables
- `src/retrieval/query_embedder.py` — query embedding
- `src/retrieval/vector_search.py` — FAISS search wrapper
- `src/retrieval/reranker.py` — context scoring and filtering
- `src/retrieval/retriever.py` — orchestrates full retrieval pipeline
- `tests/test_retrieval.py`

## Tasks

### 2.1 Query Embedder
- Same model as ingestion: `models/text-embedding-004`
- L2-normalize query vector before search (required for cosine similarity with `IndexFlatIP`)
- Cache query embeddings per session turn (avoid re-embedding same query on retry)

```python
async def embed_query(query: str) -> list[float]:
```

### 2.2 Vector Search
- Search FAISS index for top-K=20 candidates
- Return: `list[tuple[DocumentChunk, float]]` — (chunk, similarity_score)
- Filter out chunks with similarity < 0.35 (likely irrelevant)
- Support optional `doc_type` filter to narrow search domain

```python
class VectorSearch:
    def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        doc_type_filter: str | None = None
    ) -> list[tuple[DocumentChunk, float]]:
```

### 2.3 Reranker
- Input: top-20 candidates from vector search
- Output: top-5 reranked chunks for LLM context
- Reranking strategy (MVP): score = `0.7 * vector_sim + 0.3 * recency_boost`
  - `recency_boost`: prefer policy/faq docs over older destination guides (configurable)
- Deduplicate: if two chunks from same page have >80% text overlap, keep higher-scored one
- Return chunks in descending relevance order

```python
def rerank(candidates: list[tuple[DocumentChunk, float]], top_n: int = 5) -> list[DocumentChunk]:
```

### 2.4 Retriever Orchestrator
Single entry point for the RAG pipeline's retrieval step:

```python
class Retriever:
    async def retrieve(
        self,
        query: str,
        doc_type_hint: str | None = None
    ) -> RetrievalResult:

@dataclass
class RetrievalResult:
    chunks: list[DocumentChunk]     # top-5, ranked
    max_similarity: float           # used by routing layer for confidence estimation
    query_embedding: list[float]    # cached for logging
```

### 2.5 Intent-Based Filtering
Map detected intent (from Phase 4) to doc_type filter:
| Intent | doc_type filter |
|--------|----------------|
| `cancellation` | `policy` |
| `destination_info` | `destination` |
| `booking_help` | `guide`, `faq` |
| `general` | None (search all) |

This filtering is optional for MVP — implement as a config toggle.

## Acceptance Criteria
- [ ] Retriever returns top-5 chunks for any query in < 200ms locally
- [ ] Similarity threshold correctly filters irrelevant results
- [ ] Deduplication removes near-duplicate chunks
- [ ] `max_similarity` correlates with answer quality (manual spot-check on 10 queries)
- [ ] Unit tests cover: empty index, single doc, multi-doc, filter by doc_type

## Dependencies
```
faiss-cpu>=1.8
google-genai>=1.0.0
numpy>=1.26
```
