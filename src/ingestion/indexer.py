"""Phase 1 — FAISS Indexer. See phases/phase_1_ingestion.md §1.4

Uses IndexFlatIP (inner product) — vectors must be L2-normalised before add/search
so that inner product == cosine similarity.

Persists two files:
  <path>/index.faiss      — FAISS binary index
  <path>/index_meta.json  — chunk metadata + indexed doc_ids
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from src.models import ChunkMetadata, DocumentChunk

logger = logging.getLogger(__name__)

_INDEX_FILE = "index.faiss"
_META_FILE = "index_meta.json"


class FAISSIndex:
    def __init__(self) -> None:
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[dict] = []          # serialisable chunk metadata + content
        self._indexed_doc_ids: set[str] = set()
        self._dim: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, chunks: list[DocumentChunk]) -> None:
        """Add embedded chunks to the FAISS index (incremental)."""
        if not chunks:
            return

        new_chunks = [c for c in chunks if c.doc_id not in self._indexed_doc_ids]
        if not new_chunks:
            logger.info("All chunks already indexed — nothing to add.")
            return

        vectors = self._to_matrix(new_chunks)

        if self._index is None:
            self._dim = vectors.shape[1]
            self._index = faiss.IndexFlatIP(self._dim)
            logger.info("Created new FAISS index (dim=%d)", self._dim)

        self._index.add(vectors)

        for chunk in new_chunks:
            self._chunks.append(_chunk_to_dict(chunk))
            self._indexed_doc_ids.add(chunk.doc_id)

        logger.info("Added %d chunks (%d total)", len(new_chunks), len(self._chunks))

    def save(self, path: Path) -> None:
        """Persist index and metadata to <path>/index.faiss + index_meta.json."""
        if self._index is None:
            raise RuntimeError("Index is empty — nothing to save.")

        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / _INDEX_FILE))
        meta = {
            "dim": self._dim,
            "indexed_doc_ids": list(self._indexed_doc_ids),
            "chunks": self._chunks,
        }
        (path / _META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        logger.info("Saved index (%d vectors) to %s", self._index.ntotal, path)

    def load(self, path: Path) -> None:
        """Load index and metadata from disk."""
        index_path = path / _INDEX_FILE
        meta_path = path / _META_FILE

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Index files not found in {path}")

        self._index = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text())
        self._dim = meta["dim"]
        self._indexed_doc_ids = set(meta["indexed_doc_ids"])
        self._chunks = meta["chunks"]
        logger.info("Loaded index (%d vectors) from %s", self._index.ntotal, path)

    def search(
        self, query_vector: list[float], top_k: int = 20
    ) -> list[tuple[DocumentChunk, float]]:
        """Return top_k (chunk, score) pairs. Vectors must be L2-normalised."""
        if self._index is None or self._index.ntotal == 0:
            return []

        q = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(q)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q, k)

        results: list[tuple[DocumentChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((_dict_to_chunk(self._chunks[idx]), float(score)))
        return results

    @property
    def indexed_doc_ids(self) -> set[str]:
        return set(self._indexed_doc_ids)

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_matrix(chunks: list[DocumentChunk]) -> np.ndarray:
        missing = [c.chunk_id for c in chunks if not c.embedding]
        if missing:
            raise ValueError(f"{len(missing)} chunks have no embedding. Run embed_chunks first.")

        matrix = np.array([c.embedding for c in chunks], dtype=np.float32)
        faiss.normalize_L2(matrix)
        return matrix


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _chunk_to_dict(chunk: DocumentChunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "content": chunk.content,
        "metadata": asdict(chunk.metadata),
    }


def _dict_to_chunk(d: dict) -> DocumentChunk:
    meta = d["metadata"]
    return DocumentChunk(
        chunk_id=d["chunk_id"],
        doc_id=d["doc_id"],
        content=d["content"],
        metadata=ChunkMetadata(
            source=meta["source"],
            doc_type=meta["doc_type"],
            page=meta["page"],
            section=meta.get("section"),
            language=meta.get("language", "en"),
        ),
        embedding=[],  # not stored — not needed after indexing
    )
