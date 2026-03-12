"""CLI runner for document ingestion. See phases/phase_1_ingestion.md §1.5

Usage:
    python scripts/ingest.py --docs-dir ./docs --index-dir ./data/index
    python scripts/ingest.py --docs-dir ./docs --index-dir ./data/index --force-rebuild
"""
import argparse
import asyncio
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest travel agency documents into FAISS index.")
    parser.add_argument("--docs-dir", type=Path, default=Path("./docs"))
    parser.add_argument("--index-dir", type=Path, default=Path("./data/index"))
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Drop existing index and rebuild from scratch.",
    )
    return parser.parse_args()


async def run(docs_dir: Path, index_dir: Path, force_rebuild: bool) -> None:
    # Late imports so config is already loaded from .env
    from src.ingestion.chunker import chunk_document, infer_doc_type
    from src.ingestion.embedder import embed_chunks
    from src.ingestion.indexer import FAISSIndex
    from src.ingestion.parser import parse_document

    if not docs_dir.exists():
        raise FileNotFoundError(f"docs-dir not found: {docs_dir}")

    # Collect all supported files
    doc_files = [
        p for p in docs_dir.rglob("*") if p.suffix.lower() in _SUPPORTED_EXTENSIONS
    ]
    if not doc_files:
        logger.warning("No documents found in %s", docs_dir)
        return

    logger.info("Found %d document(s) in %s", len(doc_files), docs_dir)

    # Load or create index
    index = FAISSIndex()
    if force_rebuild and index_dir.exists():
        logger.info("--force-rebuild: removing existing index at %s", index_dir)
        shutil.rmtree(index_dir)
    elif index_dir.exists():
        try:
            index.load(index_dir)
            logger.info("Loaded existing index (%d vectors)", index.total_vectors)
        except FileNotFoundError:
            logger.info("No existing index found — building from scratch.")

    already_indexed = index.indexed_doc_ids
    docs_parsed = 0
    chunks_total: list = []

    for doc_path in doc_files:
        doc_id = doc_path.stem
        if doc_id in already_indexed:
            logger.info("Skipping already-indexed: %s", doc_path.name)
            continue

        logger.info("Parsing: %s", doc_path.name)
        try:
            raw_chunks = parse_document(doc_path)
        except Exception as exc:
            logger.error("Failed to parse %s: %s", doc_path.name, exc)
            continue

        doc_type = infer_doc_type(doc_path.name)
        chunks = chunk_document(raw_chunks, doc_type)
        logger.info("  → %d chunks (doc_type=%s)", len(chunks), doc_type)
        chunks_total.extend(chunks)
        docs_parsed += 1

    if not chunks_total:
        logger.info("Nothing new to embed. Index is up to date.")
        return

    logger.info("Embedding %d chunks...", len(chunks_total))
    await embed_chunks(chunks_total)

    index.add(chunks_total)
    index.save(index_dir)

    logger.info(
        "Done. Parsed %d doc(s), indexed %d chunk(s). Total vectors: %d.",
        docs_parsed,
        len(chunks_total),
        index.total_vectors,
    )


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args.docs_dir, args.index_dir, args.force_rebuild))
