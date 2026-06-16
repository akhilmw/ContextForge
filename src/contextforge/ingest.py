"""Coordinate the Phase 1 repository ingestion pipeline.

This module wires together discovery, loading, chunking, embedding, and
storage. Each individual step owns its own logic; this layer only controls the
order of operations and returns a run summary.
"""

from pathlib import Path

from contextforge.chunker import chunk_documents
from contextforge.discovery import discover_files
from contextforge.embedder import Embedder, embed_chunks
from contextforge.loaders import load_documents
from contextforge.models import IngestionResult
from contextforge.store import save_chunks


def ingest_repository(
    repo_path: Path,
    project_name: str,
    data_dir: Path,
    embedder: Embedder,
) -> IngestionResult:
    """Ingest a local repository into a project-specific chunk index.

    The embedder is injected so tests can use FakeEmbedder and production code
    can use GeminiEmbedder without changing the pipeline.
    """

    paths = discover_files(repo_path)
    documents = load_documents(repo_path, paths)
    chunks = chunk_documents(documents, project_name)
    embedded_chunks = embed_chunks(chunks, embedder)
    index_path = save_chunks(data_dir, project_name, embedded_chunks)

    return IngestionResult(
        project_name=project_name,
        files_discovered=len(paths),
        documents_loaded=len(documents),
        chunks_created=len(chunks),
        chunks_saved=len(embedded_chunks),
        index_path=index_path,
    )
