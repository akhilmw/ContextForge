"""Split loaded Documents into source-aware, overlapping Chunks.

Phase 1 uses line-based chunking so chunk boundaries and source citations are
easy to inspect. Embeddings are added later in the ingestion pipeline.
"""

import hashlib

from contextforge.models import Document, Chunk
from contextforge.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE


def chunk_document(
    document: Document,
    project_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split one Document into ordered Chunks with line metadata."""

    # Overlap must leave a positive step between consecutive chunk starts.
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("invalid configs for chunk_size/chunk_overlap")
    
    # The project name scopes chunks when multiple repositories are ingested.
    if not project_name.strip():
        raise ValueError("project name cannot be empty")
    
    if not document.content:
        return []
    
    chunks = []
    
    # Preserve line endings so joining a slice recreates the original text.
    lines = document.content.splitlines(keepends=True)

    # Example: size 20 and overlap 5 advances each new chunk by 15 lines.
    step = chunk_size - chunk_overlap

    start = 0
    while start < len(lines):
        end = min(start + chunk_size, len(lines))
        chunk_content = "".join(lines[start:end])

        # List indexes are zero-based and slice ends are exclusive. Source
        # citations are one-based and include both the start and end line.
        start_line = start + 1
        end_line = end

        # Hash source identity and content so unchanged input produces the same
        # ID while edits, path changes, or project changes produce a new ID.
        identity = (
            f"{project_name}:{document.file_path}:"
            f"{start_line}:{end_line}:{chunk_content}"
        )
        chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        chunk = Chunk(
            chunk_id,
            project_name,
            document.file_path,
            document.language,
            chunk_content,
            start_line,
            end_line
        )
        chunks.append(chunk)

        # Stop once the final source line is included. Advancing again could
        # create a redundant chunk containing only overlap from the end.
        if end == len(lines):
            break
        start = start + step

    return chunks


def chunk_documents(
    documents: list[Document],
    project_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk multiple Documents and preserve document and chunk order."""

    chunk_list = []

    for document in documents:
        # Flatten each document's chunks into one ingestion-ready list.
        chunk_list.extend(chunk_document(document, project_name, chunk_size, chunk_overlap))

    return chunk_list
