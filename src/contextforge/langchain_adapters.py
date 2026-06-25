from langchain_core.documents import Document

from contextforge.config import REQUIRED_CHUNK_METADATA
from contextforge.models import Chunk


def chunk_to_langchain_document(chunk: Chunk) -> Document:
    """Convert a ContextForge chunk into LangChain's Document type."""

    if chunk is None:
        raise ValueError("chunk cannot be None")

    # LangChain stores text in page_content and source/citation details in
    # metadata. Keep this mapping explicit so the framework boundary is clear.
    metadata = {
        "chunk_id": chunk.chunk_id,
        "project_name": chunk.project_name,
        "file_path": chunk.file_path,
        "language": chunk.language,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "embedding": chunk.embedding,
    }

    return Document(page_content=chunk.content, metadata=metadata)


def langchain_document_to_chunk(document: Document) -> Chunk:
    """Convert a LangChain Document back into a ContextForge chunk."""

    if document is None:
        raise ValueError("document cannot be None")

    missing = REQUIRED_CHUNK_METADATA - set(document.metadata)
    if missing:
        raise ValueError(
            "document metadata is missing required keys: "
            f"{sorted(missing)}"
        )

    chunk_data = {
        **document.metadata,
        "content": document.page_content,
    }
    return Chunk(**chunk_data)
