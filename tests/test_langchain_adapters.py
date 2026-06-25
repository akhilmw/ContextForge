import pytest
from langchain_core.documents import Document

from contextforge.langchain_adapters import (
    chunk_to_langchain_document,
    langchain_document_to_chunk,
)
from contextforge.models import Chunk


def make_chunk(embedding=None) -> Chunk:
    return Chunk(
        chunk_id="demo:src/parser.py:1-3",
        project_name="demo",
        file_path="src/parser.py",
        language="python",
        content="def parse_request(request):\n    return request.strip()",
        start_line=1,
        end_line=3,
        embedding=embedding,
    )


def test_chunk_to_langchain_document_preserves_content_and_metadata():
    chunk = make_chunk(embedding=[0.1, 0.2, 0.3])

    document = chunk_to_langchain_document(chunk)

    assert isinstance(document, Document)
    assert document.page_content == chunk.content
    assert document.metadata == {
        "chunk_id": chunk.chunk_id,
        "project_name": chunk.project_name,
        "file_path": chunk.file_path,
        "language": chunk.language,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "embedding": chunk.embedding,
    }


def test_chunk_to_langchain_document_preserves_missing_embedding():
    chunk = make_chunk(embedding=None)

    document = chunk_to_langchain_document(chunk)

    assert document.metadata["embedding"] is None


def test_chunk_to_langchain_document_rejects_none():
    with pytest.raises(ValueError, match="chunk cannot be None"):
        chunk_to_langchain_document(None)


def test_langchain_document_to_chunk_preserves_content_and_metadata():
    document = Document(
        page_content="def parse_request(request):\n    return request.strip()",
        metadata={
            "chunk_id": "demo:src/parser.py:1-3",
            "project_name": "demo",
            "file_path": "src/parser.py",
            "language": "python",
            "start_line": 1,
            "end_line": 3,
            "embedding": [0.1, 0.2, 0.3],
        },
    )

    chunk = langchain_document_to_chunk(document)

    assert chunk == Chunk(
        chunk_id="demo:src/parser.py:1-3",
        project_name="demo",
        file_path="src/parser.py",
        language="python",
        content="def parse_request(request):\n    return request.strip()",
        start_line=1,
        end_line=3,
        embedding=[0.1, 0.2, 0.3],
    )


def test_langchain_document_to_chunk_allows_missing_embedding():
    document = Document(
        page_content="content",
        metadata={
            "chunk_id": "chunk-1",
            "project_name": "demo",
            "file_path": "src/example.py",
            "language": "python",
            "start_line": 1,
            "end_line": 1,
        },
    )

    chunk = langchain_document_to_chunk(document)

    assert chunk.embedding is None


def test_langchain_document_to_chunk_round_trip_preserves_chunk():
    original = make_chunk(embedding=[0.1, 0.2, 0.3])

    document = chunk_to_langchain_document(original)
    restored = langchain_document_to_chunk(document)

    assert restored == original


@pytest.mark.parametrize(
    "missing_key",
    [
        "chunk_id",
        "project_name",
        "file_path",
        "language",
        "start_line",
        "end_line",
    ],
)
def test_langchain_document_to_chunk_rejects_missing_required_metadata(
    missing_key,
):
    metadata = {
        "chunk_id": "chunk-1",
        "project_name": "demo",
        "file_path": "src/example.py",
        "language": "python",
        "start_line": 1,
        "end_line": 1,
    }
    metadata.pop(missing_key)
    document = Document(page_content="content", metadata=metadata)

    with pytest.raises(
        ValueError,
        match="document metadata is missing required keys",
    ):
        langchain_document_to_chunk(document)


def test_langchain_document_to_chunk_rejects_none():
    with pytest.raises(ValueError, match="document cannot be None"):
        langchain_document_to_chunk(None)
