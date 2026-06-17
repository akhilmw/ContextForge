import math
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from contextforge.embedder import (
    FakeEmbedder,
    GeminiEmbedder,
    OpenAIEmbedder,
    embed_chunks,
    validate_embeddings,
)
from contextforge.models import Chunk


def make_chunk(
    chunk_id: str,
    content: str,
    file_path: str = "src/example.py",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        project_name="demo",
        file_path=file_path,
        language="python",
        content=content,
        start_line=1,
        end_line=1,
    )


def make_gemini_embedder(client, model="gemini-embedding-001"):
    embedder = object.__new__(GeminiEmbedder)
    embedder.api_key = "test-api-key"
    embedder.client = client
    embedder.model = model
    return embedder


def make_gemini_response(vectors):
    return SimpleNamespace(
        embeddings=[
            SimpleNamespace(values=vector)
            for vector in vectors
        ],
    )


def make_openai_embedder(client, model="text-embedding-3-small"):
    embedder = object.__new__(OpenAIEmbedder)
    embedder.api_key = "test-api-key"
    embedder.client = client
    embedder.model = model
    return embedder


def make_openai_response(vectors):
    return SimpleNamespace(
        data=[
            SimpleNamespace(embedding=vector)
            for vector in vectors
        ],
    )


def test_fake_embedder_returns_empty_list_for_empty_documents():
    embedder = FakeEmbedder()

    assert embedder.embed_documents([]) == []


def test_fake_embedder_returns_one_vector_per_document():
    embedder = FakeEmbedder()
    texts = ["one", "two words", "three"]

    vectors = embedder.embed_documents(texts)

    assert len(vectors) == len(texts)
    assert all(len(vector) == 3 for vector in vectors)


def test_fake_embedder_preserves_document_order():
    embedder = FakeEmbedder()

    vectors = embedder.embed_documents(["a", "longer text"])

    assert vectors[0][0] == 1.0
    assert vectors[1][0] == 11.0


def test_fake_embedder_is_deterministic():
    embedder = FakeEmbedder()

    first = embedder.embed_documents(["repeatable text"])
    second = embedder.embed_documents(["repeatable text"])

    assert first == second


def test_fake_embedder_returns_flat_query_vector():
    embedder = FakeEmbedder()

    vector = embedder.embed_query("question")

    assert vector == [8.0, 1.0, 888.0]
    assert all(isinstance(value, float) for value in vector)


@pytest.mark.parametrize("text", ["", "   "])
def test_fake_embedder_rejects_empty_query(text):
    embedder = FakeEmbedder()

    with pytest.raises(ValueError, match="Query text cannot be empty"):
        embedder.embed_query(text)


def test_validate_embeddings_accepts_valid_vectors():
    validate_embeddings(
        ["first", "second"],
        [[0.1, 0.2], [0.3, 0.4]],
    )


def test_validate_embeddings_accepts_empty_input():
    validate_embeddings([], [])


def test_validate_embeddings_rejects_wrong_vector_count():
    with pytest.raises(
        ValueError,
        match="Expected 2 vectors, received 1",
    ):
        validate_embeddings(
            ["first", "second"],
            [[0.1, 0.2]],
        )


def test_validate_embeddings_rejects_empty_vectors():
    with pytest.raises(
        ValueError,
        match="Embedding vectors cannot be empty",
    ):
        validate_embeddings(["first"], [[]])


def test_validate_embeddings_rejects_inconsistent_dimensions():
    with pytest.raises(
        ValueError,
        match="Embedding vectors have inconsistent dimensions",
    ):
        validate_embeddings(
            ["first", "second"],
            [[0.1, 0.2], [0.3]],
        )


@pytest.mark.parametrize("invalid_value", ["invalid", True])
def test_validate_embeddings_rejects_non_numeric_values(invalid_value):
    with pytest.raises(TypeError, match="Embedding values must be numeric"):
        validate_embeddings(["first"], [[0.1, invalid_value]])


@pytest.mark.parametrize("invalid_value", [math.nan, math.inf, -math.inf])
def test_validate_embeddings_rejects_non_finite_values(invalid_value):
    with pytest.raises(TypeError, match="Embedding values must be numeric"):
        validate_embeddings(["first"], [[invalid_value]])


def test_embed_chunks_returns_empty_list_for_empty_input():
    assert embed_chunks([], FakeEmbedder()) == []


def test_embed_chunks_attaches_vectors_in_chunk_order():
    chunks = [
        make_chunk("chunk-1", "one"),
        make_chunk("chunk-2", "two words", "src/second.py"),
    ]

    result = embed_chunks(chunks, FakeEmbedder())

    assert result is chunks
    assert chunks[0].embedding == [3.0, 1.0, 322.0]
    assert chunks[1].embedding == [9.0, 2.0, 937.0]


def test_embed_chunks_preserves_chunk_metadata():
    chunk = make_chunk("chunk-1", "one")

    embed_chunks([chunk], FakeEmbedder())

    assert chunk.chunk_id == "chunk-1"
    assert chunk.project_name == "demo"
    assert chunk.file_path == "src/example.py"
    assert chunk.language == "python"
    assert chunk.content == "one"
    assert chunk.start_line == 1
    assert chunk.end_line == 1


def test_embed_chunks_gives_each_chunk_its_own_vector():
    chunks = [
        make_chunk("chunk-1", "same"),
        make_chunk("chunk-2", "same"),
    ]

    embed_chunks(chunks, FakeEmbedder())

    assert chunks[0].embedding == chunks[1].embedding
    assert chunks[0].embedding is not chunks[1].embedding


def test_embed_chunks_validates_output_before_mutating_chunks():
    class WrongCountEmbedder:
        def embed_documents(self, texts):
            return [[1.0, 2.0]]

        def embed_query(self, text):
            return [1.0, 2.0]

    chunks = [
        make_chunk("chunk-1", "first"),
        make_chunk("chunk-2", "second"),
    ]

    with pytest.raises(
        ValueError,
        match="Expected 2 vectors, received 1",
    ):
        embed_chunks(chunks, WrongCountEmbedder())

    assert [chunk.embedding for chunk in chunks] == [None, None]


def test_gemini_embedder_returns_empty_list_without_calling_api():
    client = Mock()
    embedder = make_gemini_embedder(client)

    assert embedder.embed_documents([]) == []
    client.models.embed_content.assert_not_called()


def test_gemini_embedder_embeds_documents_with_retrieval_document_task():
    client = Mock()
    client.models.embed_content.return_value = make_gemini_response(
        [[0.1, 0.2], [0.3, 0.4]],
    )
    embedder = make_gemini_embedder(client)
    texts = ["first document", "second document"]

    vectors = embedder.embed_documents(texts)

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    call = client.models.embed_content.call_args
    assert call.kwargs["model"] == "gemini-embedding-001"
    assert call.kwargs["contents"] == texts
    assert call.kwargs["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert call.kwargs["config"].output_dimensionality == 768


def test_gemini_embedder_embeds_query_with_retrieval_query_task():
    client = Mock()
    client.models.embed_content.return_value = make_gemini_response(
        [[0.1, 0.2]],
    )
    embedder = make_gemini_embedder(client)

    vector = embedder.embed_query("How is the request parsed?")

    assert vector == [0.1, 0.2]
    call = client.models.embed_content.call_args
    assert call.kwargs["model"] == "gemini-embedding-001"
    assert call.kwargs["contents"] == "How is the request parsed?"
    assert call.kwargs["config"].task_type == "RETRIEVAL_QUERY"
    assert call.kwargs["config"].output_dimensionality == 768


@pytest.mark.parametrize("text", ["", "   "])
def test_gemini_embedder_rejects_empty_query_without_calling_api(text):
    client = Mock()
    embedder = make_gemini_embedder(client)

    with pytest.raises(ValueError, match="Query text cannot be empty"):
        embedder.embed_query(text)

    client.models.embed_content.assert_not_called()


def test_gemini_embedder_validates_document_response():
    client = Mock()
    client.models.embed_content.return_value = make_gemini_response(
        [[0.1, 0.2]],
    )
    embedder = make_gemini_embedder(client)

    with pytest.raises(
        ValueError,
        match="Expected 2 vectors, received 1",
    ):
        embedder.embed_documents(["first", "second"])


def test_gemini_embedder_validates_query_response():
    client = Mock()
    client.models.embed_content.return_value = make_gemini_response(
        [[math.nan]],
    )
    embedder = make_gemini_embedder(client)

    with pytest.raises(TypeError, match="Embedding values must be numeric"):
        embedder.embed_query("question")


def test_gemini_embedder_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Could not fetch the API Key"):
        GeminiEmbedder()


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_GEMINI_INTEGRATION") != "1",
    reason="Set RUN_GEMINI_INTEGRATION=1 to call the live Gemini API",
)
def test_gemini_embedder_live_api():
    embedder = GeminiEmbedder()

    document_vectors = embedder.embed_documents(
        [
            "Python parses an HTTP request.",
            "Go writes an HTTP response.",
        ],
    )
    query_vector = embedder.embed_query("How is an HTTP request parsed?")

    assert len(document_vectors) == 2
    assert all(len(vector) == 768 for vector in document_vectors)
    assert len(query_vector) == 768


def test_openai_embedder_returns_empty_list_without_calling_api():
    client = Mock()
    embedder = make_openai_embedder(client)

    assert embedder.embed_documents([]) == []
    client.embeddings.create.assert_not_called()


def test_openai_embedder_embeds_documents():
    client = Mock()
    client.embeddings.create.return_value = make_openai_response(
        [[0.1, 0.2], [0.3, 0.4]],
    )
    embedder = make_openai_embedder(client)
    texts = ["first document", "second document"]

    vectors = embedder.embed_documents(texts)

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    call = client.embeddings.create.call_args
    assert call.kwargs["model"] == "text-embedding-3-small"
    assert call.kwargs["input"] == texts


def test_openai_embedder_embeds_query():
    client = Mock()
    client.embeddings.create.return_value = make_openai_response(
        [[0.1, 0.2]],
    )
    embedder = make_openai_embedder(client)

    vector = embedder.embed_query("How is the request parsed?")

    assert vector == [0.1, 0.2]
    call = client.embeddings.create.call_args
    assert call.kwargs["model"] == "text-embedding-3-small"
    assert call.kwargs["input"] == ["How is the request parsed?"]


@pytest.mark.parametrize("text", ["", "   "])
def test_openai_embedder_rejects_empty_query_without_calling_api(text):
    client = Mock()
    embedder = make_openai_embedder(client)

    with pytest.raises(ValueError, match="Query text cannot be empty"):
        embedder.embed_query(text)

    client.embeddings.create.assert_not_called()


def test_openai_embedder_validates_document_response():
    client = Mock()
    client.embeddings.create.return_value = make_openai_response(
        [[0.1, 0.2]],
    )
    embedder = make_openai_embedder(client)

    with pytest.raises(
        ValueError,
        match="Expected 2 vectors, received 1",
    ):
        embedder.embed_documents(["first", "second"])


def test_openai_embedder_validates_query_response():
    client = Mock()
    client.embeddings.create.return_value = make_openai_response(
        [[math.nan]],
    )
    embedder = make_openai_embedder(client)

    with pytest.raises(TypeError, match="Embedding values must be numeric"):
        embedder.embed_query("question")


def test_openai_embedder_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Could not fetch the OpenAI API Key"):
        OpenAIEmbedder()
