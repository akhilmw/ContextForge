from pathlib import Path

import pytest

from contextforge.models import Chunk, SearchResult
from contextforge.retriever import (
    cosine_similarity,
    retrieve,
    retrieve_deduplicated,
    retrieve_diverse,
    retrieve_hybrid,
)
from contextforge.reranker import FakeReranker
from contextforge.store import save_chunks


class QueryEmbedder:
    def __init__(self, vector):
        self.vector = vector

    def embed_documents(self, texts):
        raise NotImplementedError

    def embed_query(self, text):
        return self.vector


def make_chunk(
    chunk_id,
    embedding,
    content=None,
    file_path=None,
    start_line=1,
    end_line=1,
):
    return Chunk(
        chunk_id=chunk_id,
        project_name="demo",
        file_path=file_path or f"src/{chunk_id}.py",
        language="python",
        content=content or f"content for {chunk_id}",
        start_line=start_line,
        end_line=end_line,
        embedding=embedding,
    )


def test_cosine_similarity_identical_vectors():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


@pytest.mark.parametrize(
    ("a", "b", "error_type", "message"),
    [
        ([], [], ValueError, "Vectors cannot be empty"),
        ([1.0, 2.0], [1.0], ValueError, "Vectors must have the same dimension"),
        ([0.0, 0.0], [1.0, 0.0], ValueError, "zero vectors"),
        ([1.0, float("nan")], [1.0, 2.0], TypeError, "finite numbers"),
        ([True, 1.0], [1.0, 2.0], TypeError, "finite numbers"),
    ],
)
def test_cosine_similarity_rejects_invalid_vectors(a, b, error_type, message):
    with pytest.raises(error_type, match=message):
        cosine_similarity(a, b)


def test_retrieve_returns_empty_list_for_empty_project(tmp_path):
    save_chunks(tmp_path, "demo", [])

    results = retrieve(
        data_dir=tmp_path,
        project_name="demo",
        question="What parses requests?",
        embedder=QueryEmbedder([1.0, 0.0]),
    )

    assert results == []


def test_retrieve_ranks_results_by_score_descending(tmp_path):
    chunks = [
        make_chunk("best", [1.0, 0.0]),
        make_chunk("middle", [0.5, 0.5]),
        make_chunk("worst", [-1.0, 0.0]),
    ]
    save_chunks(tmp_path, "demo", chunks)

    results = retrieve(
        data_dir=tmp_path,
        project_name="demo",
        question="What parses requests?",
        embedder=QueryEmbedder([1.0, 0.0]),
        top_k=3,
    )

    assert [result.chunk.chunk_id for result in results] == [
        "best",
        "middle",
        "worst",
    ]
    assert [result.score for result in results] == sorted(
        [result.score for result in results],
        reverse=True,
    )


def test_retrieve_respects_top_k(tmp_path):
    chunks = [
        make_chunk("best", [1.0, 0.0]),
        make_chunk("middle", [0.5, 0.5]),
        make_chunk("worst", [-1.0, 0.0]),
    ]
    save_chunks(tmp_path, "demo", chunks)

    results = retrieve(
        data_dir=tmp_path,
        project_name="demo",
        question="What parses requests?",
        embedder=QueryEmbedder([1.0, 0.0]),
        top_k=2,
    )

    assert [result.chunk.chunk_id for result in results] == ["best", "middle"]


def test_retrieve_returns_search_result_objects(tmp_path):
    chunk = make_chunk("best", [1.0, 0.0])
    save_chunks(tmp_path, "demo", [chunk])

    results = retrieve(
        data_dir=tmp_path,
        project_name="demo",
        question="What parses requests?",
        embedder=QueryEmbedder([1.0, 0.0]),
    )

    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].chunk == chunk
    assert results[0].score == pytest.approx(1.0)


def test_retrieve_rejects_empty_question(tmp_path):
    save_chunks(tmp_path, "demo", [])

    with pytest.raises(ValueError, match="question cannot be empty"):
        retrieve(
            data_dir=tmp_path,
            project_name="demo",
            question="   ",
            embedder=QueryEmbedder([1.0, 0.0]),
        )


@pytest.mark.parametrize("top_k", [0, -1])
def test_retrieve_rejects_invalid_top_k(tmp_path, top_k):
    save_chunks(tmp_path, "demo", [])

    with pytest.raises(ValueError, match="top k cannot be less than or equal to zero"):
        retrieve(
            data_dir=tmp_path,
            project_name="demo",
            question="What parses requests?",
            embedder=QueryEmbedder([1.0, 0.0]),
            top_k=top_k,
        )


def test_retrieve_propagates_missing_project_index(tmp_path):
    with pytest.raises(FileNotFoundError, match="Requested file not found"):
        retrieve(
            data_dir=tmp_path,
            project_name="missing",
            question="What parses requests?",
            embedder=QueryEmbedder([1.0, 0.0]),
        )


def test_retrieve_rejects_query_dimension_mismatch(tmp_path):
    save_chunks(tmp_path, "demo", [make_chunk("best", [1.0, 0.0])])

    with pytest.raises(ValueError, match="same dimension"):
        retrieve(
            data_dir=tmp_path,
            project_name="demo",
            question="What parses requests?",
            embedder=QueryEmbedder([1.0, 0.0, 0.0]),
        )


def test_retrieve_deduplicated_overfetches_and_replaces_overlap(tmp_path):
    chunks = [
        make_chunk(
            "best",
            [1.0, 0.0],
            file_path="src/parser.py",
            start_line=1,
            end_line=20,
        ),
        make_chunk(
            "overlap",
            [0.9, 0.1],
            file_path="src/parser.py",
            start_line=11,
            end_line=30,
        ),
        make_chunk("replacement", [0.8, 0.2]),
    ]
    save_chunks(tmp_path, "demo", chunks)

    results = retrieve_deduplicated(
        data_dir=tmp_path,
        project_name="demo",
        question="What parses requests?",
        embedder=QueryEmbedder([1.0, 0.0]),
        top_k=2,
        candidate_k=3,
        overlap_threshold=0.5,
    )

    assert [result.chunk.chunk_id for result in results] == ["best", "replacement"]


def test_retrieve_deduplicated_respects_final_top_k(tmp_path):
    chunks = [
        make_chunk("first", [1.0, 0.0]),
        make_chunk("second", [0.8, 0.2]),
        make_chunk("third", [0.6, 0.4]),
    ]
    save_chunks(tmp_path, "demo", chunks)

    results = retrieve_deduplicated(
        data_dir=tmp_path,
        project_name="demo",
        question="What parses requests?",
        embedder=QueryEmbedder([1.0, 0.0]),
        top_k=2,
        candidate_k=3,
    )

    assert len(results) == 2


def test_retrieve_deduplicated_returns_empty_list_for_empty_project(tmp_path):
    save_chunks(tmp_path, "demo", [])

    results = retrieve_deduplicated(
        data_dir=tmp_path,
        project_name="demo",
        question="What parses requests?",
        embedder=QueryEmbedder([1.0, 0.0]),
    )

    assert results == []


@pytest.mark.parametrize("top_k", [True, 1.5, "5"])
def test_retrieve_deduplicated_rejects_non_integer_top_k(tmp_path, top_k):
    with pytest.raises(TypeError, match="is not of type int"):
        retrieve_deduplicated(
            tmp_path,
            "demo",
            "question",
            QueryEmbedder([1.0, 0.0]),
            top_k=top_k,
        )


@pytest.mark.parametrize("top_k", [0, -1])
def test_retrieve_deduplicated_rejects_non_positive_top_k(tmp_path, top_k):
    with pytest.raises(ValueError, match="top k cannot"):
        retrieve_deduplicated(
            tmp_path,
            "demo",
            "question",
            QueryEmbedder([1.0, 0.0]),
            top_k=top_k,
        )


@pytest.mark.parametrize("candidate_k", [True, 2.5, "15"])
def test_retrieve_deduplicated_rejects_non_integer_candidate_k(
    tmp_path, candidate_k
):
    with pytest.raises(TypeError, match="is not of type int"):
        retrieve_deduplicated(
            tmp_path,
            "demo",
            "question",
            QueryEmbedder([1.0, 0.0]),
            candidate_k=candidate_k,
        )


def test_retrieve_deduplicated_requires_enough_candidates(tmp_path):
    with pytest.raises(ValueError, match="candidate_k must be at least top_k"):
        retrieve_deduplicated(
            tmp_path,
            "demo",
            "question",
            QueryEmbedder([1.0, 0.0]),
            top_k=3,
            candidate_k=2,
        )


@pytest.mark.parametrize(
    ("threshold", "error_type", "message"),
    [
        (0, ValueError, "is out of bounds"),
        (1.1, ValueError, "is out of bounds"),
        (True, TypeError, "is not of type numeric"),
        ("0.5", TypeError, "is not of type numeric"),
    ],
)
def test_retrieve_deduplicated_rejects_invalid_overlap_threshold(
    tmp_path, threshold, error_type, message
):
    with pytest.raises(error_type, match=message):
        retrieve_deduplicated(
            tmp_path,
            "demo",
            "question",
            QueryEmbedder([1.0, 0.0]),
            overlap_threshold=threshold,
        )


def test_retrieve_diverse_promotes_result_from_another_file(tmp_path):
    chunks = [
        make_chunk(
            "a-first", [1.0, 0.0], file_path="src/a.py", start_line=1, end_line=10
        ),
        make_chunk(
            "a-second",
            [0.9, 0.1],
            file_path="src/a.py",
            start_line=11,
            end_line=20,
        ),
        make_chunk("b-first", [0.8, 0.2], file_path="src/b.py"),
    ]
    save_chunks(tmp_path, "demo", chunks)

    results = retrieve_diverse(
        data_dir=tmp_path,
        project_name="demo",
        question="What parses requests?",
        embedder=QueryEmbedder([1.0, 0.0]),
        top_k=2,
        candidate_k=3,
        max_per_file=1,
    )

    assert [result.chunk.chunk_id for result in results] == ["a-first", "b-first"]


def test_retrieve_diverse_allows_configured_results_per_file(tmp_path):
    chunks = [
        make_chunk(
            "a-first", [1.0, 0.0], file_path="src/a.py", start_line=1, end_line=10
        ),
        make_chunk(
            "a-second",
            [0.9, 0.1],
            file_path="src/a.py",
            start_line=11,
            end_line=20,
        ),
        make_chunk("b-first", [0.8, 0.2], file_path="src/b.py"),
    ]
    save_chunks(tmp_path, "demo", chunks)

    results = retrieve_diverse(
        data_dir=tmp_path,
        project_name="demo",
        question="What parses requests?",
        embedder=QueryEmbedder([1.0, 0.0]),
        top_k=3,
        candidate_k=3,
        max_per_file=2,
    )

    assert [result.chunk.chunk_id for result in results] == [
        "a-first",
        "a-second",
        "b-first",
    ]


def test_retrieve_diverse_returns_empty_list_for_empty_project(tmp_path):
    save_chunks(tmp_path, "demo", [])

    results = retrieve_diverse(
        data_dir=tmp_path,
        project_name="demo",
        question="What parses requests?",
        embedder=QueryEmbedder([1.0, 0.0]),
    )

    assert results == []


@pytest.mark.parametrize("max_per_file", [True, 1.5, "1"])
def test_retrieve_diverse_rejects_non_integer_file_limit(tmp_path, max_per_file):
    with pytest.raises(TypeError, match="is not of type int"):
        retrieve_diverse(
            tmp_path,
            "demo",
            "question",
            QueryEmbedder([1.0, 0.0]),
            max_per_file=max_per_file,
        )


@pytest.mark.parametrize("max_per_file", [0, -1])
def test_retrieve_diverse_rejects_non_positive_file_limit(tmp_path, max_per_file):
    with pytest.raises(ValueError, match="max_per_file cannot"):
        retrieve_diverse(
            tmp_path,
            "demo",
            "question",
            QueryEmbedder([1.0, 0.0]),
            max_per_file=max_per_file,
        )


def test_retrieve_hybrid_fuses_then_deduplicates_and_diversifies(
    tmp_path, monkeypatch
):
    a = SearchResult(
        chunk=make_chunk(
            "a",
            [1.0, 0.0],
            file_path="src/shared.py",
            start_line=1,
            end_line=1,
        ),
        score=0.9,
    )
    b = SearchResult(chunk=make_chunk("b", [0.8, 0.2]), score=0.8)
    c = SearchResult(chunk=make_chunk("c", [0.7, 0.3]), score=0.7)
    d = SearchResult(
        chunk=make_chunk(
            "d",
            [0.0, 1.0],
            file_path="src/shared.py",
            start_line=2,
            end_line=2,
        ),
        score=8.0,
    )
    calls = {}

    def fake_semantic(**kwargs):
        calls["semantic"] = kwargs
        return [a, b, c]

    def fake_keyword(**kwargs):
        calls["keyword"] = kwargs
        return [b, d, a]

    monkeypatch.setattr("contextforge.retriever.retrieve", fake_semantic)
    monkeypatch.setattr("contextforge.retriever.retrieve_keywords", fake_keyword)

    results = retrieve_hybrid(
        data_dir=tmp_path,
        project_name="demo",
        question="How are requests handled?",
        embedder=QueryEmbedder([1.0, 0.0]),
        top_k=3,
        candidate_k=10,
        rank_constant=60,
        overlap_threshold=0.25,
        max_per_file=1,
        k1=1.2,
        b=0.6,
    )

    # RRF orders b, a, d, c. Diversity removes d because a from the same file
    # was seen first, promoting c into the final top three.
    assert [result.chunk.chunk_id for result in results] == ["b", "a", "c"]
    assert calls["semantic"]["top_k"] == 10
    assert calls["keyword"]["top_k"] == 10
    assert calls["keyword"]["k1"] == 1.2
    assert calls["keyword"]["b"] == 0.6


def test_retrieve_hybrid_returns_empty_when_both_strategies_are_empty(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("contextforge.retriever.retrieve", lambda **kwargs: [])
    monkeypatch.setattr(
        "contextforge.retriever.retrieve_keywords", lambda **kwargs: []
    )

    results = retrieve_hybrid(
        tmp_path,
        "demo",
        "question",
        QueryEmbedder([1.0, 0.0]),
    )

    assert results == []


def test_retrieve_hybrid_reranks_before_diversity_filter(tmp_path, monkeypatch):
    shared_first = SearchResult(
        chunk=make_chunk(
            "shared-first",
            [1.0, 0.0],
            file_path="src/shared.py",
            start_line=1,
            end_line=1,
        ),
        score=0.9,
    )
    shared_second = SearchResult(
        chunk=make_chunk(
            "shared-second",
            [0.8, 0.2],
            file_path="src/shared.py",
            start_line=2,
            end_line=2,
        ),
        score=0.8,
    )
    other = SearchResult(
        chunk=make_chunk("other", [0.7, 0.3], file_path="src/other.py"),
        score=0.7,
    )

    monkeypatch.setattr(
        "contextforge.retriever.retrieve",
        lambda **kwargs: [shared_first, shared_second, other],
    )
    monkeypatch.setattr("contextforge.retriever.retrieve_keywords", lambda **kwargs: [])

    results = retrieve_hybrid(
        data_dir=tmp_path,
        project_name="demo",
        question="How are shared requests handled?",
        embedder=QueryEmbedder([1.0, 0.0]),
        top_k=2,
        candidate_k=3,
        max_per_file=1,
        reranker=FakeReranker(["shared-second", "shared-first", "other"]),
    )

    assert [result.chunk.chunk_id for result in results] == ["shared-second", "other"]


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"top_k": True}, TypeError, "top_k is not of type int"),
        ({"top_k": 0}, ValueError, "top_k has to be a positive integer"),
        ({"candidate_k": True}, TypeError, "candidate_k is not of type int"),
        (
            {"top_k": 3, "candidate_k": 2},
            ValueError,
            "candidate_k must be at least top_k",
        ),
        ({"rank_constant": True}, TypeError, "rank_constant is not of type int"),
        (
            {"rank_constant": 0},
            ValueError,
            "rank_constant has to be a positive integer",
        ),
        (
            {"overlap_threshold": True},
            TypeError,
            "overlap_threshold is not of type numeric",
        ),
        (
            {"overlap_threshold": 0},
            ValueError,
            "overlap_threshold has to be between 0 and 1",
        ),
        ({"max_per_file": True}, TypeError, "max_per_file is not of type int"),
        (
            {"max_per_file": 0},
            ValueError,
            "max_per_file has to be a positive integer",
        ),
        ({"k1": True}, TypeError, "k1 is not of type numeric"),
        ({"k1": 0}, ValueError, "k1 has to be finite and positive"),
        ({"b": True}, TypeError, "b is not of type numeric"),
        ({"b": 1.1}, ValueError, "b has to be finite and between 0 and 1"),
    ],
)
def test_retrieve_hybrid_validates_before_retrieval(
    tmp_path, kwargs, error_type, message
):
    with pytest.raises(error_type, match=message):
        retrieve_hybrid(
            tmp_path,
            "demo",
            "question",
            QueryEmbedder([1.0, 0.0]),
            **kwargs,
        )


def test_retrieve_hybrid_rejects_empty_question_before_retrieval(tmp_path):
    with pytest.raises(ValueError, match="question cannot be empty"):
        retrieve_hybrid(
            tmp_path,
            "demo",
            "   ",
            QueryEmbedder([1.0, 0.0]),
        )
