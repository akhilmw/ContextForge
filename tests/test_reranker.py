import pytest

from contextforge.models import Chunk, SearchResult
from contextforge.reranker import (
    FakeReranker,
    HeuristicReranker,
    heuristic_score,
    validate_reranked_results,
)


def make_result(
    chunk_id: str,
    score: float,
    file_path: str | None = None,
    content: str | None = None,
) -> SearchResult:
    chunk = Chunk(
        chunk_id=chunk_id,
        project_name="demo",
        file_path=file_path or f"src/{chunk_id}.py",
        language="python",
        content=content or f"content for {chunk_id}",
        start_line=1,
        end_line=3,
        embedding=[1.0, 0.0],
    )
    return SearchResult(chunk=chunk, score=score)


def test_validate_reranked_results_accepts_same_chunks_in_new_order():
    first = make_result("first", 0.9)
    second = make_result("second", 0.8)
    third = make_result("third", 0.7)

    validate_reranked_results(
        candidates=[first, second, third],
        reranked=[third, first, second],
    )


def test_validate_reranked_results_rejects_duplicate_candidates():
    first = make_result("same", 0.9)
    duplicate = make_result("same", 0.8)

    with pytest.raises(ValueError, match="candidates contain duplicate chunk IDs"):
        validate_reranked_results([first, duplicate], [first, duplicate])


def test_validate_reranked_results_rejects_duplicate_reranked_results():
    first = make_result("first", 0.9)
    second = make_result("second", 0.8)

    with pytest.raises(
        ValueError,
        match="reranked results contain duplicate chunk IDs",
    ):
        validate_reranked_results([first, second], [first, first])


def test_validate_reranked_results_rejects_missing_candidate():
    first = make_result("first", 0.9)
    second = make_result("second", 0.8)

    with pytest.raises(
        ValueError,
        match="reranked results must contain exactly the candidate chunks",
    ):
        validate_reranked_results([first, second], [first])


def test_validate_reranked_results_rejects_new_candidate():
    first = make_result("first", 0.9)
    second = make_result("second", 0.8)
    unknown = make_result("unknown", 0.1)

    with pytest.raises(
        ValueError,
        match="reranked results must contain exactly the candidate chunks",
    ):
        validate_reranked_results([first, second], [first, unknown])


def test_fake_reranker_moves_configured_chunks_first():
    first = make_result("first", 0.9)
    second = make_result("second", 0.8)
    third = make_result("third", 0.7)
    reranker = FakeReranker(["third", "first"])

    reranked = reranker.rerank("How does parsing work?", [first, second, third])

    assert reranked == [third, first, second]


def test_fake_reranker_ignores_unknown_ordered_chunk_ids():
    first = make_result("first", 0.9)
    second = make_result("second", 0.8)
    reranker = FakeReranker(["missing", "second"])

    reranked = reranker.rerank("How does parsing work?", [first, second])

    assert reranked == [second, first]


def test_fake_reranker_preserves_existing_result_objects():
    first = make_result("first", 0.9)
    second = make_result("second", 0.8)
    reranker = FakeReranker(["second"])

    reranked = reranker.rerank("How does parsing work?", [first, second])

    assert reranked[0] is second
    assert reranked[1] is first


def test_fake_reranker_returns_empty_list_for_no_candidates():
    reranker = FakeReranker(["first"])

    assert reranker.rerank("How does parsing work?", []) == []


@pytest.mark.parametrize("question", ["", "   "])
def test_fake_reranker_rejects_empty_question(question):
    reranker = FakeReranker(["first"])

    with pytest.raises(ValueError, match="question cannot be empty"):
        reranker.rerank(question, [make_result("first", 0.9)])


def test_heuristic_score_rewards_content_coverage():
    weak = make_result("weak", 0.9, content="parse request")
    strong = make_result("strong", 0.8, content="parse request header body")

    weak_score = heuristic_score({"parse", "request", "header", "body"}, weak, 1)
    strong_score = heuristic_score({"parse", "request", "header", "body"}, strong, 2)

    assert strong_score > weak_score


def test_heuristic_score_rewards_path_matches():
    generic = make_result("generic", 0.9, file_path="src/server.py")
    path_match = make_result("path-match", 0.8, file_path="internal/request/parser.py")

    generic_score = heuristic_score({"request"}, generic, 1)
    path_match_score = heuristic_score({"request"}, path_match, 2)

    assert path_match_score > generic_score


def test_heuristic_score_gives_small_bonus_to_implementation_files():
    implementation = make_result("impl", 0.8, file_path="internal/request/request.go")
    test_file = make_result("test", 0.9, file_path="internal/request/request_test.go")

    implementation_score = heuristic_score({"request"}, implementation, 1)
    test_score = heuristic_score({"request"}, test_file, 1)

    assert implementation_score > test_score


def test_heuristic_reranker_reorders_candidates_by_heuristic_score():
    docs = make_result(
        "docs",
        0.9,
        file_path="docs/architecture.md",
        content="overview of the project",
    )
    request_parser = make_result(
        "request-parser",
        0.7,
        file_path="internal/request/parser.go",
        content="parse request header body",
    )
    reranker = HeuristicReranker()

    reranked = reranker.rerank("How does request parser handle headers?", [docs, request_parser])

    assert reranked == [request_parser, docs]


def test_heuristic_reranker_preserves_input_order_when_scores_tie():
    first = make_result("first", 0.9, content="parse request")
    second = make_result("second", 0.8, content="parse request")
    reranker = HeuristicReranker()

    reranked = reranker.rerank("parse request", [first, second])

    assert reranked == [first, second]


def test_heuristic_reranker_returns_empty_list_for_no_candidates():
    reranker = HeuristicReranker()

    assert reranker.rerank("How does parsing work?", []) == []


@pytest.mark.parametrize("question", ["", "   "])
def test_heuristic_reranker_rejects_empty_question(question):
    reranker = HeuristicReranker()

    with pytest.raises(ValueError, match="question cannot be empty"):
        reranker.rerank(question, [make_result("first", 0.9)])


def test_heuristic_reranker_rejects_question_with_no_searchable_terms():
    reranker = HeuristicReranker()

    with pytest.raises(ValueError, match="question must contain searchable terms"):
        reranker.rerank("???", [make_result("first", 0.9)])
