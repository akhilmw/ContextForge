from typing import Protocol

from contextforge.keyword_retriever import tokenize
from contextforge.models import SearchResult


class Reranker(Protocol):
    """Reorder a retrieval shortlist for a user question."""

    def rerank(
        self,
        question: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        ...


def validate_reranked_results(
    candidates: list[SearchResult],
    reranked: list[SearchResult],
) -> None:
    """Ensure reranking only reorders the original candidate set."""

    candidate_ids = [result.chunk.chunk_id for result in candidates]
    reranked_ids = [result.chunk.chunk_id for result in reranked]

    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidates contain duplicate chunk IDs")

    if len(reranked_ids) != len(set(reranked_ids)):
        raise ValueError("reranked results contain duplicate chunk IDs")

    if set(candidate_ids) != set(reranked_ids):
        raise ValueError("reranked results must contain exactly the candidate chunks")


class FakeReranker:
    """Deterministic reranker used by tests before real rerankers exist."""

    def __init__(self, ordered_chunk_ids: list[str]):
        self.ordered_chunk_ids = ordered_chunk_ids

    def rerank(
        self,
        question: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        """Move configured chunk IDs first, then keep remaining candidate order."""

        if not question.strip():
            raise ValueError("question cannot be empty")

        if not candidates:
            return []

        candidates_by_id = {result.chunk.chunk_id: result for result in candidates}
        ordered: list[SearchResult] = []

        # Test fixtures specify the preferred chunk order. Unknown IDs are
        # ignored so one fake reranker can be reused across smaller shortlists.
        for chunk_id in self.ordered_chunk_ids:
            candidate = candidates_by_id.get(chunk_id)
            if candidate is not None and candidate not in ordered:
                ordered.append(candidate)

        ordered_ids = {result.chunk.chunk_id for result in ordered}
        ordered.extend(
            result
            for result in candidates
            if result.chunk.chunk_id not in ordered_ids
        )

        validate_reranked_results(candidates, ordered)
        return ordered


def heuristic_score(
    question_terms: set[str],
    result: SearchResult,
    original_rank: int,
) -> float:
    """Score one candidate using lightweight code-aware ranking signals."""

    content_terms = set(tokenize(result.chunk.content))
    path_terms = set(tokenize(result.chunk.file_path))

    # Coverage rewards chunks that mention more of the user's question terms.
    coverage = len(question_terms & content_terms) / len(question_terms)

    # Path matches help when the question names a file, package, or concept
    # that appears in the repository path.
    path_match = len(question_terms & path_terms)

    # Implementation files often answer behavior questions better than tests
    # or docs, but this is only a small bonus so evidence can still win.
    implementation_bonus = (
        1.0
        if not result.chunk.file_path.endswith(("_test.go", "_test.py", ".md", ".txt"))
        else 0.0
    )

    # Original rank is a tie-breaker so the upstream retriever still matters.
    rank_bonus = 1.0 / original_rank

    return (
        coverage * 3.0
        + path_match * 2.0
        + implementation_bonus * 0.5
        + rank_bonus * 0.25
    )


class HeuristicReranker:
    """Rerank candidates with local rules instead of a model/API call."""

    def rerank(
        self,
        question: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        """Apply a simple heuristic to reorder the candidates."""

        if not question.strip():
            raise ValueError("question cannot be empty")

        if not candidates:
            return []

        query_terms = set(tokenize(question))
        if not query_terms:
            raise ValueError("question must contain searchable terms")

        scored: list[tuple[float, SearchResult]] = []

        # The reranker only changes ordering. It keeps the original SearchResult
        # objects so source metadata and upstream scores remain intact.
        for rank, result in enumerate(candidates, start=1):
            score = heuristic_score(query_terms, result, rank)
            scored.append((score, result))

        scored.sort(key=lambda x: x[0], reverse=True)
        reranked = [result for _, result in scored]
        validate_reranked_results(candidates, reranked)
        return reranked
