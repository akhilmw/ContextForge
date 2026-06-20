"""Calculate deterministic metrics for retrieval evaluation results."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationCaseResult:
    """The ranked retrieval outcome for one evaluation question.

    Attributes:
        question_id: Stable identifier from the evaluation dataset.
        retrieved_files: Source paths returned in retrieval order.
        expected_files: Source paths considered relevant for the question.
        first_relevant_rank: One-based rank of the first relevant path, or None.
    """

    question_id: str
    retrieved_files: list[str]
    expected_files: list[str]
    first_relevant_rank: int | None


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate retrieval metrics for an evaluation run.

    Attributes:
        total_questions: Number of evaluated questions.
        k: Retrieval cutoff used to count hits.
        hits: Number of questions with a relevant result by k.
        hit_rate: Fraction of questions with a relevant result by k.
        mrr: Mean reciprocal rank across all questions.
    """

    total_questions: int
    k: int
    hits: int
    hit_rate: float
    mrr: float


def first_relevant_rank(
    retrieved_files: list[str],
    expected_files: list[str],
) -> int | None:
    """Return the one-based rank of the first expected file, or None."""

    if not expected_files:
        raise ValueError("expected_files cannot be empty")

    # A set makes relevance independent of expected-files ordering.
    expected = set(expected_files)

    # Retrieval order determines which relevant file has the best rank.
    for rank, file_path in enumerate(retrieved_files, start=1):
        if file_path in expected:
            return rank

    return None


def reciprocal_rank(rank: int | None) -> float:
    """Return the reciprocal of a positive rank, or zero for no match."""
    if rank is None:
        return 0.0

    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError(f"{rank} is not of type int")

    if rank <= 0:
        raise ValueError("rank has to be greater than 0")

    return 1 / rank


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    """On average, how quickly does my system find the first correct answer?"""

    if not ranks:
        raise ValueError("ranks cannot be empty")

    scores = [reciprocal_rank(rank) for rank in ranks]
    return sum(scores) / len(scores)


def hit_at_k(rank: int | None, k: int) -> bool:
    """Did the first relevant result appear within the first k results?"""
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"{k} is not of type int")

    if k <= 0:
        raise ValueError("k has to be greater than 0")

    if rank is None:
        return False

    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError(f"{rank} is not of type int")

    if rank <= 0:
        raise ValueError("rank has to be greater than 0")

    return rank <= k


def hit_rate_at_k(ranks: list[int | None], k: int) -> float:
    """Return the fraction of retrieval questions with a relevant result by k."""

    if not ranks:
        raise ValueError("ranks cannot be empty")

    hits = [hit_at_k(rank, k) for rank in ranks]
    return sum(hits) / len(hits)


def evaluate_case(
    question_id: str,
    retrieved_files: list[str],
    expected_files: list[str],
) -> EvaluationCaseResult:
    """Evaluate one ranked file list against its expected relevant files."""

    if not question_id.strip():
        raise ValueError("question_id cannot be empty")

    rank = first_relevant_rank(retrieved_files, expected_files)

    return EvaluationCaseResult(
        question_id=question_id,
        retrieved_files=retrieved_files,
        expected_files=expected_files,
        first_relevant_rank=rank,
    )


def summarize_results(
    results: list[EvaluationCaseResult],
    k: int,
) -> EvaluationSummary:
    """Aggregate individual evaluation results into run-level metrics."""

    if not results:
        raise ValueError("results cannot be empty")

    ranks = [result.first_relevant_rank for result in results]
    hits = sum(hit_at_k(rank, k) for rank in ranks)
    hit_rate = hit_rate_at_k(ranks, k)
    mrr = mean_reciprocal_rank(ranks)

    return EvaluationSummary(
        total_questions=len(results),
        k=k,
        hits=hits,
        hit_rate=hit_rate,
        mrr=mrr,
    )
