import pytest

from contextforge.evaluation import (
    EvaluationCaseResult,
    EvaluationSummary,
    evaluate_case,
    first_relevant_rank,
    hit_at_k,
    hit_rate_at_k,
    mean_reciprocal_rank,
    reciprocal_rank,
    summarize_results,
)


def test_first_relevant_rank_returns_one_for_first_result():
    assert first_relevant_rank(["request.go"], ["request.go"]) == 1


def test_first_relevant_rank_returns_one_based_position():
    retrieved_files = ["README.md", "request.go"]

    assert first_relevant_rank(retrieved_files, ["request.go"]) == 2


def test_first_relevant_rank_returns_earliest_of_multiple_expected_files():
    retrieved_files = ["README.md", "request.go", "server.go"]
    expected_files = ["server.go", "request.go"]

    assert first_relevant_rank(retrieved_files, expected_files) == 2


def test_first_relevant_rank_returns_none_when_no_expected_file_is_retrieved():
    retrieved_files = ["README.md", "main.go"]

    assert first_relevant_rank(retrieved_files, ["request.go"]) is None


def test_first_relevant_rank_rejects_empty_expected_files():
    with pytest.raises(ValueError, match="expected_files cannot be empty"):
        first_relevant_rank(["README.md"], [])


def test_reciprocal_rank_returns_zero_for_no_relevant_result():
    assert reciprocal_rank(None) == 0.0


@pytest.mark.parametrize(
    ("rank", "expected_score"),
    [
        (1, 1.0),
        (2, 0.5),
        (3, 1 / 3),
    ],
)
def test_reciprocal_rank_returns_inverse_of_positive_rank(rank, expected_score):
    assert reciprocal_rank(rank) == pytest.approx(expected_score)


@pytest.mark.parametrize("rank", [0, -1])
def test_reciprocal_rank_rejects_non_positive_ranks(rank):
    with pytest.raises(ValueError, match="rank has to be greater than 0"):
        reciprocal_rank(rank)


@pytest.mark.parametrize("rank", [True, False, 1.5, "2"])
def test_reciprocal_rank_rejects_non_integer_ranks(rank):
    with pytest.raises(TypeError, match="is not of type int"):
        reciprocal_rank(rank)


def test_mean_reciprocal_rank_averages_question_scores():
    assert mean_reciprocal_rank([1, 2, None]) == pytest.approx(0.5)


def test_mean_reciprocal_rank_returns_zero_when_all_questions_miss():
    assert mean_reciprocal_rank([None, None]) == 0.0


def test_mean_reciprocal_rank_rejects_empty_ranks():
    with pytest.raises(ValueError, match="ranks cannot be empty"):
        mean_reciprocal_rank([])


@pytest.mark.parametrize(
    ("ranks", "error_type", "message"),
    [
        ([0], ValueError, "rank has to be greater than 0"),
        ([True], TypeError, "is not of type int"),
    ],
)
def test_mean_reciprocal_rank_reuses_rank_validation(ranks, error_type, message):
    with pytest.raises(error_type, match=message):
        mean_reciprocal_rank(ranks)


@pytest.mark.parametrize(
    ("rank", "k", "expected"),
    [
        (1, 3, True),
        (3, 3, True),
        (4, 3, False),
        (None, 3, False),
    ],
)
def test_hit_at_k_reports_whether_relevant_result_is_within_limit(
    rank, k, expected
):
    assert hit_at_k(rank, k) is expected


@pytest.mark.parametrize("rank", [0, -1])
def test_hit_at_k_rejects_non_positive_ranks(rank):
    with pytest.raises(ValueError, match="rank has to be greater than 0"):
        hit_at_k(rank, 3)


@pytest.mark.parametrize("rank", [True, False, 1.5, "2"])
def test_hit_at_k_rejects_non_integer_ranks(rank):
    with pytest.raises(TypeError, match="is not of type int"):
        hit_at_k(rank, 3)


@pytest.mark.parametrize("k", [0, -1])
def test_hit_at_k_rejects_non_positive_k_even_when_rank_is_none(k):
    with pytest.raises(ValueError, match="k has to be greater than 0"):
        hit_at_k(None, k)


@pytest.mark.parametrize("k", [True, False, 1.5, "3"])
def test_hit_at_k_rejects_non_integer_k_even_when_rank_is_none(k):
    with pytest.raises(TypeError, match="is not of type int"):
        hit_at_k(None, k)


def test_hit_rate_at_k_returns_fraction_of_questions_with_hits():
    assert hit_rate_at_k([1, 4, None], 3) == pytest.approx(1 / 3)


def test_hit_rate_at_k_returns_zero_when_all_questions_miss():
    assert hit_rate_at_k([None, None], 3) == 0.0


def test_hit_rate_at_k_returns_one_when_all_questions_hit():
    assert hit_rate_at_k([1, 2, 3], 3) == 1.0


def test_hit_rate_at_k_rejects_empty_ranks():
    with pytest.raises(ValueError, match="ranks cannot be empty"):
        hit_rate_at_k([], 3)


@pytest.mark.parametrize(
    ("ranks", "k", "error_type", "message"),
    [
        ([0], 3, ValueError, "rank has to be greater than 0"),
        ([True], 3, TypeError, "is not of type int"),
        ([None], 0, ValueError, "k has to be greater than 0"),
        ([None], True, TypeError, "is not of type int"),
    ],
)
def test_hit_rate_at_k_reuses_hit_validation(
    ranks, k, error_type, message
):
    with pytest.raises(error_type, match=message):
        hit_rate_at_k(ranks, k)


def test_evaluate_case_returns_ranked_result():
    result = evaluate_case(
        question_id="partial-reads",
        retrieved_files=["README.md", "internal/request/request.go"],
        expected_files=["internal/request/request.go"],
    )

    assert result == EvaluationCaseResult(
        question_id="partial-reads",
        retrieved_files=["README.md", "internal/request/request.go"],
        expected_files=["internal/request/request.go"],
        first_relevant_rank=2,
    )


def test_evaluate_case_records_none_when_retrieval_misses():
    result = evaluate_case(
        question_id="partial-reads",
        retrieved_files=["README.md"],
        expected_files=["internal/request/request.go"],
    )

    assert result.first_relevant_rank is None


@pytest.mark.parametrize("question_id", ["", "   "])
def test_evaluate_case_rejects_empty_question_id(question_id):
    with pytest.raises(ValueError, match="question_id cannot be empty"):
        evaluate_case(question_id, ["README.md"], ["request.go"])


def test_evaluate_case_rejects_empty_expected_files():
    with pytest.raises(ValueError, match="expected_files cannot be empty"):
        evaluate_case("partial-reads", ["README.md"], [])


def test_summarize_results_aggregates_case_metrics():
    results = [
        EvaluationCaseResult("first", ["a.go"], ["a.go"], 1),
        EvaluationCaseResult("second", ["x.go", "b.go"], ["b.go"], 2),
        EvaluationCaseResult("third", ["x.go"], ["c.go"], None),
    ]

    summary = summarize_results(results, k=3)

    assert summary.total_questions == 3
    assert summary.k == 3
    assert summary.hits == 2
    assert summary.hit_rate == pytest.approx(2 / 3)
    assert summary.mrr == pytest.approx(0.5)


def test_summarize_results_returns_zero_metrics_when_all_cases_miss():
    results = [
        EvaluationCaseResult("first", ["x.go"], ["a.go"], None),
        EvaluationCaseResult("second", ["y.go"], ["b.go"], None),
    ]

    assert summarize_results(results, k=3) == EvaluationSummary(
        total_questions=2,
        k=3,
        hits=0,
        hit_rate=0.0,
        mrr=0.0,
    )


def test_summarize_results_rejects_empty_results():
    with pytest.raises(ValueError, match="results cannot be empty"):
        summarize_results([], k=3)


@pytest.mark.parametrize(
    ("k", "error_type", "message"),
    [
        (0, ValueError, "k has to be greater than 0"),
        (True, TypeError, "is not of type int"),
    ],
)
def test_summarize_results_reuses_k_validation(k, error_type, message):
    results = [EvaluationCaseResult("miss", ["x.go"], ["a.go"], None)]

    with pytest.raises(error_type, match=message):
        summarize_results(results, k=k)
