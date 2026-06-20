import pytest

from contextforge.models import Chunk, SearchResult
from contextforge.ranking import (
    deduplicate_overlapping_results,
    deduplicate_results,
    limit_results_per_file,
    line_overlap_ratio,
)


def make_chunk(
    chunk_id: str,
    start_line: int,
    end_line: int,
    file_path: str = "src/example.py",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        project_name="demo",
        file_path=file_path,
        language="python",
        content=f"content for {chunk_id}",
        start_line=start_line,
        end_line=end_line,
        embedding=[1.0, 0.0],
    )


def make_result(
    chunk_id: str,
    score: float,
    file_path: str = "src/example.py",
    start_line: int = 1,
    end_line: int = 1,
) -> SearchResult:
    chunk = make_chunk(chunk_id, start_line, end_line, file_path)
    return SearchResult(chunk=chunk, score=score)


def test_deduplicate_results_returns_empty_list_for_empty_input():
    assert deduplicate_results([]) == []


def test_deduplicate_results_leaves_unique_results_unchanged():
    results = [make_result("chunk-a", 0.9), make_result("chunk-b", 0.8)]

    assert deduplicate_results(results) == results


def test_deduplicate_results_keeps_first_result_for_duplicate_chunk_id():
    first = make_result("chunk-a", 0.9)
    duplicate = make_result("chunk-a", 0.7)

    assert deduplicate_results([first, duplicate]) == [first]


def test_deduplicate_results_preserves_ranking_order():
    first = make_result("chunk-a", 0.9)
    second = make_result("chunk-b", 0.8)
    duplicate = make_result("chunk-a", 0.7)
    third = make_result("chunk-c", 0.6)

    assert deduplicate_results([first, second, duplicate, third]) == [
        first,
        second,
        third,
    ]


def test_deduplicate_results_keeps_different_chunks_from_same_file():
    first = make_result("chunk-a", 0.9, file_path="src/parser.py")
    second = make_result("chunk-b", 0.8, file_path="src/parser.py")

    assert deduplicate_results([first, second]) == [first, second]


def test_deduplicate_results_does_not_mutate_input_list():
    first = make_result("chunk-a", 0.9)
    duplicate = make_result("chunk-a", 0.7)
    results = [first, duplicate]
    original_results = results.copy()

    deduplicate_results(results)

    assert results == original_results


def test_line_overlap_ratio_returns_zero_for_different_files():
    first = make_chunk("first", 1, 20, "src/first.py")
    second = make_chunk("second", 1, 20, "src/second.py")

    assert line_overlap_ratio(first, second) == 0.0


def test_line_overlap_ratio_returns_one_for_identical_ranges():
    first = make_chunk("first", 1, 20)
    second = make_chunk("second", 1, 20)

    assert line_overlap_ratio(first, second) == 1.0


def test_line_overlap_ratio_handles_one_line_chunks():
    first = make_chunk("first", 8, 8)
    second = make_chunk("second", 8, 8)

    assert line_overlap_ratio(first, second) == 1.0


def test_line_overlap_ratio_calculates_partial_overlap():
    first = make_chunk("first", 1, 20)
    second = make_chunk("second", 11, 30)

    assert line_overlap_ratio(first, second) == 0.5


def test_line_overlap_ratio_returns_one_for_contained_chunk():
    outer = make_chunk("outer", 1, 20)
    inner = make_chunk("inner", 5, 10)

    assert line_overlap_ratio(outer, inner) == 1.0


def test_line_overlap_ratio_returns_zero_for_separate_ranges():
    first = make_chunk("first", 1, 10)
    second = make_chunk("second", 11, 20)

    assert line_overlap_ratio(first, second) == 0.0


def test_line_overlap_ratio_is_symmetric():
    first = make_chunk("first", 1, 20)
    second = make_chunk("second", 11, 30)

    assert line_overlap_ratio(first, second) == line_overlap_ratio(second, first)


def test_deduplicate_overlapping_results_returns_empty_list():
    assert deduplicate_overlapping_results([]) == []


def test_deduplicate_overlapping_results_removes_at_threshold():
    first = make_result("first", 0.9, start_line=1, end_line=20)
    overlapping = make_result("overlap", 0.8, start_line=11, end_line=30)

    assert deduplicate_overlapping_results([first, overlapping], 0.5) == [first]


def test_deduplicate_overlapping_results_keeps_below_threshold():
    first = make_result("first", 0.9, start_line=1, end_line=20)
    overlapping = make_result("overlap", 0.8, start_line=11, end_line=30)

    assert deduplicate_overlapping_results([first, overlapping], 0.6) == [
        first,
        overlapping,
    ]


def test_deduplicate_overlapping_results_keeps_different_files():
    first = make_result("first", 0.9, "src/first.py", 1, 20)
    second = make_result("second", 0.8, "src/second.py", 1, 20)

    assert deduplicate_overlapping_results([first, second]) == [first, second]


def test_deduplicate_overlapping_results_removes_exact_chunk_ids_first():
    first = make_result("same-id", 0.9, "src/first.py", 1, 20)
    duplicate = make_result("same-id", 0.8, "src/second.py", 1, 20)

    assert deduplicate_overlapping_results([first, duplicate]) == [first]


def test_deduplicate_overlapping_results_preserves_order_and_input():
    first = make_result("first", 0.9, start_line=1, end_line=20)
    overlapping = make_result("overlap", 0.8, start_line=11, end_line=30)
    third = make_result("third", 0.7, start_line=31, end_line=40)
    results = [first, overlapping, third]
    original_results = results.copy()

    assert deduplicate_overlapping_results(results) == [first, third]
    assert results == original_results


@pytest.mark.parametrize("threshold", [0, -0.1, 1.1])
def test_deduplicate_overlapping_results_rejects_out_of_bounds_threshold(
    threshold,
):
    with pytest.raises(ValueError, match="is out of bounds"):
        deduplicate_overlapping_results([], threshold)


@pytest.mark.parametrize("threshold", [True, False, "0.5", None])
def test_deduplicate_overlapping_results_rejects_non_numeric_threshold(
    threshold,
):
    with pytest.raises(TypeError, match="is not of type numeric"):
        deduplicate_overlapping_results([], threshold)


def test_limit_results_per_file_returns_empty_list_for_empty_input():
    assert limit_results_per_file([], max_per_file=1) == []


def test_limit_results_per_file_keeps_one_result_from_each_file():
    first_a = make_result("a-1", 0.9, "src/a.py")
    second_a = make_result("a-2", 0.8, "src/a.py")
    first_b = make_result("b-1", 0.7, "src/b.py")

    assert limit_results_per_file([first_a, second_a, first_b], 1) == [
        first_a,
        first_b,
    ]


def test_limit_results_per_file_keeps_configured_number_per_file():
    first_a = make_result("a-1", 0.9, "src/a.py")
    first_b = make_result("b-1", 0.8, "src/b.py")
    second_a = make_result("a-2", 0.7, "src/a.py")
    third_a = make_result("a-3", 0.6, "src/a.py")
    second_b = make_result("b-2", 0.5, "src/b.py")

    assert limit_results_per_file(
        [first_a, first_b, second_a, third_a, second_b],
        2,
    ) == [first_a, first_b, second_a, second_b]


def test_limit_results_per_file_preserves_input_list():
    first = make_result("a-1", 0.9, "src/a.py")
    second = make_result("a-2", 0.8, "src/a.py")
    results = [first, second]
    original_results = results.copy()

    limit_results_per_file(results, 1)

    assert results == original_results


@pytest.mark.parametrize("max_per_file", [True, False, 1.5, "1"])
def test_limit_results_per_file_rejects_non_integer_limit(max_per_file):
    with pytest.raises(TypeError, match="is not of type int"):
        limit_results_per_file([], max_per_file)


@pytest.mark.parametrize("max_per_file", [0, -1])
def test_limit_results_per_file_rejects_non_positive_limit(max_per_file):
    with pytest.raises(ValueError, match="should be a positive integer"):
        limit_results_per_file([], max_per_file)
