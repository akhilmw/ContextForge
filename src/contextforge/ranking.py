"""Post-process ranked retrieval candidates without changing their scores."""

from collections import defaultdict

from contextforge.models import Chunk, SearchResult


def deduplicate_results(
    results: list[SearchResult],
) -> list[SearchResult]:
    """Keep the first ranked result for each unique chunk ID."""

    chunk_ids: set[str] = set()
    new_results: list[SearchResult] = []

    for search_result in results:
        if search_result.chunk.chunk_id not in chunk_ids:
            new_results.append(search_result)
            chunk_ids.add(search_result.chunk.chunk_id)

    return new_results


def line_overlap_ratio(first: Chunk, second: Chunk) -> float:
    """Return the fraction of the shorter chunk overlapped within one file."""

    if first.file_path != second.file_path:
        return 0.0

    # Source line ranges are inclusive, so both overlap and lengths need +1.
    overlap_start = max(first.start_line, second.start_line)
    overlap_end = min(first.end_line, second.end_line)
    overlap_lines = max(0, overlap_end - overlap_start + 1)
    first_length = first.end_line - first.start_line + 1
    second_length = second.end_line - second.start_line + 1
    shorter_chunk_length = min(first_length, second_length)

    return overlap_lines / shorter_chunk_length


def deduplicate_overlapping_results(
    results: list[SearchResult],
    threshold: float = 0.5,
) -> list[SearchResult]:
    """Keep ranked results that do not overlap retained chunks by threshold."""

    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError(f"{threshold} is not of type numeric")

    if threshold <= 0 or threshold > 1:
        raise ValueError(f"threshold {threshold} is out of bounds")

    if not results:
        return []

    kept: list[SearchResult] = []

    # Input order is ranking order, so earlier retained chunks take precedence.
    for candidate in deduplicate_results(results):
        overlaps_kept = any(
            line_overlap_ratio(candidate.chunk, existing.chunk) >= threshold
            for existing in kept
        )

        if not overlaps_kept:
            kept.append(candidate)

    return kept


def limit_results_per_file(
    results: list[SearchResult],
    max_per_file: int,
) -> list[SearchResult]:
    """Keep at most the configured number of ranked chunks from each file."""

    if isinstance(max_per_file, bool) or not isinstance(max_per_file, int):
        raise TypeError(f"{max_per_file} is not of type int")

    if max_per_file <= 0:
        raise ValueError("max_per_file should be a positive integer")

    if not results:
        return []

    # Counts are independent per path, while input order preserves ranking.
    count_per_file: defaultdict[str, int] = defaultdict(int)

    final_results: list[SearchResult] = []
    for result in results:
        file_path = result.chunk.file_path
        if count_per_file[file_path] < max_per_file:
            final_results.append(result)
            count_per_file[file_path] += 1

    return final_results
