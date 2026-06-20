"""Retrieve the most relevant stored chunks for a user question."""

import math
from pathlib import Path

from contextforge.embedder import Embedder
from contextforge.models import SearchResult
from contextforge.ranking import (
    deduplicate_overlapping_results,
    limit_results_per_file,
)
from contextforge.store import load_chunks


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return cosine similarity for two finite, non-zero vectors."""
    if not a or not b:
        raise ValueError("Vectors cannot be empty")
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimension")

    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in [*a, *b]):
        raise TypeError("Vector values must be finite numbers")

    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors")

    return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)


def retrieve(
    data_dir: Path,
    project_name: str,
    question: str,
    embedder: Embedder,
    top_k: int = 5,
) -> list[SearchResult]:
    """Embed the question, score stored chunks, and return the top matches."""

    if not question.strip():
        raise ValueError("question cannot be empty")

    if top_k <= 0:
        raise ValueError("top k cannot be less than or equal to zero")

    chunks = load_chunks(data_dir, project_name)

    if not chunks:
        return []

    # The query vector must live in the same embedding space as stored chunks.
    question_vector = embedder.embed_query(question)

    results = []
    for chunk in chunks:
        similarity_score = cosine_similarity(question_vector, chunk.embedding)
        results.append(SearchResult(chunk=chunk, score=similarity_score))

    results.sort(key=lambda x: x.score, reverse=True)

    return results[:top_k]


def retrieve_deduplicated(
    data_dir: Path,
    project_name: str,
    question: str,
    embedder: Embedder,
    top_k: int = 5,
    candidate_k: int = 15,
    overlap_threshold: float = 0.5,
) -> list[SearchResult]:
    """Overfetch semantic candidates, remove overlap, and return final top-k."""

    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError(f"{top_k} is not of type int")

    if top_k <= 0:
        raise ValueError("top k cannot be less than or equal to zero")

    if isinstance(candidate_k, bool) or not isinstance(candidate_k, int):
        raise TypeError(f"{candidate_k} is not of type int")

    if candidate_k < top_k:
        raise ValueError("candidate_k must be at least top_k")

    if isinstance(overlap_threshold, bool) or not isinstance(
        overlap_threshold, (int, float)
    ):
        raise TypeError(f"{overlap_threshold} is not of type numeric")

    if overlap_threshold <= 0 or overlap_threshold > 1:
        raise ValueError(f"threshold {overlap_threshold} is out of bounds")

    # Fetch extra candidates so removed overlaps can be replaced before slicing.
    candidates = retrieve(data_dir, project_name, question, embedder, candidate_k)
    deduplicated = deduplicate_overlapping_results(candidates, overlap_threshold)

    return deduplicated[:top_k]


def retrieve_diverse(
    data_dir: Path,
    project_name: str,
    question: str,
    embedder: Embedder,
    top_k: int = 5,
    candidate_k: int = 15,
    overlap_threshold: float = 0.5,
    max_per_file: int = 1,
) -> list[SearchResult]:
    """Overfetch, remove overlap, limit each file, and return final top-k."""

    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError(f"{top_k} is not of type int")

    if top_k <= 0:
        raise ValueError("top k cannot be less than or equal to zero")

    if isinstance(candidate_k, bool) or not isinstance(candidate_k, int):
        raise TypeError(f"{candidate_k} is not of type int")

    if candidate_k < top_k:
        raise ValueError("candidate_k must be at least top_k")

    if isinstance(overlap_threshold, bool) or not isinstance(
        overlap_threshold, (int, float)
    ):
        raise TypeError(f"{overlap_threshold} is not of type numeric")

    if overlap_threshold <= 0 or overlap_threshold > 1:
        raise ValueError(f"threshold {overlap_threshold} is out of bounds")

    if isinstance(max_per_file, bool) or not isinstance(max_per_file, int):
        raise TypeError(f"{max_per_file} is not of type int")

    if max_per_file <= 0:
        raise ValueError("max_per_file cannot be less than or equal to zero")

    # Diversity is applied after overlap removal so redundant ranges do not use
    # a file's result allowance before final top-k selection.
    candidates = retrieve(data_dir, project_name, question, embedder, candidate_k)
    deduplicated = deduplicate_overlapping_results(candidates, overlap_threshold)
    max_files = limit_results_per_file(deduplicated, max_per_file)

    return max_files[:top_k]
