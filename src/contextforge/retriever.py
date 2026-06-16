"""Retrieve the most relevant stored chunks for a user question."""

import math
from pathlib import Path

from contextforge.embedder import Embedder
from contextforge.models import SearchResult
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
