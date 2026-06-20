"""Retrieve stored chunks using token-based keyword relevance."""

import math
import re
from pathlib import Path
from collections import Counter
from contextforge.models import SearchResult
from contextforge.store import load_chunks


RAW_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")
IDENTIFIER_PART_PATTERN = re.compile(
    r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z0-9]|\b)"
)


def tokenize(text: str) -> list[str]:
    """Normalize text while preserving and splitting code identifiers."""

    if not text:
        return []

    # Underscores remain in raw tokens so exact snake_case matches are retained.
    raw_tokens = RAW_TOKEN_PATTERN.findall(text)

    result: list[str] = []
    for token in raw_tokens:
        # Preserve the full normalized identifier before adding its components.
        lower_token = token.lower()
        result.append(lower_token)

        camel_parts = IDENTIFIER_PART_PATTERN.findall(token)

        sub_parts: list[str] = []
        if "_" in token:
            for part in token.split("_"):
                if part:
                    nested_camel = IDENTIFIER_PART_PATTERN.findall(part)
                    if nested_camel and len(nested_camel) > 1:
                        sub_parts.extend(nested_camel)
                    else:
                        sub_parts.append(part)

        elif camel_parts and len(camel_parts) > 1:
            sub_parts.extend(camel_parts)

        for sub in sub_parts:
            low_sub = sub.lower()
            if low_sub != lower_token:
                result.append(low_sub)

    return result


def term_frequencies(tokens: list[str]) -> dict[str, int]:
    """Return the raw occurrence count for each token in one document."""

    if not tokens:
        return {}

    return dict(Counter(tokens))


def document_frequencies(
    tokenized_documents: list[list[str]],
) -> dict[str, int]:
    """Return the number of documents containing each distinct token."""

    if not tokenized_documents:
        return {}

    counts: Counter[str] = Counter()

    for tokens in tokenized_documents:
        # Repeated terms count once per document, unlike term frequency.
        counts.update(set(tokens))

    return dict(counts)


def inverse_document_frequency(
    total_documents: int,
    document_frequency: int,
) -> float:
    """Return the positive BM25 rarity weight for one token."""

    if isinstance(total_documents, bool) or not isinstance(total_documents, int):
        raise TypeError("total_documents is not of type int")

    if isinstance(document_frequency, bool) or not isinstance(document_frequency, int):
        raise TypeError("document_frequency is not of type int")

    if total_documents <= 0:
        raise ValueError("total_documents has to be a positive integer")

    if document_frequency < 0 or document_frequency > total_documents:
        raise ValueError(
            "document_frequency has to be between 0 and total_documents"
        )

    # The leading 1 keeps IDF positive even when a token appears in every chunk.
    return math.log(
        1
        + (
            total_documents - document_frequency + 0.5
        ) / (
            document_frequency + 0.5
        )
    )


def average_document_length(
    tokenized_documents: list[list[str]],
) -> float:
    """Return the mean token count across all documents in the corpus."""

    if not tokenized_documents:
        raise ValueError("tokenized_documents cannot be empty")

    # Empty documents contribute zero tokens but still count as corpus members.
    total_tokens = sum(len(tokens) for tokens in tokenized_documents)
    return total_tokens / len(tokenized_documents)


def bm25_term_score(
    term_frequency: int,
    document_frequency: int,
    document_length: int,
    average_document_length: float,
    total_documents: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Score one query term against one document using BM25.

    Args:
        term_frequency: Occurrences of the term in the current document.
        document_frequency: Corpus documents containing the term.
        document_length: Token count of the current document.
        average_document_length: Mean token count across the corpus.
        total_documents: Number of documents in the corpus.
        k1: BM25 configuration controlling term-frequency saturation.
        b: BM25 configuration controlling document-length normalization.
    """

    if isinstance(term_frequency, bool) or not isinstance(term_frequency, int):
        raise TypeError("term_frequency is not of type int")

    if term_frequency < 0:
        raise ValueError("term_frequency has to be a non-negative integer")

    if isinstance(document_length, bool) or not isinstance(document_length, int):
        raise TypeError("document_length is not of type int")

    if document_length < 0:
        raise ValueError("document_length has to be a non-negative integer")

    # Corpus frequency determines how much this term's rarity should matter.
    idf = inverse_document_frequency(
        total_documents,
        document_frequency,
    )

    if isinstance(average_document_length, bool) or not isinstance(
        average_document_length, (int, float)
    ):
        raise TypeError("average_document_length is not of type numeric")

    if not math.isfinite(average_document_length) or average_document_length <= 0:
        raise ValueError("average_document_length has to be finite and positive")

    if isinstance(k1, bool) or not isinstance(k1, (int, float)):
        raise TypeError("k1 is not of type numeric")

    if not math.isfinite(k1) or k1 <= 0:
        raise ValueError("k1 has to be finite and positive")

    if isinstance(b, bool) or not isinstance(b, (int, float)):
        raise TypeError("b is not of type numeric")

    if not math.isfinite(b) or b < 0 or b > 1:
        raise ValueError("b has to be finite and between 0 and 1")

    if term_frequency == 0:
        return 0.0

    # Repetition increases relevance with diminishing returns, while the length
    # ratio prevents long documents from winning only because they contain more.
    term_weight = (
        term_frequency * (k1 + 1)
    ) / (
        term_frequency
        + k1
        * (
            1 - b
            + b * document_length / average_document_length
        )
    )

    return idf * term_weight


def bm25_document_score(
    query_tokens: list[str],
    document_term_frequencies: dict[str, int],
    document_length: int,
    corpus_document_frequencies: dict[str, int],
    average_document_length: float,
    total_documents: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Sum BM25 relevance for unique query terms against one document.

    Args:
        query_tokens: Tokens from the current user question.
        document_term_frequencies: Term counts for the current document.
        document_length: Token count of the current document.
        corpus_document_frequencies: Per-term document counts across the corpus.
        average_document_length: Mean token count across the corpus.
        total_documents: Number of documents in the corpus.
        k1: BM25 term-frequency saturation configuration.
        b: BM25 document-length normalization configuration.
    """

    score = 0.0

    # Query repetition should not multiply a term's contribution in this variant.
    for term in set(query_tokens):
        score += bm25_term_score(
            term_frequency=document_term_frequencies.get(term, 0),
            document_frequency=corpus_document_frequencies.get(term, 0),
            document_length=document_length,
            average_document_length=average_document_length,
            total_documents=total_documents,
            k1=k1,
            b=b,
        )

    return score


def retrieve_keywords(
    data_dir: Path,
    project_name: str,
    question: str,
    top_k: int = 5,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[SearchResult]:
    """Rank one project's stored chunks using BM25 keyword relevance.

    The question supplies query terms, each chunk supplies document terms and
    length, and all stored chunks together supply corpus frequency statistics.
    File paths are indexed with content so exact source names can also match.
    """

    if not question.strip():
        raise ValueError("question cannot be empty")

    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k has to be of type integer")

    if top_k <= 0:
        raise ValueError("top_k has to be a positive integer")

    if isinstance(k1, bool) or not isinstance(k1, (int, float)):
        raise TypeError("k1 is not of type numeric")

    if not math.isfinite(k1) or k1 <= 0:
        raise ValueError("k1 has to be finite and positive")

    if isinstance(b, bool) or not isinstance(b, (int, float)):
        raise TypeError("b is not of type numeric")

    if not math.isfinite(b) or b < 0 or b > 1:
        raise ValueError("b has to be finite and between 0 and 1")

    chunks = load_chunks(data_dir, project_name)

    if not chunks:
        return []

    query_tokens = tokenize(question)
    if not query_tokens:
        return []

    # Each stored chunk is one BM25 document. Including its path lets code and
    # file identifiers contribute to keyword relevance alongside source text.
    tokenized_documents = [
        tokenize(f"{chunk.file_path}\n{chunk.content}")
        for chunk in chunks
    ]

    corpus_frequencies = document_frequencies(tokenized_documents)
    average_length = average_document_length(tokenized_documents)
    if average_length == 0:
        return []
    total_documents = len(chunks)

    results: list[SearchResult] = []

    # Score each chunk with its local statistics and the shared corpus statistics.
    for chunk, tokens in zip(chunks, tokenized_documents):
        score = bm25_document_score(
            query_tokens=query_tokens,
            document_term_frequencies=term_frequencies(tokens),
            document_length=len(tokens),
            corpus_document_frequencies=corpus_frequencies,
            average_document_length=average_length,
            total_documents=total_documents,
            k1=k1,
            b=b,
        )

        # Zero-score chunks contain no query evidence and should not fill top-k.
        if score > 0:
            results.append(SearchResult(chunk=chunk, score=score))

    results.sort(key=lambda result: result.score, reverse=True)
    return results[:top_k]
