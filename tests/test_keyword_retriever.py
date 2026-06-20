import math

import pytest

from contextforge.keyword_retriever import (
    average_document_length,
    bm25_document_score,
    bm25_term_score,
    document_frequencies,
    inverse_document_frequency,
    term_frequencies,
    tokenize,
    retrieve_keywords,
)
from contextforge.models import Chunk, SearchResult
from contextforge.store import save_chunks


def make_keyword_chunk(
    chunk_id: str,
    file_path: str,
    content: str,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        project_name="demo",
        file_path=file_path,
        language="python",
        content=content,
        start_line=1,
        end_line=1,
        embedding=[1.0],
    )


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_tokenize_returns_empty_list_for_empty_text(text):
    assert tokenize(text) == []


def test_tokenize_lowercases_and_ignores_punctuation():
    assert tokenize("Hello, WORLD!") == ["hello", "world"]


def test_tokenize_preserves_and_splits_pascal_case_identifier():
    assert tokenize("RequestFromReader") == [
        "requestfromreader",
        "request",
        "from",
        "reader",
    ]


def test_tokenize_preserves_and_splits_snake_case_identifier():
    assert tokenize("parse_request(request)") == [
        "parse_request",
        "parse",
        "request",
        "request",
    ]


def test_tokenize_splits_file_path_components():
    assert tokenize("internal/response/response.go") == [
        "internal",
        "response",
        "response",
        "go",
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("HTTPResponse", ["httpresponse", "http", "response"]),
        (
            "parseHTTPResponse",
            ["parsehttpresponse", "parse", "http", "response"],
        ),
        (
            "parse_HTTPResponse",
            ["parse_httpresponse", "parse", "http", "response"],
        ),
        ("XML", ["xml"]),
        ("version2Handler", ["version2handler", "version2", "handler"]),
    ],
)
def test_tokenize_handles_code_identifier_variants(text, expected):
    assert tokenize(text) == expected


def test_tokenize_is_deterministic():
    text = "RequestFromReader parses HTTPResponse"

    assert tokenize(text) == tokenize(text)


def test_term_frequencies_counts_repeated_tokens():
    tokens = ["request", "reader", "request", "parse"]

    assert term_frequencies(tokens) == {
        "request": 2,
        "reader": 1,
        "parse": 1,
    }


def test_term_frequencies_returns_empty_dict_for_empty_tokens():
    assert term_frequencies([]) == {}


def test_term_frequencies_preserves_token_strings():
    assert term_frequencies(["Request", "request"]) == {
        "Request": 1,
        "request": 1,
    }


def test_term_frequencies_returns_plain_dictionary():
    assert type(term_frequencies(["request"])) is dict


def test_term_frequencies_does_not_mutate_input():
    tokens = ["request", "request", "reader"]
    original_tokens = tokens.copy()

    term_frequencies(tokens)

    assert tokens == original_tokens


def test_document_frequencies_counts_documents_containing_each_token():
    documents = [
        ["request", "request", "reader"],
        ["request", "server"],
        ["response", "server"],
    ]

    assert document_frequencies(documents) == {
        "request": 2,
        "reader": 1,
        "server": 2,
        "response": 1,
    }


def test_document_frequencies_counts_repeated_token_once_per_document():
    assert document_frequencies([["request", "request", "request"]]) == {
        "request": 1
    }


def test_document_frequencies_returns_empty_dict_for_empty_corpus():
    assert document_frequencies([]) == {}


def test_document_frequencies_ignores_empty_documents():
    assert document_frequencies([[], ["request"], []]) == {"request": 1}


def test_document_frequencies_preserves_token_strings():
    assert document_frequencies([["Request"], ["request"]]) == {
        "Request": 1,
        "request": 1,
    }


def test_document_frequencies_returns_plain_dictionary():
    assert type(document_frequencies([["request"]])) is dict


def test_document_frequencies_does_not_mutate_input():
    documents = [["request", "request"], ["reader"]]
    original_documents = [tokens.copy() for tokens in documents]

    document_frequencies(documents)

    assert documents == original_documents


def test_inverse_document_frequency_matches_bm25_formula():
    expected = math.log(1 + (10 - 2 + 0.5) / (2 + 0.5))

    assert inverse_document_frequency(10, 2) == pytest.approx(expected)


def test_inverse_document_frequency_rewards_rarer_tokens():
    assert inverse_document_frequency(10, 1) > inverse_document_frequency(10, 9)


@pytest.mark.parametrize("document_frequency", [0, 1, 10])
def test_inverse_document_frequency_returns_finite_positive_value(
    document_frequency,
):
    result = inverse_document_frequency(10, document_frequency)

    assert math.isfinite(result)
    assert result > 0


@pytest.mark.parametrize("total_documents", [0, -1])
def test_inverse_document_frequency_rejects_non_positive_document_total(
    total_documents,
):
    with pytest.raises(ValueError, match="has to be a positive integer"):
        inverse_document_frequency(total_documents, 0)


@pytest.mark.parametrize("document_frequency", [-1, 11])
def test_inverse_document_frequency_rejects_out_of_bounds_frequency(
    document_frequency,
):
    with pytest.raises(ValueError, match="has to be between"):
        inverse_document_frequency(10, document_frequency)


@pytest.mark.parametrize("total_documents", [True, False, 10.0, "10"])
def test_inverse_document_frequency_rejects_non_integer_document_total(
    total_documents,
):
    with pytest.raises(TypeError, match="total_documents is not of type int"):
        inverse_document_frequency(total_documents, 1)


@pytest.mark.parametrize("document_frequency", [True, False, 1.0, "1"])
def test_inverse_document_frequency_rejects_non_integer_frequency(
    document_frequency,
):
    with pytest.raises(TypeError, match="document_frequency is not of type int"):
        inverse_document_frequency(10, document_frequency)


def test_average_document_length_calculates_mean_token_count():
    documents = [
        ["request", "reader"],
        ["response", "writer", "chunk"],
        ["server"],
    ]

    assert average_document_length(documents) == pytest.approx(2.0)


def test_average_document_length_counts_empty_documents_in_average():
    assert average_document_length([[], ["request"], []]) == pytest.approx(1 / 3)


def test_average_document_length_returns_zero_for_all_empty_documents():
    assert average_document_length([[], []]) == 0.0


def test_average_document_length_returns_float():
    assert type(average_document_length([["request"], ["reader"]])) is float


def test_average_document_length_rejects_empty_corpus():
    with pytest.raises(ValueError, match="tokenized_documents cannot be empty"):
        average_document_length([])


def test_average_document_length_does_not_mutate_input():
    documents = [["request", "reader"], ["server"]]
    original_documents = [tokens.copy() for tokens in documents]

    average_document_length(documents)

    assert documents == original_documents


def test_bm25_term_score_matches_formula():
    idf = inverse_document_frequency(10, 1)
    expected_weight = (2 * (1.5 + 1)) / (2 + 1.5 * (1 - 0.75 + 0.75))

    assert bm25_term_score(2, 1, 10, 10.0, 10) == pytest.approx(
        idf * expected_weight
    )


def test_bm25_term_score_returns_zero_when_term_is_absent():
    assert bm25_term_score(0, 1, 10, 10.0, 10) == 0.0


def test_bm25_term_score_rewards_rarer_term():
    rare = bm25_term_score(1, 1, 10, 10.0, 10)
    common = bm25_term_score(1, 9, 10, 10.0, 10)

    assert rare > common


def test_bm25_term_score_increases_with_saturating_term_frequency():
    once = bm25_term_score(1, 2, 10, 10.0, 10)
    twice = bm25_term_score(2, 2, 10, 10.0, 10)
    three_times = bm25_term_score(3, 2, 10, 10.0, 10)

    assert once < twice < three_times
    assert twice - once > three_times - twice


def test_bm25_term_score_penalizes_longer_document():
    short = bm25_term_score(1, 2, 5, 10.0, 10)
    long = bm25_term_score(1, 2, 20, 10.0, 10)

    assert short > long


def test_bm25_term_score_disables_length_normalization_when_b_is_zero():
    short = bm25_term_score(1, 2, 5, 10.0, 10, b=0.0)
    long = bm25_term_score(1, 2, 20, 10.0, 10, b=0.0)

    assert short == pytest.approx(long)


def test_bm25_term_score_returns_finite_non_negative_value():
    result = bm25_term_score(2, 1, 10, 10.0, 10)

    assert math.isfinite(result)
    assert result >= 0


@pytest.mark.parametrize("term_frequency", [True, 1.5, "1"])
def test_bm25_term_score_rejects_non_integer_term_frequency(term_frequency):
    with pytest.raises(TypeError, match="term_frequency is not of type int"):
        bm25_term_score(term_frequency, 1, 10, 10.0, 10)


def test_bm25_term_score_rejects_negative_term_frequency():
    with pytest.raises(ValueError, match="term_frequency has to be a non-negative"):
        bm25_term_score(-1, 1, 10, 10.0, 10)


@pytest.mark.parametrize("document_length", [True, 10.5, "10"])
def test_bm25_term_score_rejects_non_integer_document_length(document_length):
    with pytest.raises(TypeError, match="document_length is not of type int"):
        bm25_term_score(1, 1, document_length, 10.0, 10)


def test_bm25_term_score_rejects_negative_document_length():
    with pytest.raises(ValueError, match="document_length has to be a non-negative"):
        bm25_term_score(1, 1, -1, 10.0, 10)


@pytest.mark.parametrize("average_length", [True, "10"])
def test_bm25_term_score_rejects_non_numeric_average_length(average_length):
    with pytest.raises(TypeError, match="average_document_length is not of type"):
        bm25_term_score(1, 1, 10, average_length, 10)


@pytest.mark.parametrize(
    "average_length",
    [0, -1, float("nan"), float("inf")],
)
def test_bm25_term_score_rejects_invalid_average_length(average_length):
    with pytest.raises(ValueError, match="average_document_length has to be"):
        bm25_term_score(1, 1, 10, average_length, 10)


@pytest.mark.parametrize("k1", [True, "1.5"])
def test_bm25_term_score_rejects_non_numeric_k1(k1):
    with pytest.raises(TypeError, match="k1 is not of type numeric"):
        bm25_term_score(1, 1, 10, 10.0, 10, k1=k1)


@pytest.mark.parametrize("k1", [0, -1, float("nan"), float("inf")])
def test_bm25_term_score_rejects_invalid_k1(k1):
    with pytest.raises(ValueError, match="k1 has to be finite and positive"):
        bm25_term_score(1, 1, 10, 10.0, 10, k1=k1)


@pytest.mark.parametrize("b", [True, "0.75"])
def test_bm25_term_score_rejects_non_numeric_b(b):
    with pytest.raises(TypeError, match="b is not of type numeric"):
        bm25_term_score(1, 1, 10, 10.0, 10, b=b)


@pytest.mark.parametrize("b", [-0.1, 1.1, float("nan"), float("inf")])
def test_bm25_term_score_rejects_invalid_b(b):
    with pytest.raises(ValueError, match="b has to be finite and between 0 and 1"):
        bm25_term_score(1, 1, 10, 10.0, 10, b=b)


def test_bm25_term_score_validates_corpus_when_term_is_absent():
    with pytest.raises(ValueError, match="total_documents has to be a positive"):
        bm25_term_score(0, 0, 0, 1.0, 0)


def test_bm25_document_score_returns_zero_for_empty_query():
    assert bm25_document_score([], {"request": 1}, 1, {"request": 1}, 1.0, 1) == 0.0


def test_bm25_document_score_returns_zero_when_no_query_terms_match():
    score = bm25_document_score(
        ["response"],
        {"request": 2},
        2,
        {"request": 1},
        2.0,
        1,
    )

    assert score == 0.0


def test_bm25_document_score_sums_matching_term_scores():
    query_tokens = ["request", "reader"]
    term_counts = {"request": 2, "reader": 1}
    corpus_counts = {"request": 2, "reader": 1}

    result = bm25_document_score(
        query_tokens,
        term_counts,
        3,
        corpus_counts,
        3.0,
        3,
    )
    expected = bm25_term_score(2, 2, 3, 3.0, 3) + bm25_term_score(
        1, 1, 3, 3.0, 3
    )

    assert result == pytest.approx(expected)


def test_bm25_document_score_counts_repeated_query_term_once():
    single = bm25_document_score(
        ["request"], {"request": 2}, 2, {"request": 1}, 2.0, 2
    )
    repeated = bm25_document_score(
        ["request", "request"],
        {"request": 2},
        2,
        {"request": 1},
        2.0,
        2,
    )

    assert repeated == pytest.approx(single)


def test_bm25_document_score_reuses_term_score_validation():
    with pytest.raises(ValueError, match="average_document_length has to be"):
        bm25_document_score(
            ["request"],
            {"request": 1},
            1,
            {"request": 1},
            0.0,
            1,
        )


def test_bm25_document_score_does_not_mutate_inputs():
    query_tokens = ["request", "reader"]
    term_counts = {"request": 1, "reader": 1}
    corpus_counts = {"request": 2, "reader": 1}
    original_query = query_tokens.copy()
    original_term_counts = term_counts.copy()
    original_corpus_counts = corpus_counts.copy()

    bm25_document_score(
        query_tokens,
        term_counts,
        2,
        corpus_counts,
        2.0,
        2,
    )

    assert query_tokens == original_query
    assert term_counts == original_term_counts
    assert corpus_counts == original_corpus_counts


def test_retrieve_keywords_ranks_exact_identifier_match_first(tmp_path):
    chunks = [
        make_keyword_chunk(
            "request",
            "internal/request/request.go",
            "func RequestFromReader reads a request",
        ),
        make_keyword_chunk(
            "response",
            "internal/response/response.go",
            "func WriteResponse writes a response",
        ),
    ]
    save_chunks(tmp_path, "demo", chunks)

    results = retrieve_keywords(tmp_path, "demo", "RequestFromReader")

    assert results
    assert isinstance(results[0], SearchResult)
    assert results[0].chunk.chunk_id == "request"


def test_retrieve_keywords_includes_file_path_terms(tmp_path):
    chunks = [
        make_keyword_chunk("request", "internal/request/request.go", "parse"),
        make_keyword_chunk("response", "internal/response/response.go", "write"),
    ]
    save_chunks(tmp_path, "demo", chunks)

    results = retrieve_keywords(tmp_path, "demo", "response.go")

    assert results[0].chunk.chunk_id == "response"


def test_retrieve_keywords_excludes_zero_score_chunks(tmp_path):
    chunks = [
        make_keyword_chunk("request", "request.go", "parse request"),
        make_keyword_chunk("readme", "README.md", "general architecture"),
    ]
    save_chunks(tmp_path, "demo", chunks)

    results = retrieve_keywords(tmp_path, "demo", "request")

    assert [result.chunk.chunk_id for result in results] == ["request"]


def test_retrieve_keywords_respects_top_k(tmp_path):
    chunks = [
        make_keyword_chunk("first", "first.go", "request request"),
        make_keyword_chunk("second", "second.go", "request"),
    ]
    save_chunks(tmp_path, "demo", chunks)

    assert len(retrieve_keywords(tmp_path, "demo", "request", top_k=1)) == 1


def test_retrieve_keywords_returns_empty_list_for_empty_project(tmp_path):
    save_chunks(tmp_path, "demo", [])

    assert retrieve_keywords(tmp_path, "demo", "request") == []


def test_retrieve_keywords_returns_empty_list_for_tokenless_query(tmp_path):
    save_chunks(
        tmp_path,
        "demo",
        [make_keyword_chunk("request", "request.go", "request")],
    )

    assert retrieve_keywords(tmp_path, "demo", "!!!") == []


def test_retrieve_keywords_returns_empty_list_for_tokenless_corpus(tmp_path):
    save_chunks(
        tmp_path,
        "demo",
        [make_keyword_chunk("empty", "---", "!!!")],
    )

    assert retrieve_keywords(tmp_path, "demo", "request") == []


def test_retrieve_keywords_rejects_empty_question(tmp_path):
    with pytest.raises(ValueError, match="question cannot be empty"):
        retrieve_keywords(tmp_path, "missing", "   ")


@pytest.mark.parametrize("top_k", [True, 1.5, "5"])
def test_retrieve_keywords_rejects_non_integer_top_k(tmp_path, top_k):
    with pytest.raises(TypeError, match="top_k has to be of type integer"):
        retrieve_keywords(tmp_path, "missing", "request", top_k=top_k)


@pytest.mark.parametrize("top_k", [0, -1])
def test_retrieve_keywords_rejects_non_positive_top_k(tmp_path, top_k):
    with pytest.raises(ValueError, match="top_k has to be a positive integer"):
        retrieve_keywords(tmp_path, "missing", "request", top_k=top_k)


@pytest.mark.parametrize(
    ("name", "value", "error_type", "message"),
    [
        ("k1", True, TypeError, "k1 is not of type numeric"),
        ("k1", 0, ValueError, "k1 has to be finite and positive"),
        ("b", True, TypeError, "b is not of type numeric"),
        ("b", 1.1, ValueError, "b has to be finite and between"),
    ],
)
def test_retrieve_keywords_validates_configuration_before_loading_index(
    tmp_path, name, value, error_type, message
):
    with pytest.raises(error_type, match=message):
        retrieve_keywords(
            tmp_path,
            "missing",
            "request",
            **{name: value},
        )


def test_retrieve_keywords_propagates_missing_project_index(tmp_path):
    with pytest.raises(FileNotFoundError, match="Requested file not found"):
        retrieve_keywords(tmp_path, "missing", "request")
