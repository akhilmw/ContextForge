from pathlib import Path

import pytest

from contextforge.langchain_rag import ask_with_langchain, retrieve_with_langchain
from contextforge.models import Answer, Chunk, SearchResult
from contextforge.store import save_chunks


class QueryEmbedder:
    def __init__(self, vector):
        self.vector = vector
        self.query_calls = []

    def embed_documents(self, texts):
        raise NotImplementedError

    def embed_query(self, text):
        self.query_calls.append(text)
        return self.vector


class RecordingLLM:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.response


def make_chunk(
    chunk_id,
    embedding,
    content=None,
    file_path=None,
    start_line=1,
    end_line=1,
):
    return Chunk(
        chunk_id=chunk_id,
        project_name="demo",
        file_path=file_path or f"src/{chunk_id}.py",
        language="python",
        content=content or f"content for {chunk_id}",
        start_line=start_line,
        end_line=end_line,
        embedding=embedding,
    )


def test_retrieve_with_langchain_returns_empty_list_for_empty_project(tmp_path):
    embedder = QueryEmbedder([1.0, 0.0])
    save_chunks(tmp_path, "demo", [])

    results = retrieve_with_langchain(
        data_dir=tmp_path,
        project_name="demo",
        question="What parses requests?",
        embedder=embedder,
    )

    assert results == []
    assert embedder.query_calls == []


def test_retrieve_with_langchain_ranks_results_by_score_descending(tmp_path):
    chunks = [
        make_chunk("best", [1.0, 0.0]),
        make_chunk("middle", [0.5, 0.5]),
        make_chunk("worst", [-1.0, 0.0]),
    ]
    save_chunks(tmp_path, "demo", chunks)

    results = retrieve_with_langchain(
        data_dir=tmp_path,
        project_name="demo",
        question="What parses requests?",
        embedder=QueryEmbedder([1.0, 0.0]),
        top_k=3,
    )

    assert [result.chunk.chunk_id for result in results] == [
        "best",
        "middle",
        "worst",
    ]
    assert [result.score for result in results] == sorted(
        [result.score for result in results],
        reverse=True,
    )


def test_retrieve_with_langchain_respects_top_k(tmp_path):
    chunks = [
        make_chunk("best", [1.0, 0.0]),
        make_chunk("middle", [0.5, 0.5]),
        make_chunk("worst", [-1.0, 0.0]),
    ]
    save_chunks(tmp_path, "demo", chunks)

    results = retrieve_with_langchain(
        data_dir=tmp_path,
        project_name="demo",
        question="What parses requests?",
        embedder=QueryEmbedder([1.0, 0.0]),
        top_k=2,
    )

    assert [result.chunk.chunk_id for result in results] == ["best", "middle"]


def test_retrieve_with_langchain_returns_search_result_objects(tmp_path):
    chunk = make_chunk(
        "best",
        [1.0, 0.0],
        content="def parse_request(request): return request.strip()",
        file_path="src/parser.py",
        start_line=10,
        end_line=12,
    )
    save_chunks(tmp_path, "demo", [chunk])

    results = retrieve_with_langchain(
        data_dir=tmp_path,
        project_name="demo",
        question="What parses requests?",
        embedder=QueryEmbedder([1.0, 0.0]),
    )

    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].chunk == chunk
    assert results[0].score == pytest.approx(1.0)


def test_retrieve_with_langchain_rejects_empty_question(tmp_path):
    save_chunks(tmp_path, "demo", [])

    with pytest.raises(ValueError, match="question cannot be empty"):
        retrieve_with_langchain(
            data_dir=tmp_path,
            project_name="demo",
            question="   ",
            embedder=QueryEmbedder([1.0, 0.0]),
        )


@pytest.mark.parametrize("top_k", [True, 1.5, "5"])
def test_retrieve_with_langchain_rejects_non_integer_top_k(tmp_path, top_k):
    with pytest.raises(TypeError, match="is not of type int"):
        retrieve_with_langchain(
            tmp_path,
            "demo",
            "question",
            QueryEmbedder([1.0, 0.0]),
            top_k=top_k,
        )


@pytest.mark.parametrize("top_k", [0, -1])
def test_retrieve_with_langchain_rejects_non_positive_top_k(tmp_path, top_k):
    with pytest.raises(ValueError, match="top k cannot be less than or equal to zero"):
        retrieve_with_langchain(
            tmp_path,
            "demo",
            "question",
            QueryEmbedder([1.0, 0.0]),
            top_k=top_k,
        )


def test_retrieve_with_langchain_propagates_missing_project_index(tmp_path):
    with pytest.raises(FileNotFoundError, match="Requested file not found"):
        retrieve_with_langchain(
            data_dir=tmp_path,
            project_name="missing",
            question="What parses requests?",
            embedder=QueryEmbedder([1.0, 0.0]),
        )


def test_retrieve_with_langchain_rejects_query_dimension_mismatch(tmp_path):
    save_chunks(tmp_path, "demo", [make_chunk("best", [1.0, 0.0])])

    with pytest.raises(ValueError, match="same dimension"):
        retrieve_with_langchain(
            data_dir=tmp_path,
            project_name="demo",
            question="What parses requests?",
            embedder=QueryEmbedder([1.0, 0.0, 0.0]),
        )


def test_retrieve_with_langchain_accepts_pathlike_data_dir(tmp_path):
    chunk = make_chunk("best", [1.0, 0.0])
    save_chunks(tmp_path, "demo", [chunk])

    results = retrieve_with_langchain(
        data_dir=Path(tmp_path),
        project_name="demo",
        question="What parses requests?",
        embedder=QueryEmbedder([1.0, 0.0]),
    )

    assert [result.chunk.chunk_id for result in results] == ["best"]


def test_ask_with_langchain_returns_answer_with_sources(tmp_path):
    save_chunks(
        tmp_path,
        "demo",
        [
            make_chunk(
                "parser",
                [1.0, 0.0],
                "def parse_request(request):\n    return request.strip()\n",
                "src/parser.py",
                start_line=1,
                end_line=2,
            ),
        ],
    )
    llm = RecordingLLM("Requests are parsed by stripping whitespace. [S1]")

    answer = ask_with_langchain(
        data_dir=tmp_path,
        project_name="demo",
        question="How are requests parsed?",
        embedder=QueryEmbedder([1.0, 0.0]),
        llm=llm,
    )

    assert isinstance(answer, Answer)
    assert answer.text == "Requests are parsed by stripping whitespace. [S1]"
    assert len(answer.sources) == 1
    assert answer.sources[0].label == "S1"
    assert answer.sources[0].file_path == "src/parser.py"
    assert answer.sources[0].start_line == 1
    assert answer.sources[0].end_line == 2
    assert answer.sources[0].score == pytest.approx(1.0)


def test_ask_with_langchain_passes_grounded_prompt_to_llm(tmp_path):
    save_chunks(
        tmp_path,
        "demo",
        [
            make_chunk(
                "parser",
                [1.0, 0.0],
                "def parse_request(request):\n    return request.strip()\n",
                "src/parser.py",
                start_line=1,
                end_line=2,
            ),
        ],
    )
    llm = RecordingLLM("Answer")

    ask_with_langchain(
        data_dir=tmp_path,
        project_name="demo",
        question="How are requests parsed?",
        embedder=QueryEmbedder([1.0, 0.0]),
        llm=llm,
    )

    assert len(llm.prompts) == 1
    prompt = llm.prompts[0]
    assert "Question:\nHow are requests parsed?" in prompt
    assert "[S1] src/parser.py:1-2" in prompt
    assert "return request.strip()" in prompt
    assert "Answer the question using only the provided sources." in prompt


def test_ask_with_langchain_respects_top_k(tmp_path):
    save_chunks(
        tmp_path,
        "demo",
        [
            make_chunk("best", [1.0, 0.0], "best content\n"),
            make_chunk("second", [0.5, 0.5], "second content\n"),
        ],
    )
    llm = RecordingLLM("Answer using only the top result. [S1]")

    answer = ask_with_langchain(
        data_dir=tmp_path,
        project_name="demo",
        question="What is the best match?",
        embedder=QueryEmbedder([1.0, 0.0]),
        llm=llm,
        top_k=1,
    )

    assert [source.file_path for source in answer.sources] == ["src/best.py"]
    assert "best content" in llm.prompts[0]
    assert "second content" not in llm.prompts[0]


def test_ask_with_langchain_handles_empty_retrieval_results(tmp_path):
    save_chunks(tmp_path, "demo", [])
    llm = RecordingLLM("I do not have enough evidence to answer.")

    answer = ask_with_langchain(
        data_dir=tmp_path,
        project_name="demo",
        question="What does the project do?",
        embedder=QueryEmbedder([1.0, 0.0]),
        llm=llm,
    )

    assert answer.text == "I do not have enough evidence to answer."
    assert answer.sources == []
    assert "No sources were retrieved." in llm.prompts[0]


def test_ask_with_langchain_propagates_empty_question_error(tmp_path):
    save_chunks(tmp_path, "demo", [])
    llm = RecordingLLM("Answer")

    with pytest.raises(ValueError, match="question cannot be empty"):
        ask_with_langchain(
            data_dir=tmp_path,
            project_name="demo",
            question="   ",
            embedder=QueryEmbedder([1.0, 0.0]),
            llm=llm,
        )

    assert llm.prompts == []
