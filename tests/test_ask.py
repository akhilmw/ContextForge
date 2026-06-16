import pytest

from contextforge.ask import ask_question
from contextforge.models import Answer, Chunk
from contextforge.store import save_chunks


class QueryEmbedder:
    def __init__(self, vector):
        self.vector = vector

    def embed_documents(self, texts):
        raise NotImplementedError

    def embed_query(self, text):
        return self.vector


class RecordingLLM:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.response


def make_chunk(chunk_id, embedding, content=None, file_path=None):
    return Chunk(
        chunk_id=chunk_id,
        project_name="demo",
        file_path=file_path or f"src/{chunk_id}.py",
        language="python",
        content=content or f"content for {chunk_id}",
        start_line=1,
        end_line=2,
        embedding=embedding,
    )


def test_ask_question_returns_answer_with_sources(tmp_path):
    save_chunks(
        tmp_path,
        "demo",
        [
            make_chunk(
                "parser",
                [1.0, 0.0],
                "def parse_request(request):\n    return request.strip()\n",
                "src/parser.py",
            ),
        ],
    )
    llm = RecordingLLM("Requests are parsed by stripping whitespace. [S1]")

    answer = ask_question(
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


def test_ask_question_passes_grounded_prompt_to_llm(tmp_path):
    save_chunks(
        tmp_path,
        "demo",
        [
            make_chunk(
                "parser",
                [1.0, 0.0],
                "def parse_request(request):\n    return request.strip()\n",
                "src/parser.py",
            ),
        ],
    )
    llm = RecordingLLM("Answer")

    ask_question(
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


def test_ask_question_respects_top_k(tmp_path):
    save_chunks(
        tmp_path,
        "demo",
        [
            make_chunk("best", [1.0, 0.0], "best content\n"),
            make_chunk("second", [0.5, 0.5], "second content\n"),
        ],
    )
    llm = RecordingLLM("Answer using only the top result. [S1]")

    answer = ask_question(
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


def test_ask_question_handles_empty_retrieval_results(tmp_path):
    save_chunks(tmp_path, "demo", [])
    llm = RecordingLLM("I do not have enough evidence to answer.")

    answer = ask_question(
        data_dir=tmp_path,
        project_name="demo",
        question="What does the project do?",
        embedder=QueryEmbedder([1.0, 0.0]),
        llm=llm,
    )

    assert answer.text == "I do not have enough evidence to answer."
    assert answer.sources == []
    assert "No sources were retrieved." in llm.prompts[0]


def test_ask_question_propagates_empty_question_error(tmp_path):
    save_chunks(tmp_path, "demo", [])
    llm = RecordingLLM("Answer")

    with pytest.raises(ValueError, match="question cannot be empty"):
        ask_question(
            data_dir=tmp_path,
            project_name="demo",
            question="   ",
            embedder=QueryEmbedder([1.0, 0.0]),
            llm=llm,
        )

    assert llm.prompts == []
