import pytest

from contextforge.models import Chunk, SearchResult, Source
from contextforge.prompt_builder import build_prompt, results_to_sources


def make_result(
    chunk_id: str,
    file_path: str,
    content: str,
    score: float,
    start_line: int = 1,
    end_line: int = 3,
) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            project_name="demo",
            file_path=file_path,
            language="python",
            content=content,
            start_line=start_line,
            end_line=end_line,
            embedding=[0.1, 0.2, 0.3],
        ),
        score=score,
    )


def test_results_to_sources_preserves_order_and_scores():
    results = [
        make_result("chunk-1", "src/parser.py", "def parse():\n    pass\n", 0.91),
        make_result("chunk-2", "README.md", "# Demo\n", 0.72, 4, 4),
    ]

    sources = results_to_sources(results)

    assert sources == [
        Source("S1", "src/parser.py", 1, 3, 0.91),
        Source("S2", "README.md", 4, 4, 0.72),
    ]


def test_build_prompt_rejects_empty_question():
    with pytest.raises(ValueError, match="question cannot be empty"):
        build_prompt("   ", [])


def test_build_prompt_handles_empty_results():
    prompt = build_prompt("What parses requests?", [])

    assert "Question:\nWhat parses requests?" in prompt
    assert "Sources:\nNo sources were retrieved." in prompt
    assert "I do not have enough evidence to answer." in prompt


def test_build_prompt_includes_source_labels_paths_lines_and_content():
    result = make_result(
        "chunk-1",
        "src/parser.py",
        "def parse_request(request):\n    return request.strip()\n",
        0.91,
        10,
        11,
    )

    prompt = build_prompt("How are requests parsed?", [result])

    assert "[S1] src/parser.py:10-11" in prompt
    assert "def parse_request(request):" in prompt
    assert "return request.strip()" in prompt


def test_build_prompt_preserves_result_order():
    first = make_result("chunk-1", "src/first.py", "first content\n", 0.91)
    second = make_result("chunk-2", "src/second.py", "second content\n", 0.72)

    prompt = build_prompt("What is relevant?", [first, second])

    first_position = prompt.index("[S1] src/first.py")
    second_position = prompt.index("[S2] src/second.py")

    assert first_position < second_position


def test_build_prompt_excludes_embeddings():
    result = make_result(
        "chunk-1",
        "src/parser.py",
        "def parse_request(request):\n    return request.strip()\n",
        0.91,
    )

    prompt = build_prompt("How are requests parsed?", [result])

    assert "[0.1, 0.2, 0.3]" not in prompt
    assert "embedding" not in prompt.lower()


def test_build_prompt_includes_grounding_instructions():
    prompt = build_prompt("What is relevant?", [])

    assert "Answer the question using only the provided sources." in prompt
    assert "- Cite sources using [S1], [S2], etc." in prompt
    assert "- Do not cite files that are not listed." in prompt
    assert "- Do not guess beyond the sources." in prompt
