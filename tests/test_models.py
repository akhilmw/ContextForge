import pytest
from contextforge.models import Answer, Chunk, Document, SearchResult, Source


def test_document_stores_source_information():
    document = Document(
        file_path="src/parser.py",
        language="python",
        content="def parse():\n    pass\n",
    )

    assert document.file_path == "src/parser.py"
    assert document.language == "python"
    assert document.content == "def parse():\n    pass\n"


def test_chunk_defaults_to_no_embedding():
    chunk = Chunk(
        chunk_id="chunk-1",
        project_name="demo",
        file_path="src/parser.py",
        language="python",
        content="def parse():",
        start_line=1,
        end_line=1,
    )

    assert chunk.embedding is None


def test_chunk_accepts_embedding():
    chunk = Chunk(
        chunk_id="chunk-1",
        project_name="demo",
        file_path="src/parser.py",
        language="python",
        content="def parse():",
        start_line=1,
        end_line=1,
        embedding=[0.1, 0.2, 0.3],
    )

    assert chunk.embedding == [0.1, 0.2, 0.3]


@pytest.mark.parametrize(
    ("start_line", "end_line"),
    [
        (0, 1),
        (1, 0),
        (-1, 1),
        (1, -1),
    ],
)
def test_chunk_rejects_non_positive_line_numbers(start_line, end_line):
    with pytest.raises(
        ValueError,
        match="start line or end line cannot be zero or negative",
    ):
        Chunk(
            chunk_id="chunk-1",
            project_name="demo",
            file_path="src/parser.py",
            language="python",
            content="def parse():",
            start_line=start_line,
            end_line=end_line,
        )


def test_chunk_rejects_start_line_after_end_line():
    with pytest.raises(
        ValueError,
        match="start line cannot be greater than end line",
    ):
        Chunk(
            chunk_id="chunk-1",
            project_name="demo",
            file_path="src/parser.py",
            language="python",
            content="def parse():",
            start_line=10,
            end_line=5,
        )


def test_chunk_allows_single_line_range():
    chunk = Chunk(
        chunk_id="chunk-1",
        project_name="demo",
        file_path="src/parser.py",
        language="python",
        content="def parse():",
        start_line=4,
        end_line=4,
    )

    assert chunk.start_line == chunk.end_line == 4


def test_search_result_retains_chunk_and_score():
    chunk = Chunk(
        chunk_id="chunk-1",
        project_name="demo",
        file_path="src/parser.py",
        language="python",
        content="def parse():",
        start_line=1,
        end_line=1,
    )

    result = SearchResult(chunk=chunk, score=0.91)

    assert result.chunk is chunk
    assert result.score == 0.91


def test_source_stores_citation_information():
    source = Source(
        label="S1",
        file_path="src/parser.py",
        start_line=4,
        end_line=8,
        score=0.87,
    )

    assert source.label == "S1"
    assert source.file_path == "src/parser.py"
    assert source.start_line == 4
    assert source.end_line == 8
    assert source.score == 0.87


def test_answer_stores_text_and_sources():
    source = Source(
        label="S1",
        file_path="src/parser.py",
        start_line=4,
        end_line=8,
        score=0.87,
    )

    answer = Answer(
        text="The parser handles input here. [S1]",
        sources=[source],
    )

    assert answer.text == "The parser handles input here. [S1]"
    assert answer.sources == [source]


def test_answer_allows_no_sources():
    answer = Answer(
        text="Insufficient evidence.",
        sources=[],
    )

    assert answer.sources == []
