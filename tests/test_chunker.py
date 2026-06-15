import pytest

from contextforge.chunker import chunk_document, chunk_documents
from contextforge.models import Document


def make_document(content: str, file_path: str = "src/example.py") -> Document:
    return Document(
        file_path=file_path,
        language="python",
        content=content,
    )


def test_short_document_creates_one_chunk():
    document = make_document("line 1\nline 2\n")

    chunks = chunk_document(document, "demo", chunk_size=3, chunk_overlap=1)

    assert len(chunks) == 1
    assert chunks[0].content == "line 1\nline 2\n"
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 2


def test_exact_size_document_creates_one_chunk():
    document = make_document("line 1\nline 2\nline 3\n")

    chunks = chunk_document(document, "demo", chunk_size=3, chunk_overlap=1)

    assert len(chunks) == 1
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 3


def test_long_document_creates_multiple_chunks_with_overlap():
    document = make_document("line 1\nline 2\nline 3\nline 4\nline 5\n")

    chunks = chunk_document(document, "demo", chunk_size=3, chunk_overlap=1)

    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [
        (1, 3),
        (3, 5),
    ]
    assert chunks[0].content == "line 1\nline 2\nline 3\n"
    assert chunks[1].content == "line 3\nline 4\nline 5\n"


def test_final_partial_chunk_is_retained():
    document = make_document(
        "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\n",
    )

    chunks = chunk_document(document, "demo", chunk_size=4, chunk_overlap=1)

    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [
        (1, 4),
        (4, 6),
    ]
    assert chunks[-1].content == "line 4\nline 5\nline 6\n"


def test_empty_document_returns_no_chunks():
    document = make_document("")

    assert chunk_document(document, "demo") == []


def test_chunk_preserves_document_and_project_metadata():
    document = Document(
        file_path="internal/request.go",
        language="go",
        content="package request\n",
    )

    chunk = chunk_document(document, "http-go")[0]

    assert chunk.project_name == "http-go"
    assert chunk.file_path == "internal/request.go"
    assert chunk.language == "go"
    assert chunk.embedding is None


def test_chunk_ids_are_stable_for_unchanged_input():
    document = make_document("line 1\nline 2\nline 3\n")

    first = chunk_document(document, "demo", chunk_size=2, chunk_overlap=1)
    second = chunk_document(document, "demo", chunk_size=2, chunk_overlap=1)

    assert [chunk.chunk_id for chunk in first] == [
        chunk.chunk_id for chunk in second
    ]


def test_content_change_produces_different_chunk_id():
    original = make_document("line 1\nline 2\n")
    changed = make_document("line 1\nchanged line\n")

    original_chunk = chunk_document(original, "demo")[0]
    changed_chunk = chunk_document(changed, "demo")[0]

    assert original_chunk.chunk_id != changed_chunk.chunk_id


def test_project_name_changes_chunk_id():
    document = make_document("line 1\n")

    first = chunk_document(document, "project-one")[0]
    second = chunk_document(document, "project-two")[0]

    assert first.chunk_id != second.chunk_id


def test_chunk_documents_preserves_document_and_chunk_order():
    first_document = make_document(
        "a1\na2\na3\n",
        file_path="a.py",
    )
    second_document = make_document(
        "b1\nb2\n",
        file_path="b.py",
    )

    chunks = chunk_documents(
        [first_document, second_document],
        "demo",
        chunk_size=2,
        chunk_overlap=1,
    )

    assert [
        (chunk.file_path, chunk.start_line, chunk.end_line)
        for chunk in chunks
    ] == [
        ("a.py", 1, 2),
        ("a.py", 2, 3),
        ("b.py", 1, 2),
    ]


def test_chunk_documents_returns_empty_list_for_empty_input():
    assert chunk_documents([], "demo") == []


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (0, 0),
        (-1, 0),
        (3, -1),
        (3, 3),
        (3, 4),
    ],
)
def test_rejects_invalid_chunk_configuration(chunk_size, chunk_overlap):
    document = make_document("line 1\n")

    with pytest.raises(
        ValueError,
        match="invalid configs for chunk_size/chunk_overlap",
    ):
        chunk_document(
            document,
            "demo",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


@pytest.mark.parametrize("project_name", ["", "   "])
def test_rejects_empty_project_name(project_name):
    document = make_document("line 1\n")

    with pytest.raises(
        ValueError,
        match="project name cannot be empty",
    ):
        chunk_document(document, project_name)
