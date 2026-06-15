from pathlib import Path

import pytest

from contextforge.discovery import discover_files
from contextforge.loaders import load_document, load_documents
from contextforge.models import Document


SAMPLE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


def test_loads_python_file_as_document():
    document = load_document(SAMPLE_REPO, Path("src/parser.py"))

    assert isinstance(document, Document)
    assert document.file_path == "src/parser.py"
    assert document.language == "python"
    assert "def parse_request" in document.content


def test_normalizes_repository_relative_path():
    document = load_document(SAMPLE_REPO, Path("src/../README.md"))

    assert document.file_path == "README.md"


@pytest.mark.parametrize(
    ("file_name", "expected_language"),
    [
        ("example.py", "python"),
        ("example.go", "go"),
        ("example.md", "markdown"),
        ("example.yaml", "yaml"),
        ("example.yml", "yaml"),
        ("example.json", "json"),
        ("example.sql", "sql"),
        ("example.txt", "text"),
    ],
)
def test_maps_supported_extensions_to_languages(
    tmp_path,
    file_name,
    expected_language,
):
    file_path = tmp_path / file_name
    file_path.write_text("sample content", encoding="utf-8")

    document = load_document(tmp_path, Path(file_name))

    assert document.language == expected_language


def test_loads_multiple_documents_in_input_order():
    file_paths = [
        Path("src/server.go"),
        Path("README.md"),
        Path("src/parser.py"),
    ]

    documents = load_documents(SAMPLE_REPO, file_paths)

    assert [document.file_path for document in documents] == [
        "src/server.go",
        "README.md",
        "src/parser.py",
    ]


def test_load_documents_returns_empty_list_for_empty_input():
    assert load_documents(SAMPLE_REPO, []) == []


def test_discovery_output_can_be_loaded():
    file_paths = discover_files(SAMPLE_REPO)

    documents = load_documents(SAMPLE_REPO, file_paths)

    assert [document.file_path for document in documents] == [
        str(path) for path in file_paths
    ]


def test_raises_for_missing_file():
    with pytest.raises(
        FileNotFoundError,
        match="File does not exists",
    ):
        load_document(SAMPLE_REPO, Path("missing.py"))


def test_raises_when_file_path_is_a_directory():
    with pytest.raises(
        IsADirectoryError,
        match="Path is not a file path",
    ):
        load_document(SAMPLE_REPO, Path("docs"))


def test_raises_for_unsupported_extension():
    with pytest.raises(
        ValueError,
        match="Unsupported extension",
    ):
        load_document(SAMPLE_REPO, Path("assets/logo.png"))


def test_rejects_path_outside_repository(tmp_path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("outside content", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Path is outside repository",
    ):
        load_document(repo_path, Path("../outside.txt"))


def test_raises_for_invalid_utf8(tmp_path):
    invalid_file = tmp_path / "invalid.txt"
    invalid_file.write_bytes(b"\xff\xfe\xfa")

    with pytest.raises(UnicodeDecodeError):
        load_document(tmp_path, Path("invalid.txt"))
