from pathlib import Path

import pytest

from contextforge.discovery import discover_files


SAMPLE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


def test_discovers_supported_files_recursively():
    discovered = discover_files(SAMPLE_REPO)

    assert discovered == [
        Path("README.md"),
        Path("config/service.yaml"),
        Path("db/schema.sql"),
        Path("docs/architecture.md"),
        Path("notes/debug.txt"),
        Path("src/parser.py"),
        Path("src/server.go"),
    ]


def test_returns_repository_relative_paths():
    discovered = discover_files(SAMPLE_REPO)

    assert discovered
    assert all(not path.is_absolute() for path in discovered)
    assert all(SAMPLE_REPO not in path.parents for path in discovered)


def test_skips_unsupported_extensions():
    discovered = discover_files(SAMPLE_REPO)

    assert Path("assets/logo.png") not in discovered


@pytest.mark.parametrize(
    "ignored_file",
    [
        Path(".git/internal.txt"),
        Path("build/generated.py"),
        Path("node_modules/dependency.py"),
        Path("src/__pycache__/cached.py"),
    ],
)
def test_skips_files_in_ignored_directories(tmp_path, ignored_file):
    file_path = tmp_path / ignored_file
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("This file must be ignored.")

    discovered = discover_files(tmp_path)

    assert ignored_file not in discovered


def test_returns_paths_in_sorted_order():
    discovered = discover_files(SAMPLE_REPO)

    assert discovered == sorted(discovered)


def test_raises_for_missing_repository(tmp_path):
    missing_repo = tmp_path / "missing"

    with pytest.raises(
        FileNotFoundError,
        match="Repository does not exist",
    ):
        discover_files(missing_repo)


def test_raises_when_repository_path_is_a_file(tmp_path):
    file_path = tmp_path / "not-a-repository.txt"
    file_path.write_text("This is a file, not a repository.")

    with pytest.raises(
        NotADirectoryError,
        match="Repository path is not a directory",
    ):
        discover_files(file_path)


def test_empty_repository_returns_empty_list(tmp_path):
    assert discover_files(tmp_path) == []
