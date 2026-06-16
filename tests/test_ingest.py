from pathlib import Path

import pytest

from contextforge.embedder import FakeEmbedder
from contextforge.ingest import ingest_repository
from contextforge.models import IngestionResult
from contextforge.store import load_chunks


SAMPLE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


def test_ingest_repository_runs_full_pipeline_and_returns_summary(tmp_path):
    result = ingest_repository(
        repo_path=SAMPLE_REPO,
        project_name="demo",
        data_dir=tmp_path,
        embedder=FakeEmbedder(),
    )

    assert result == IngestionResult(
        project_name="demo",
        files_discovered=7,
        documents_loaded=7,
        chunks_created=7,
        chunks_saved=7,
        index_path=tmp_path / "projects" / "demo" / "chunks.json",
    )


def test_ingest_repository_saves_embedded_chunks(tmp_path):
    result = ingest_repository(
        repo_path=SAMPLE_REPO,
        project_name="demo",
        data_dir=tmp_path,
        embedder=FakeEmbedder(),
    )

    chunks = load_chunks(tmp_path, "demo")

    assert result.index_path.exists()
    assert len(chunks) == result.chunks_saved
    assert all(chunk.embedding is not None for chunk in chunks)


def test_ingest_repository_propagates_project_name_to_chunks(tmp_path):
    ingest_repository(
        repo_path=SAMPLE_REPO,
        project_name="sample-project",
        data_dir=tmp_path,
        embedder=FakeEmbedder(),
    )

    chunks = load_chunks(tmp_path, "sample-project")

    assert {chunk.project_name for chunk in chunks} == {"sample-project"}


def test_ingest_repository_creates_empty_index_for_empty_repo(tmp_path):
    repo_path = tmp_path / "empty-repo"
    repo_path.mkdir()
    data_dir = tmp_path / "data"

    result = ingest_repository(
        repo_path=repo_path,
        project_name="empty",
        data_dir=data_dir,
        embedder=FakeEmbedder(),
    )

    assert result == IngestionResult(
        project_name="empty",
        files_discovered=0,
        documents_loaded=0,
        chunks_created=0,
        chunks_saved=0,
        index_path=data_dir / "projects" / "empty" / "chunks.json",
    )
    assert load_chunks(data_dir, "empty") == []


def test_ingest_repository_raises_for_missing_repository(tmp_path):
    with pytest.raises(FileNotFoundError, match="Repository does not exist"):
        ingest_repository(
            repo_path=tmp_path / "missing-repo",
            project_name="demo",
            data_dir=tmp_path / "data",
            embedder=FakeEmbedder(),
        )


def test_ingest_repository_raises_for_invalid_project_name(tmp_path):
    with pytest.raises(
        ValueError,
        match="Project name does not match the requirements",
    ):
        ingest_repository(
            repo_path=SAMPLE_REPO,
            project_name="bad/name",
            data_dir=tmp_path,
            embedder=FakeEmbedder(),
        )


def test_ingest_repository_replaces_existing_project_index(tmp_path):
    first_result = ingest_repository(
        repo_path=SAMPLE_REPO,
        project_name="demo",
        data_dir=tmp_path,
        embedder=FakeEmbedder(),
    )
    first_chunks = load_chunks(tmp_path, "demo")

    repo_path = tmp_path / "smaller-repo"
    repo_path.mkdir()
    (repo_path / "README.md").write_text("# Smaller repo\n", encoding="utf-8")

    second_result = ingest_repository(
        repo_path=repo_path,
        project_name="demo",
        data_dir=tmp_path,
        embedder=FakeEmbedder(),
    )
    second_chunks = load_chunks(tmp_path, "demo")

    assert first_result.index_path == second_result.index_path
    assert len(first_chunks) == 7
    assert len(second_chunks) == 1
    assert second_chunks[0].file_path == "README.md"
