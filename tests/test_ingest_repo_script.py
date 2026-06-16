import subprocess
import sys
from pathlib import Path

from contextforge.store import load_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "ingest_repo.py"
SAMPLE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


def run_script(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_ingest_repo_script_runs_with_fake_embedder(tmp_path):
    result = run_script(
        "--path",
        str(SAMPLE_REPO),
        "--name",
        "demo",
        "--data-dir",
        str(tmp_path),
        "--embedder",
        "fake",
    )

    assert result.returncode == 0
    assert "Ingested project: demo" in result.stdout
    assert "Files discovered: 7" in result.stdout
    assert "Documents loaded: 7" in result.stdout
    assert "Chunks created: 7" in result.stdout
    assert "Chunks saved: 7" in result.stdout
    assert f"Index path: {tmp_path / 'projects' / 'demo' / 'chunks.json'}" in result.stdout

    chunks = load_chunks(tmp_path, "demo")
    assert len(chunks) == 7
    assert all(chunk.embedding is not None for chunk in chunks)


def test_ingest_repo_script_rejects_unknown_embedder(tmp_path):
    result = run_script(
        "--path",
        str(SAMPLE_REPO),
        "--name",
        "demo",
        "--data-dir",
        str(tmp_path),
        "--embedder",
        "unknown",
    )

    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_ingest_repo_script_requires_path_name_and_data_dir_defaults():
    result = run_script(
        "--name",
        "demo",
        "--embedder",
        "fake",
    )

    assert result.returncode != 0
    assert "the following arguments are required: --path" in result.stderr
