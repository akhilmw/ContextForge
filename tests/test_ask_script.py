import subprocess
import sys
from pathlib import Path

from contextforge.embedder import FakeEmbedder
from contextforge.ingest import ingest_repository


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "ask.py"
SAMPLE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


def run_script(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_ask_script_prints_answer_and_sources(tmp_path):
    ingest_repository(
        repo_path=SAMPLE_REPO,
        project_name="demo",
        data_dir=tmp_path,
        embedder=FakeEmbedder(),
    )

    result = run_script(
        "--project",
        "demo",
        "--data-dir",
        str(tmp_path),
        "--question",
        "sample request parsing",
        "--top-k",
        "2",
        "--embedder",
        "fake",
        "--llm",
        "fake",
    )

    assert result.returncode == 0
    assert "Question: sample request parsing" in result.stdout
    assert "I do not have enough evidence to answer." in result.stdout
    assert "Sources:" in result.stdout
    assert "[S1] score=" in result.stdout
    assert "[S2] score=" in result.stdout


def test_ask_script_rejects_unknown_embedder(tmp_path):
    result = run_script(
        "--project",
        "demo",
        "--data-dir",
        str(tmp_path),
        "--question",
        "sample request parsing",
        "--embedder",
        "unknown",
    )

    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_ask_script_rejects_unknown_llm(tmp_path):
    result = run_script(
        "--project",
        "demo",
        "--data-dir",
        str(tmp_path),
        "--question",
        "sample request parsing",
        "--llm",
        "unknown",
    )

    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_ask_script_requires_question(tmp_path):
    result = run_script(
        "--project",
        "demo",
        "--data-dir",
        str(tmp_path),
        "--embedder",
        "fake",
        "--llm",
        "fake",
    )

    assert result.returncode != 0
    assert "the following arguments are required: --question" in result.stderr
