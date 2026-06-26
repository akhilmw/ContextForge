import subprocess
import sys
from pathlib import Path

from contextforge.embedder import FakeEmbedder
from contextforge.ingest import ingest_repository


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "compare_manual_vs_langchain.py"
SAMPLE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


def run_script(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_compare_manual_vs_langchain_script_reports_matching_results(tmp_path):
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
        "3",
        "--embedder",
        "fake",
    )

    assert result.returncode == 0
    assert "Question: sample request parsing" in result.stdout
    assert "Manual:" in result.stdout
    assert "LangChain:" in result.stdout
    assert "[1] score=" in result.stdout
    assert "Match: yes" in result.stdout


def test_compare_manual_vs_langchain_script_rejects_unknown_embedder(tmp_path):
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


def test_compare_manual_vs_langchain_script_requires_question(tmp_path):
    result = run_script(
        "--project",
        "demo",
        "--data-dir",
        str(tmp_path),
        "--embedder",
        "fake",
    )

    assert result.returncode != 0
    assert "the following arguments are required: --question" in result.stderr
