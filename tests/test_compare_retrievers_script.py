import json
import subprocess
import sys
from pathlib import Path

from contextforge.embedder import FakeEmbedder
from contextforge.ingest import ingest_repository


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "compare_retrievers.py"
SAMPLE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


def run_script(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def write_eval_file(tmp_path, data_dir):
    eval_file = tmp_path / "eval.json"
    eval_file.write_text(
        json.dumps(
            {
                "name": "test-compare",
                "repo_path": str(SAMPLE_REPO),
                "project_name": "demo",
                "data_dir": str(data_dir),
                "embedder": "fake",
                "top_k": 3,
                "questions": [
                    {
                        "id": "sample-question",
                        "question": "sample request parsing",
                        "expected_files": [
                            "docs/architecture.md",
                            "notes/debug.txt",
                            "src/parser.py",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return eval_file


def test_compare_retrievers_script_prints_all_strategy_rows(tmp_path):
    data_dir = tmp_path / "data"
    ingest_repository(
        repo_path=SAMPLE_REPO,
        project_name="demo",
        data_dir=data_dir,
        embedder=FakeEmbedder(),
    )
    eval_file = write_eval_file(tmp_path, data_dir)

    result = run_script(
        "--eval-file",
        str(eval_file),
        "--candidate-k",
        "5",
        "--rank-constant",
        "40",
        "--overlap-threshold",
        "0.6",
        "--max-per-file",
        "2",
        "--bm25-k1",
        "1.2",
        "--bm25-b",
        "0.6",
    )

    assert result.returncode == 0
    assert "Eval: test-compare" in result.stdout
    assert "Project: demo" in result.stdout
    assert "Strategy" in result.stdout
    assert "semantic" in result.stdout
    assert "deduplicated" in result.stdout
    assert "diverse" in result.stdout
    assert "keyword" in result.stdout
    assert "hybrid" in result.stdout
    assert "hybrid-reranked" in result.stdout
    assert "1/1" in result.stdout
    assert "1.0000" in result.stdout


def test_compare_retrievers_script_requires_eval_file():
    result = run_script()

    assert result.returncode != 0
    assert "the following arguments are required: --eval-file" in result.stderr
