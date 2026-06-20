import json
import subprocess
import sys
from pathlib import Path

from contextforge.embedder import FakeEmbedder
from contextforge.ingest import ingest_repository


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "eval_retrieval.py"
SAMPLE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


def run_script(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def write_eval_file(tmp_path, data_dir, expected_files):
    eval_file = tmp_path / "eval.json"
    eval_file.write_text(
        json.dumps(
            {
                "name": "test-eval",
                "repo_path": str(SAMPLE_REPO),
                "project_name": "demo",
                "data_dir": str(data_dir),
                "embedder": "fake",
                "top_k": 3,
                "questions": [
                    {
                        "id": "sample-question",
                        "question": "sample request parsing",
                        "expected_files": expected_files,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return eval_file


def test_eval_retrieval_script_reports_pass(tmp_path):
    data_dir = tmp_path / "data"
    ingest_repository(
        repo_path=SAMPLE_REPO,
        project_name="demo",
        data_dir=data_dir,
        embedder=FakeEmbedder(),
    )
    eval_file = write_eval_file(
        tmp_path,
        data_dir,
        ["docs/architecture.md", "notes/debug.txt"],
    )

    result = run_script("--eval-file", str(eval_file))

    assert result.returncode == 0
    assert "Eval: test-eval" in result.stdout
    assert "Strategy: semantic" in result.stdout
    assert "PASS sample-question" in result.stdout
    assert "Summary: 1/1 passed" in result.stdout
    assert "Hit Rate@3: 1.0000" in result.stdout
    assert "MRR: 1.0000" in result.stdout


def test_eval_retrieval_script_reports_fail_with_nonzero_exit(tmp_path):
    data_dir = tmp_path / "data"
    ingest_repository(
        repo_path=SAMPLE_REPO,
        project_name="demo",
        data_dir=data_dir,
        embedder=FakeEmbedder(),
    )
    eval_file = write_eval_file(tmp_path, data_dir, ["missing.py"])

    result = run_script("--eval-file", str(eval_file))

    assert result.returncode == 1
    assert "FAIL sample-question" in result.stdout
    assert "Summary: 0/1 passed" in result.stdout
    assert "Hit Rate@3: 0.0000" in result.stdout
    assert "MRR: 0.0000" in result.stdout


def test_eval_retrieval_script_can_ingest_before_eval(tmp_path):
    data_dir = tmp_path / "data"
    eval_file = write_eval_file(
        tmp_path,
        data_dir,
        ["docs/architecture.md", "notes/debug.txt"],
    )

    result = run_script("--eval-file", str(eval_file), "--ingest")

    assert result.returncode == 0
    assert "PASS sample-question" in result.stdout
    assert "Summary: 1/1 passed" in result.stdout


def test_eval_retrieval_script_runs_deduplicated_strategy(tmp_path):
    data_dir = tmp_path / "data"
    ingest_repository(
        repo_path=SAMPLE_REPO,
        project_name="demo",
        data_dir=data_dir,
        embedder=FakeEmbedder(),
    )
    eval_file = write_eval_file(
        tmp_path,
        data_dir,
        ["docs/architecture.md", "notes/debug.txt"],
    )

    result = run_script(
        "--eval-file",
        str(eval_file),
        "--strategy",
        "deduplicated",
        "--candidate-k",
        "5",
        "--overlap-threshold",
        "0.6",
    )

    assert result.returncode == 0
    assert "Strategy: deduplicated" in result.stdout
    assert "Candidate K: 5" in result.stdout
    assert "Overlap threshold: 0.6" in result.stdout
    assert "PASS sample-question" in result.stdout


def test_eval_retrieval_script_rejects_unknown_strategy():
    result = run_script(
        "--eval-file",
        "unused.json",
        "--strategy",
        "unknown",
    )

    assert result.returncode != 0
    assert "invalid choice: 'unknown'" in result.stderr


def test_eval_retrieval_script_requires_eval_file():
    result = run_script()

    assert result.returncode != 0
    assert "the following arguments are required: --eval-file" in result.stderr
