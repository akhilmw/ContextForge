"""Command-line retrieval evaluation for Phase 1 indexes."""

import argparse
import json
import sys
from pathlib import Path

# Temporary bootstrap for direct script execution with the src/ layout.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from contextforge.embedder import FakeEmbedder, GeminiEmbedder
from contextforge.ingest import ingest_repository
from contextforge.retriever import retrieve


def load_eval_config(eval_file: Path) -> dict:
    """Load the JSON eval config from disk."""
    with eval_file.open(encoding="utf-8") as file:
        return json.load(file)


def make_embedder(name: str):
    """Create the configured embedding provider."""
    if name == "gemini":
        return GeminiEmbedder()
    if name == "fake":
        return FakeEmbedder()
    raise ValueError(f"Unsupported embedder: {name}")


def main():
    """Run retrieval questions and print top-k pass/fail results."""
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality for an ingested ContextForge project."
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        required=True,
        help="Path to the JSON eval file",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest the eval repo before running retrieval",
    )

    args = parser.parse_args()
    config = load_eval_config(Path(args.eval_file))

    data_dir = Path(config["data_dir"])
    project_name = config["project_name"]
    embedder_name = config.get("embedder", "fake")
    top_k = config.get("top_k", 3)
    questions = config["questions"]

    embedder = make_embedder(embedder_name)

    if args.ingest:
        # Rebuild the project index when the eval should reflect current repo
        # contents. Otherwise, reuse the existing index to avoid extra API calls.
        ingest_repository(
            repo_path=Path(config["repo_path"]),
            project_name=project_name,
            data_dir=data_dir,
            embedder=embedder,
        )

    passed = 0
    total = len(questions)

    print(f"Eval: {config.get('name', args.eval_file)}")
    print(f"Project: {project_name}")
    print(f"Embedder: {embedder_name}")
    print(f"Top K: {top_k}")
    print()

    for item in questions:
        results = retrieve(
            data_dir=data_dir,
            project_name=project_name,
            question=item["question"],
            embedder=embedder,
            top_k=top_k,
        )

        retrieved_files = [result.chunk.file_path for result in results]
        expected_files = item["expected_files"]
        hit = any(file_path in retrieved_files for file_path in expected_files)

        if hit:
            passed += 1

        status = "PASS" if hit else "FAIL"
        print(f"{status} {item['id']}")
        print(f"  expected any: {expected_files}")
        print(f"  got: {retrieved_files}")
        print()

    print(f"Summary: {passed}/{total} passed")

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
