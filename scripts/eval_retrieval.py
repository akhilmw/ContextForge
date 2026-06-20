"""Command-line retrieval evaluation for Phase 1 indexes."""

import argparse
import json
import sys
from pathlib import Path

# Temporary bootstrap for direct script execution with the src/ layout.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from contextforge.embedder import FakeEmbedder, GeminiEmbedder, OpenAIEmbedder
from contextforge.evaluation import evaluate_case, summarize_results
from contextforge.ingest import ingest_repository
from contextforge.retriever import retrieve, retrieve_deduplicated, retrieve_diverse


def load_eval_config(eval_file: Path) -> dict:
    """Load the JSON eval config from disk."""
    with eval_file.open(encoding="utf-8") as file:
        return json.load(file)


def make_embedder(name: str):
    """Create the configured embedding provider."""
    if name == "gemini":
        return GeminiEmbedder()
    if name == "openai":
        return OpenAIEmbedder()
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
    parser.add_argument(
        "--strategy",
        choices=("semantic", "deduplicated", "diverse"),
        default="semantic",
        help="Retrieval strategy to evaluate (default: semantic)",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=15,
        help="Semantic candidates fetched before deduplication (default: 15)",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.5,
        help="Overlap ratio removed by deduplicated retrieval (default: 0.5)",
    )
    parser.add_argument(
        "--max-per-file",
        type=int,
        default=1,
        help="Maximum final candidates retained per file (default: 1)",
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
    print(f"Strategy: {args.strategy}")
    print(f"Top K: {top_k}")
    if args.strategy in ("deduplicated", "diverse"):
        print(f"Candidate K: {args.candidate_k}")
        print(f"Overlap threshold: {args.overlap_threshold}")
    if args.strategy == "diverse":
        print(f"Max per file: {args.max_per_file}")
    print()

    # Keep each result so run-level metrics can be calculated after retrieval.
    case_results = []
    for item in questions:
        retrieval_args = {
            "data_dir": data_dir,
            "project_name": project_name,
            "question": item["question"],
            "embedder": embedder,
            "top_k": top_k,
        }
        if args.strategy == "deduplicated":
            results = retrieve_deduplicated(
                **retrieval_args,
                candidate_k=args.candidate_k,
                overlap_threshold=args.overlap_threshold,
            )
        elif args.strategy == "diverse":
            results = retrieve_diverse(
                **retrieval_args,
                candidate_k=args.candidate_k,
                overlap_threshold=args.overlap_threshold,
                max_per_file=args.max_per_file,
            )
        else:
            results = retrieve(**retrieval_args)

        retrieved_files = [result.chunk.file_path for result in results]
        expected_files = item["expected_files"]
        case_result = evaluate_case(
            question_id=item["id"],
            retrieved_files=retrieved_files,
            expected_files=expected_files,
        )
        case_results.append(case_result)
        hit = (
            case_result.first_relevant_rank is not None
            and case_result.first_relevant_rank <= top_k
        )

        if hit:
            passed += 1

        status = "PASS" if hit else "FAIL"
        print(f"{status} {item['id']}")
        print(f"  expected any: {expected_files}")
        print(f"  got: {retrieved_files}")
        print()

    # Hit rate measures success frequency; MRR rewards earlier relevant results.
    summary = summarize_results(case_results, k=top_k)
    print(f"Summary: {passed}/{total} passed")
    print(f"Hit Rate@{top_k}: {summary.hit_rate:.4f}")
    print(f"MRR: {summary.mrr:.4f}")

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
