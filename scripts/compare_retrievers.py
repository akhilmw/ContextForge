"""Compare retrieval strategies on one eval file."""

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
from contextforge.keyword_retriever import retrieve_keywords
from contextforge.reranker import HeuristicReranker
from contextforge.retriever import (
    retrieve,
    retrieve_deduplicated,
    retrieve_diverse,
    retrieve_hybrid,
)

STRATEGIES = (
    "semantic",
    "deduplicated",
    "diverse",
    "keyword",
    "hybrid",
    "hybrid-reranked",
)


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


def retrieve_for_strategy(
    strategy: str,
    data_dir: Path,
    project_name: str,
    question: str,
    embedder,
    top_k: int,
    candidate_k: int,
    overlap_threshold: float,
    max_per_file: int,
    bm25_k1: float,
    bm25_b: float,
    rank_constant: int,
):
    """Run one retrieval strategy for one question."""

    if strategy == "keyword":
        return retrieve_keywords(
            data_dir=data_dir,
            project_name=project_name,
            question=question,
            top_k=top_k,
            k1=bm25_k1,
            b=bm25_b,
        )

    retrieval_args = {
        "data_dir": data_dir,
        "project_name": project_name,
        "question": question,
        "embedder": embedder,
        "top_k": top_k,
    }

    if strategy == "semantic":
        return retrieve(**retrieval_args)

    if strategy == "deduplicated":
        return retrieve_deduplicated(
            **retrieval_args,
            candidate_k=candidate_k,
            overlap_threshold=overlap_threshold,
        )

    if strategy == "diverse":
        return retrieve_diverse(
            **retrieval_args,
            candidate_k=candidate_k,
            overlap_threshold=overlap_threshold,
            max_per_file=max_per_file,
        )

    if strategy == "hybrid":
        return retrieve_hybrid(
            **retrieval_args,
            candidate_k=candidate_k,
            rank_constant=rank_constant,
            overlap_threshold=overlap_threshold,
            max_per_file=max_per_file,
            k1=bm25_k1,
            b=bm25_b,
        )

    if strategy == "hybrid-reranked":
        return retrieve_hybrid(
            **retrieval_args,
            candidate_k=candidate_k,
            rank_constant=rank_constant,
            overlap_threshold=overlap_threshold,
            max_per_file=max_per_file,
            k1=bm25_k1,
            b=bm25_b,
            reranker=HeuristicReranker(),
        )

    raise ValueError(f"Unsupported strategy: {strategy}")


def evaluate_strategy(
    strategy: str,
    config: dict,
    embedder,
    candidate_k: int,
    overlap_threshold: float,
    max_per_file: int,
    bm25_k1: float,
    bm25_b: float,
    rank_constant: int,
):
    """Evaluate one strategy across every question in the config."""

    data_dir = Path(config["data_dir"])
    project_name = config["project_name"]
    top_k = config.get("top_k", 3)
    case_results = []

    for item in config["questions"]:
        results = retrieve_for_strategy(
            strategy=strategy,
            data_dir=data_dir,
            project_name=project_name,
            question=item["question"],
            embedder=embedder,
            top_k=top_k,
            candidate_k=candidate_k,
            overlap_threshold=overlap_threshold,
            max_per_file=max_per_file,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
            rank_constant=rank_constant,
        )
        retrieved_files = [result.chunk.file_path for result in results]
        case_results.append(
            evaluate_case(
                question_id=item["id"],
                retrieved_files=retrieved_files,
                expected_files=item["expected_files"],
            )
        )

    return summarize_results(case_results, k=top_k)


def main():
    """Print a compact metrics table across retrieval strategies."""

    parser = argparse.ArgumentParser(
        description="Compare ContextForge retrieval strategies on one eval file."
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        required=True,
        help="Path to the JSON eval file",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=15,
        help="Candidates fetched before post-processing (default: 15)",
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
    parser.add_argument(
        "--bm25-k1",
        type=float,
        default=1.5,
        help="BM25 term-frequency saturation setting (default: 1.5)",
    )
    parser.add_argument(
        "--bm25-b",
        type=float,
        default=0.75,
        help="BM25 document-length normalization setting (default: 0.75)",
    )
    parser.add_argument(
        "--rank-constant",
        type=int,
        default=60,
        help="RRF rank constant used by hybrid retrieval (default: 60)",
    )

    args = parser.parse_args()
    config = load_eval_config(Path(args.eval_file))
    embedder_name = config.get("embedder", "fake")
    embedder = make_embedder(embedder_name)

    print(f"Eval: {config.get('name', args.eval_file)}")
    print(f"Project: {config['project_name']}")
    print(f"Embedder: {embedder_name}")
    print(f"Top K: {config.get('top_k', 3)}")
    print()
    print(f"{'Strategy':<18} {'Hits':>7} {'Hit Rate':>10} {'MRR':>8}")
    print("-" * 46)

    for strategy in STRATEGIES:
        summary = evaluate_strategy(
            strategy=strategy,
            config=config,
            embedder=embedder,
            candidate_k=args.candidate_k,
            overlap_threshold=args.overlap_threshold,
            max_per_file=args.max_per_file,
            bm25_k1=args.bm25_k1,
            bm25_b=args.bm25_b,
            rank_constant=args.rank_constant,
        )
        hits = f"{summary.hits}/{summary.total_questions}"
        print(
            f"{strategy:<18} "
            f"{hits:>7} "
            f"{summary.hit_rate:>10.4f} "
            f"{summary.mrr:>8.4f}"
        )


if __name__ == "__main__":
    main()
