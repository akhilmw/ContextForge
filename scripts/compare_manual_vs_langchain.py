"""Compare manual retrieval with the LangChain-boundary retrieval path."""

import argparse
import sys
from pathlib import Path

# Temporary bootstrap for direct script execution with the src/ layout.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from contextforge.embedder import FakeEmbedder, GeminiEmbedder, OpenAIEmbedder
from contextforge.langchain_rag import retrieve_with_langchain
from contextforge.models import SearchResult
from contextforge.retriever import retrieve


def make_embedder(name: str):
    """Create the configured embedding provider for CLI usage."""
    if name == "gemini":
        return GeminiEmbedder()
    if name == "openai":
        return OpenAIEmbedder()
    return FakeEmbedder()


def result_paths(results: list[SearchResult]) -> list[str]:
    """Return ranked file paths for comparison output."""
    return [result.chunk.file_path for result in results]


def print_results(label: str, results: list[SearchResult]) -> None:
    """Print one ranked retrieval result list."""
    print(f"{label}:")
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        print(
            f"  [{index}] score={result.score:.4f} "
            f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}"
        )


def main():
    """Compare ranked manual and LangChain-boundary retrieval results."""
    parser = argparse.ArgumentParser(
        description="Compare manual retrieval with LangChain-boundary retrieval."
    )

    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="Name of the ingested project to compare",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing processed project data",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to retrieve relevant source chunks for",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of ranked chunks to compare",
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default="fake",
        choices=["fake", "gemini", "openai"],
        help="The embedding model type to use",
    )

    args = parser.parse_args()
    embedder = make_embedder(args.embedder)
    data_dir = Path(args.data_dir)

    manual_results = retrieve(
        data_dir=data_dir,
        project_name=args.project,
        question=args.question,
        embedder=embedder,
        top_k=args.top_k,
    )
    langchain_results = retrieve_with_langchain(
        data_dir=data_dir,
        project_name=args.project,
        question=args.question,
        embedder=embedder,
        top_k=args.top_k,
    )

    matches = result_paths(manual_results) == result_paths(langchain_results)

    print(f"Question: {args.question}")
    print()
    print_results("Manual", manual_results)
    print()
    print_results("LangChain", langchain_results)
    print()
    print(f"Match: {'yes' if matches else 'no'}")

    if not matches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
