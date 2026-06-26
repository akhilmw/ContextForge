"""Command-line entry point for searching through the LangChain boundary."""

import argparse
import sys
from pathlib import Path

# Temporary bootstrap for direct script execution with the src/ layout.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from contextforge.embedder import FakeEmbedder, GeminiEmbedder, OpenAIEmbedder
from contextforge.langchain_rag import retrieve_with_langchain


def make_embedder(name: str):
    """Create the configured embedding provider for CLI usage."""
    if name == "gemini":
        return GeminiEmbedder()
    if name == "openai":
        return OpenAIEmbedder()
    return FakeEmbedder()


def main():
    """Parse CLI arguments, run LangChain-boundary retrieval, and print chunks."""
    parser = argparse.ArgumentParser(
        description="Search an ingested index through the LangChain boundary."
    )

    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="Name of the ingested project to search",
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
        help="Maximum number of ranked chunks to return",
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
    results = retrieve_with_langchain(
        data_dir=Path(args.data_dir),
        project_name=args.project,
        question=args.question,
        embedder=embedder,
        top_k=args.top_k,
    )

    print(f"Top {len(results)} LangChain-boundary results for: {args.question}")

    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        print()
        print(
            f"[{index}] score={result.score:.4f} "
            f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}"
        )
        print(chunk.content.rstrip())


if __name__ == "__main__":
    main()
