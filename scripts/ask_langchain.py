"""Command-line entry point for answering through the LangChain boundary."""

import argparse
import sys
from pathlib import Path

# Temporary bootstrap for direct script execution with the src/ layout.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from contextforge.embedder import FakeEmbedder, GeminiEmbedder, OpenAIEmbedder
from contextforge.langchain_rag import ask_with_langchain
from contextforge.llm import FakeLLM, GeminiLLM, OpenAILLM


def make_embedder(name: str):
    """Create the configured embedding provider for CLI usage."""
    if name == "gemini":
        return GeminiEmbedder()
    if name == "openai":
        return OpenAIEmbedder()
    return FakeEmbedder()


def make_llm(name: str):
    """Create the configured answer-generation provider for CLI usage."""
    if name == "gemini":
        return GeminiLLM()
    if name == "openai":
        return OpenAILLM()
    return FakeLLM("I do not have enough evidence to answer.")


def main():
    """Parse CLI arguments, run LangChain-boundary ask, and print the answer."""
    parser = argparse.ArgumentParser(
        description="Ask through the LangChain boundary for an ingested project."
    )

    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="Name of the ingested project to ask about",
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
        help="Question to answer using retrieved source chunks",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of source chunks to use",
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default="fake",
        choices=["fake", "gemini", "openai"],
        help="The embedding model type to use",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="fake",
        choices=["fake", "gemini", "openai"],
        help="The answer-generation model type to use",
    )

    args = parser.parse_args()

    embedder = make_embedder(args.embedder)
    llm = make_llm(args.llm)

    answer = ask_with_langchain(
        data_dir=Path(args.data_dir),
        project_name=args.project,
        question=args.question,
        embedder=embedder,
        llm=llm,
        top_k=args.top_k,
    )

    print(f"Question: {args.question}")
    print()
    print(answer.text)

    if answer.sources:
        print()
        print("Sources:")
        for source in answer.sources:
            print(
                f"[{source.label}] score={source.score:.4f} "
                f"{source.file_path}:{source.start_line}-{source.end_line}"
            )


if __name__ == "__main__":
    main()
