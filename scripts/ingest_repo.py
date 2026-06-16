"""Command-line entry point for Phase 1 repository ingestion."""

import argparse
import sys
from pathlib import Path

# The project currently uses a src/ layout without an installed console script.
# This lets the script run directly with `uv run python scripts/ingest_repo.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from contextforge.ingest import ingest_repository
from contextforge.embedder import FakeEmbedder, GeminiEmbedder


def main():
    """Parse CLI arguments, run ingestion, and print a human-readable summary."""
    parser = argparse.ArgumentParser(
        description = "Ingest a repository for embedding and analysis."
    )

    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the repository to ingest"
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the Project"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to store processed data"
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default="fake",
        choices=["fake", "gemini"],
        help="The embedding model type to use"
    )

    args = parser.parse_args()

    # Keep provider choice at the edge; ingestion only receives an Embedder.
    embedder = GeminiEmbedder() if args.embedder == "gemini" else FakeEmbedder()
    repo_path = Path(args.path)
    data_dir = Path(args.data_dir)

    result = ingest_repository(repo_path, args.name, data_dir, embedder)

    print("--- Ingestion Completed ---")

    print(f"Ingested project: {result.project_name}")
    print(f"Files discovered: {result.files_discovered}")
    print(f"Documents loaded: {result.documents_loaded}")
    print(f"Chunks created: {result.chunks_created}")
    print(f"Chunks saved: {result.chunks_saved}")
    print(f"Index path: {result.index_path}")

    print("-------------------------")


if __name__ == "__main__":
    main()
