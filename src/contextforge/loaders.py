"""Load discovered repository files into Document objects.

Discovery decides which relative file paths are eligible. This module safely
reads those files and attaches the metadata needed by later pipeline stages.
"""

from pathlib import Path

from contextforge.models import Document
from contextforge.config import LANGUAGE_BY_EXTENSION


def load_document(repo_path: Path, file_path: Path) -> Document:
    """Read one repository-relative file and return its Document model."""

    # Resolve both paths before validating containment. This normalizes `..`
    # segments and follows the actual filesystem location of the source file.
    repo_abs = repo_path.resolve()
    full_path = (repo_path / file_path).resolve()

    # Prevent a caller from reading a file outside the requested repository.
    if not full_path.is_relative_to(repo_abs):
        raise ValueError("Path is outside repository")
    
    if not full_path.exists():
        raise FileNotFoundError("File does not exists")
    
    if full_path.is_dir():
        raise IsADirectoryError("Path is not a file path")
    
    # Map the supported extension to a stable language label.
    extension = full_path.suffix
    if extension in LANGUAGE_BY_EXTENSION:
        language = LANGUAGE_BY_EXTENSION[extension]

        # Explicit UTF-8 decoding makes behavior consistent across machines.
        content = full_path.read_text(encoding="utf-8")

        # Store a normalized, portable path for future source citations.
        relative_path = full_path.relative_to(repo_abs)
        document = Document(str(relative_path), language, content)
        return document
    else:
        raise ValueError("Unsupported extension")


def load_documents(repo_path : Path, file_paths : list[Path]) -> list[Document]:
    """Load multiple repository-relative files while preserving input order."""

    documents = []
    for file_path in file_paths:
        # Reuse the single-file checks and metadata handling for every path.
        document = load_document(repo_path, file_path)
        documents.append(document)

    return documents
