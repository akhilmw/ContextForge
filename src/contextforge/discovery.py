"""Find source files that are eligible for repository ingestion.

This module only discovers paths. Reading file contents and creating Document
objects are handled later by the loader.
"""

from pathlib import Path

from contextforge.config import SUPPORTED_EXTENSIONS, IGNORED_DIRECTORIES


def _dfs(repo_path : Path, path : Path, paths : list[Path]):
    """Recursively collect supported files below the current path."""
    if not path.exists():
        return
    
    for child in path.iterdir():
        # Prune ignored directories so none of their contents are inspected.
        if child.is_dir() and child.name in IGNORED_DIRECTORIES:
            continue

        if child.is_file() and child.suffix in SUPPORTED_EXTENSIONS:
            # Store portable paths instead of machine-specific absolute paths.
            relative_path = child.relative_to(repo_path)
            paths.append(relative_path)

        if child.is_dir():
            _dfs(repo_path, child, paths)


def discover_files(repo_path: Path) -> list[Path]:
    """Return sorted, repository-relative paths for supported source files."""

    # Validate at the public boundary so traversal failures have clear messages.
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository does not exist: {repo_path}")

    if not repo_path.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {repo_path}")

    paths = []
    _dfs(repo_path, repo_path, paths)
    # Filesystem traversal order varies, so sort for repeatable ingestion.
    paths.sort()
    return paths
