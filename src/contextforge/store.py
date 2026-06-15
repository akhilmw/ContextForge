"""Persist embedded project chunks as a validated local JSON index.

Phase 1 uses one index file per project. The store converts Chunk models to
JSON-compatible dictionaries and writes replacements atomically so a failed
ingestion does not corrupt an existing index.
"""

import json
import os
import re
import tempfile
from dataclasses import asdict
from pathlib import Path

from contextforge.config import PROJECT_NAME_ALLOWED_PATTERN
from contextforge.embedder import validate_embeddings
from contextforge.models import Chunk


def chunk_to_dict(chunk: Chunk) -> dict:
    """Return an independent JSON-compatible representation of a Chunk."""
    return asdict(chunk)


def chunk_from_dict(data: dict) -> Chunk:
    """Reconstruct a Chunk and rerun its model-level validation."""
    return Chunk(**data)


def save_chunks(
    data_dir: Path,
    project_name: str,
    chunks: list[Chunk],
) -> Path:
    """Validate and atomically replace one project's local chunk index."""

    # Restrict names to one safe path segment so projects cannot escape the
    # configured data directory or create unexpected nested paths.
    if not re.match(PROJECT_NAME_ALLOWED_PATTERN, project_name):
        raise ValueError("Project name does not match the requirements")

    texts = [chunk.content for chunk in chunks]
    vectors = []

    for chunk in chunks:
        # Only complete ingestion results should become durable stored data.
        if chunk.embedding is None:
            raise ValueError(
                f"Chunk {chunk.chunk_id} has no embedding"
            )

        # Keep each persisted index isolated to exactly one project.
        if chunk.project_name != project_name:
            raise ValueError(
                f"Chunk {chunk.chunk_id} does not belong to {project_name}"
            )

        vectors.append(chunk.embedding)

    # Reuse embedding validation to enforce count, dimensions, and finite
    # numeric values before writing the index.
    validate_embeddings(texts, vectors)

    embedding_dimension = len(vectors[0]) if vectors else None

    # Store every project under its own stable index path.
    target_path = data_dir / "projects" / project_name / "chunks.json"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    temp_file_path = None

    # Write a complete temporary index in the destination directory. Keeping
    # both paths on the same filesystem allows os.replace() to be atomic.
    try:
        with tempfile.NamedTemporaryFile('w', dir = target_path.parent, delete=False, encoding='utf-8') as tf:
            temp_file_path = tf.name

            payload = {
                "schema_version": 1,
                "project_name": project_name,
                "embedding_dimension": embedding_dimension,
                "chunks": [chunk_to_dict(c) for c in chunks]
            }

            json.dump(payload, tf, indent=4)
            tf.flush()
            os.fsync(tf.fileno())
            temp_file_path = tf.name

        # Replace the previous index only after the new file is fully written
        # and closed, preserving the old index if serialization fails.
        os.replace(temp_file_path, target_path)
        temp_file_path = None
    finally:
        # Remove an incomplete temporary file after any failed write or swap.
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass

    return target_path
