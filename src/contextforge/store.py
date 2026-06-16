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

from contextforge.config import PROJECT_NAME_ALLOWED_PATTERN, SCHEMA_VERSION
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

def load_chunks(
    data_dir: Path,
    project_name: str,
) -> list[Chunk]:
    """Load and validate one project's stored chunk index."""

    if not re.match(PROJECT_NAME_ALLOWED_PATTERN, project_name):
        raise ValueError("Project name does not match the requirements")

    target_path = data_dir / "projects" / project_name / "chunks.json"
    if not target_path.exists():
        raise FileNotFoundError("Requested file not found")

    vectors = []
    texts = []
    chunks = []
    with open(target_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("Stored index must be a JSON object")

        schema_version = data.get("schema_version")
        if schema_version is None:
            raise ValueError("Schema Version not present")
        elif schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema version: {schema_version}"
            )

        if data.get("project_name") != project_name:
            raise ValueError(f"Invalid project name : {project_name}")

        expected_dimension = data.get("embedding_dimension")
        chunk_dicts = data.get("chunks")

        if not isinstance(chunk_dicts, list):
            raise ValueError("Stored chunks must be a list")

        if not chunk_dicts:
            if expected_dimension is not None:
                raise ValueError("Empty project cannot have an embedding dimension")
            return []

        if expected_dimension is None:
            raise ValueError("Non-empty project must have an embedding dimension")

        for chunk_dict in chunk_dicts:
            chunk = chunk_from_dict(chunk_dict)
            if chunk.project_name != project_name:
                raise ValueError(
                    f"Chunk {chunk.chunk_id} does not belong to {project_name}"
                )

            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.chunk_id} has no embedding")

            if len(chunk.embedding) != expected_dimension:
                raise ValueError("Embedding size mismatch")

            texts.append(chunk.content)
            vectors.append(chunk.embedding)
            chunks.append(chunk)

    validate_embeddings(texts, vectors)

    return chunks
