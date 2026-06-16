import json
from pathlib import Path
from unittest.mock import patch

import pytest

from contextforge.models import Chunk
from contextforge.store import chunk_from_dict, chunk_to_dict, load_chunks, save_chunks


def make_chunk(embedding=None):
    return Chunk(
        chunk_id="chunk-1",
        project_name="demo",
        file_path="src/parser.py",
        language="python",
        content="def parse():\n    pass\n",
        start_line=1,
        end_line=2,
        embedding=embedding,
    )


def test_chunk_to_dict_contains_all_chunk_fields():
    chunk = make_chunk([0.1, 0.2, 0.3])

    data = chunk_to_dict(chunk)

    assert data == {
        "chunk_id": "chunk-1",
        "project_name": "demo",
        "file_path": "src/parser.py",
        "language": "python",
        "content": "def parse():\n    pass\n",
        "start_line": 1,
        "end_line": 2,
        "embedding": [0.1, 0.2, 0.3],
    }


def test_embedded_chunk_round_trips_through_dictionary():
    original = make_chunk([0.1, 0.2, 0.3])

    restored = chunk_from_dict(chunk_to_dict(original))

    assert restored == original


def test_unembedded_chunk_round_trips_through_dictionary():
    original = make_chunk()

    restored = chunk_from_dict(chunk_to_dict(original))

    assert restored == original
    assert restored.embedding is None


def test_round_trip_creates_independent_chunk_and_embedding_objects():
    original = make_chunk([0.1, 0.2, 0.3])

    restored = chunk_from_dict(chunk_to_dict(original))

    assert restored is not original
    assert restored.embedding is not original.embedding


def test_modifying_serialized_data_does_not_change_original_chunk():
    original = make_chunk([0.1, 0.2])
    data = chunk_to_dict(original)

    data["file_path"] = "src/changed.py"
    data["embedding"].append(0.3)

    assert original.file_path == "src/parser.py"
    assert original.embedding == [0.1, 0.2]


def test_chunk_from_dict_rejects_missing_required_fields():
    with pytest.raises(TypeError):
        chunk_from_dict({})


def test_chunk_from_dict_rejects_unknown_fields():
    data = chunk_to_dict(make_chunk())
    data["unknown"] = "value"

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        chunk_from_dict(data)


def test_chunk_from_dict_applies_chunk_validation():
    data = chunk_to_dict(make_chunk())
    data["start_line"] = 10
    data["end_line"] = 5

    with pytest.raises(
        ValueError,
        match="start line cannot be greater than end line",
    ):
        chunk_from_dict(data)


def test_save_chunks_writes_project_index_and_returns_path(tmp_path):
    chunks = [make_chunk([0.1, 0.2, 0.3])]

    saved_path = save_chunks(tmp_path, "demo", chunks)

    assert saved_path == tmp_path / "projects" / "demo" / "chunks.json"
    assert saved_path.exists()


def test_save_chunks_writes_expected_payload(tmp_path):
    chunk = make_chunk([0.1, 0.2, 0.3])

    saved_path = save_chunks(tmp_path, "demo", [chunk])
    payload = json.loads(saved_path.read_text(encoding="utf-8"))

    assert payload == {
        "schema_version": 1,
        "project_name": "demo",
        "embedding_dimension": 3,
        "chunks": [chunk_to_dict(chunk)],
    }


def test_save_chunks_records_null_dimension_for_empty_project(tmp_path):
    saved_path = save_chunks(tmp_path, "empty-project", [])
    payload = json.loads(saved_path.read_text(encoding="utf-8"))

    assert payload["embedding_dimension"] is None
    assert payload["chunks"] == []


def test_save_chunks_creates_missing_parent_directories(tmp_path):
    data_dir = tmp_path / "nested" / "data"

    saved_path = save_chunks(
        data_dir,
        "demo",
        [make_chunk([0.1, 0.2])],
    )

    assert saved_path.exists()
    assert saved_path.parent == data_dir / "projects" / "demo"


def test_save_chunks_replaces_existing_project_index(tmp_path):
    old_chunk = make_chunk([0.1, 0.2])
    new_chunk = Chunk(
        chunk_id="chunk-2",
        project_name="demo",
        file_path="src/new_parser.py",
        language="python",
        content="def parse_new():\n    pass\n",
        start_line=1,
        end_line=2,
        embedding=[0.3, 0.4],
    )

    save_chunks(tmp_path, "demo", [old_chunk])
    saved_path = save_chunks(tmp_path, "demo", [new_chunk])
    payload = json.loads(saved_path.read_text(encoding="utf-8"))

    assert [chunk["chunk_id"] for chunk in payload["chunks"]] == ["chunk-2"]


@pytest.mark.parametrize(
    "project_name",
    [
        "",
        "   ",
        ".",
        "..",
        "../outside",
        "nested/project",
        r"nested\project",
        "project name",
    ],
)
def test_save_chunks_rejects_invalid_project_names(tmp_path, project_name):
    with pytest.raises(
        ValueError,
        match="Project name does not match the requirements",
    ):
        save_chunks(tmp_path, project_name, [])


def test_save_chunks_rejects_chunk_without_embedding(tmp_path):
    chunk = make_chunk()

    with pytest.raises(
        ValueError,
        match="Chunk chunk-1 has no embedding",
    ):
        save_chunks(tmp_path, "demo", [chunk])


def test_save_chunks_rejects_chunk_from_different_project(tmp_path):
    chunk = make_chunk([0.1, 0.2])
    chunk.project_name = "other-project"

    with pytest.raises(
        ValueError,
        match="Chunk chunk-1 does not belong to demo",
    ):
        save_chunks(tmp_path, "demo", [chunk])


def test_save_chunks_rejects_inconsistent_embedding_dimensions(tmp_path):
    first = make_chunk([0.1, 0.2])
    second = Chunk(
        chunk_id="chunk-2",
        project_name="demo",
        file_path="src/second.py",
        language="python",
        content="second chunk",
        start_line=1,
        end_line=1,
        embedding=[0.3, 0.4, 0.5],
    )

    with pytest.raises(
        ValueError,
        match="Embedding vectors have inconsistent dimensions",
    ):
        save_chunks(tmp_path, "demo", [first, second])


def test_save_chunks_leaves_existing_index_unchanged_when_write_fails(tmp_path):
    original = make_chunk([0.1, 0.2])
    saved_path = save_chunks(tmp_path, "demo", [original])
    original_content = saved_path.read_text(encoding="utf-8")

    replacement = Chunk(
        chunk_id="chunk-2",
        project_name="demo",
        file_path="src/replacement.py",
        language="python",
        content="replacement",
        start_line=1,
        end_line=1,
        embedding=[0.3, 0.4],
    )

    with patch(
        "contextforge.store.json.dump",
        side_effect=RuntimeError("write failed"),
    ):
        with pytest.raises(RuntimeError, match="write failed"):
            save_chunks(tmp_path, "demo", [replacement])

    assert saved_path.read_text(encoding="utf-8") == original_content


def test_save_chunks_removes_temporary_file_when_write_fails(tmp_path):
    chunk = make_chunk([0.1, 0.2])
    project_dir = tmp_path / "projects" / "demo"

    with patch(
        "contextforge.store.json.dump",
        side_effect=RuntimeError("write failed"),
    ):
        with pytest.raises(RuntimeError, match="write failed"):
            save_chunks(tmp_path, "demo", [chunk])

    assert project_dir.exists()
    assert list(project_dir.iterdir()) == []


def test_saved_chunks_are_valid_json(tmp_path):
    saved_path = save_chunks(
        tmp_path,
        "demo",
        [make_chunk([0.1, 0.2])],
    )

    with saved_path.open(encoding="utf-8") as saved_file:
        payload = json.load(saved_file)

    assert isinstance(payload, dict)


def write_index(data_dir, project_name, payload):
    project_dir = data_dir / "projects" / project_name
    project_dir.mkdir(parents=True)
    index_path = project_dir / "chunks.json"
    index_path.write_text(json.dumps(payload), encoding="utf-8")
    return index_path


def test_load_chunks_round_trips_saved_chunks(tmp_path):
    chunks = [
        make_chunk([0.1, 0.2]),
        Chunk(
            chunk_id="chunk-2",
            project_name="demo",
            file_path="src/second.py",
            language="python",
            content="second chunk",
            start_line=1,
            end_line=1,
            embedding=[0.3, 0.4],
        ),
    ]

    save_chunks(tmp_path, "demo", chunks)
    loaded = load_chunks(tmp_path, "demo")

    assert loaded == chunks
    assert loaded is not chunks
    assert loaded[0] is not chunks[0]


def test_load_chunks_preserves_stored_order(tmp_path):
    chunks = [
        make_chunk([0.1, 0.2]),
        Chunk(
            chunk_id="chunk-2",
            project_name="demo",
            file_path="src/second.py",
            language="python",
            content="second chunk",
            start_line=1,
            end_line=1,
            embedding=[0.3, 0.4],
        ),
    ]

    save_chunks(tmp_path, "demo", chunks)
    loaded = load_chunks(tmp_path, "demo")

    assert [chunk.chunk_id for chunk in loaded] == ["chunk-1", "chunk-2"]


def test_load_chunks_loads_empty_project(tmp_path):
    save_chunks(tmp_path, "empty-project", [])

    assert load_chunks(tmp_path, "empty-project") == []


def test_load_chunks_raises_for_missing_index(tmp_path):
    with pytest.raises(FileNotFoundError, match="Requested file not found"):
        load_chunks(tmp_path, "demo")


@pytest.mark.parametrize(
    "project_name",
    [
        "",
        "   ",
        ".",
        "..",
        "../outside",
        "nested/project",
        r"nested\project",
        "project name",
    ],
)
def test_load_chunks_rejects_invalid_project_names(tmp_path, project_name):
    with pytest.raises(
        ValueError,
        match="Project name does not match the requirements",
    ):
        load_chunks(tmp_path, project_name)


def test_load_chunks_raises_for_invalid_json(tmp_path):
    project_dir = tmp_path / "projects" / "demo"
    project_dir.mkdir(parents=True)
    (project_dir / "chunks.json").write_text("{invalid json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_chunks(tmp_path, "demo")


def test_load_chunks_rejects_non_object_json_root(tmp_path):
    write_index(tmp_path, "demo", [])

    with pytest.raises(
        ValueError,
        match="Stored index must be a JSON object",
    ):
        load_chunks(tmp_path, "demo")


def test_load_chunks_rejects_missing_schema_version(tmp_path):
    payload = {
        "project_name": "demo",
        "embedding_dimension": None,
        "chunks": [],
    }
    write_index(tmp_path, "demo", payload)

    with pytest.raises(ValueError, match="Schema Version not present"):
        load_chunks(tmp_path, "demo")


def test_load_chunks_rejects_unsupported_schema_version(tmp_path):
    payload = {
        "schema_version": 2,
        "project_name": "demo",
        "embedding_dimension": None,
        "chunks": [],
    }
    write_index(tmp_path, "demo", payload)

    with pytest.raises(ValueError, match="Unsupported schema version: 2"):
        load_chunks(tmp_path, "demo")


def test_load_chunks_rejects_wrong_project_metadata(tmp_path):
    payload = {
        "schema_version": 1,
        "project_name": "other-project",
        "embedding_dimension": None,
        "chunks": [],
    }
    write_index(tmp_path, "demo", payload)

    with pytest.raises(ValueError, match="Invalid project name"):
        load_chunks(tmp_path, "demo")


@pytest.mark.parametrize("chunks_value", [None, {"chunk_id": "chunk-1"}])
def test_load_chunks_rejects_missing_or_non_list_chunks(tmp_path, chunks_value):
    payload = {
        "schema_version": 1,
        "project_name": "demo",
        "embedding_dimension": None,
        "chunks": chunks_value,
    }
    if chunks_value is None:
        payload.pop("chunks")

    write_index(tmp_path, "demo", payload)

    with pytest.raises(ValueError, match="Stored chunks must be a list"):
        load_chunks(tmp_path, "demo")


def test_load_chunks_rejects_empty_project_with_dimension(tmp_path):
    payload = {
        "schema_version": 1,
        "project_name": "demo",
        "embedding_dimension": 2,
        "chunks": [],
    }
    write_index(tmp_path, "demo", payload)

    with pytest.raises(
        ValueError,
        match="Empty project cannot have an embedding dimension",
    ):
        load_chunks(tmp_path, "demo")


def test_load_chunks_rejects_non_empty_project_without_dimension(tmp_path):
    chunk = make_chunk([0.1, 0.2])
    payload = {
        "schema_version": 1,
        "project_name": "demo",
        "embedding_dimension": None,
        "chunks": [chunk_to_dict(chunk)],
    }
    write_index(tmp_path, "demo", payload)

    with pytest.raises(
        ValueError,
        match="Non-empty project must have an embedding dimension",
    ):
        load_chunks(tmp_path, "demo")


def test_load_chunks_rejects_chunk_from_different_project(tmp_path):
    chunk = make_chunk([0.1, 0.2])
    chunk.project_name = "other-project"
    payload = {
        "schema_version": 1,
        "project_name": "demo",
        "embedding_dimension": 2,
        "chunks": [chunk_to_dict(chunk)],
    }
    write_index(tmp_path, "demo", payload)

    with pytest.raises(
        ValueError,
        match="Chunk chunk-1 does not belong to demo",
    ):
        load_chunks(tmp_path, "demo")


def test_load_chunks_rejects_chunk_without_embedding(tmp_path):
    chunk = make_chunk()
    payload = {
        "schema_version": 1,
        "project_name": "demo",
        "embedding_dimension": 2,
        "chunks": [chunk_to_dict(chunk)],
    }
    write_index(tmp_path, "demo", payload)

    with pytest.raises(ValueError, match="Chunk chunk-1 has no embedding"):
        load_chunks(tmp_path, "demo")


def test_load_chunks_rejects_embedding_dimension_mismatch(tmp_path):
    chunk = make_chunk([0.1, 0.2])
    payload = {
        "schema_version": 1,
        "project_name": "demo",
        "embedding_dimension": 3,
        "chunks": [chunk_to_dict(chunk)],
    }
    write_index(tmp_path, "demo", payload)

    with pytest.raises(ValueError, match="Embedding size mismatch"):
        load_chunks(tmp_path, "demo")


def test_load_chunks_applies_chunk_validation(tmp_path):
    chunk = make_chunk([0.1, 0.2])
    chunk_data = chunk_to_dict(chunk)
    chunk_data["start_line"] = 10
    chunk_data["end_line"] = 5
    payload = {
        "schema_version": 1,
        "project_name": "demo",
        "embedding_dimension": 2,
        "chunks": [chunk_data],
    }
    write_index(tmp_path, "demo", payload)

    with pytest.raises(
        ValueError,
        match="start line cannot be greater than end line",
    ):
        load_chunks(tmp_path, "demo")
