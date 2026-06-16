from dataclasses import dataclass
from pathlib import Path

@dataclass
class Document:
    """A source file loaded from a repository.

    Attributes:
        file_path: Repository-relative path to the source file.
        language: Programming or document language inferred from the file type.
        content: Complete text content of the source file.
    """

    file_path : str
    language: str
    content: str

@dataclass
class Chunk:
    """A searchable section of a document.

    Attributes:
        chunk_id: Stable identifier for this section of source content.
        project_name: Name of the project that owns the source file.
        file_path: Repository-relative path to the original source file.
        language: Programming or document language of the source content.
        content: Text contained within this chunk.
        start_line: One-based first source line included in the chunk.
        end_line: One-based last source line included in the chunk.
        embedding: Vector representation of the content, or None before embedding.
    """

    chunk_id: str
    project_name: str
    file_path: str
    language: str
    content: str
    start_line: int
    end_line: int
    embedding: list[float] | None = None

    def __post_init__(self):
        if self.start_line <= 0 or self.end_line <= 0:
            raise ValueError("start line or end line cannot be zero or negative")
        elif self.start_line > self.end_line:
            raise ValueError("start line cannot be greater than end line")
    

@dataclass
class SearchResult:
    """A chunk returned by retrieval with its relevance score.

    Attributes:
        chunk: Retrieved source chunk.
        score: Cosine-similarity score between the question and chunk.
    """

    chunk : Chunk
    score : float

@dataclass
class Source:
    """Citation information presented with a generated answer.

    Attributes:
        label: Short prompt and answer reference, such as S1.
        file_path: Repository-relative path to the cited source file.
        start_line: One-based first cited source line.
        end_line: One-based last cited source line.
        score: Retrieval relevance score for the cited chunk.
    """

    label: str
    file_path: str
    start_line: int
    end_line: int
    score: float

@dataclass
class Answer:
    """A generated response and the sources used to ground it.

    Attributes:
        text: Generated answer text.
        sources: Citations for the retrieved chunks used by the answer.
    """

    text: str
    sources: list[Source]

@dataclass
class IngestionResult:
    project_name: str
    files_discovered: int
    documents_loaded: int
    chunks_created: int
    chunks_saved: int
    index_path: Path
