"""Generate and validate vectors for chunks and user questions.

The Embedder protocol keeps the ingestion and retrieval pipelines independent
of a specific provider. Gemini and OpenAI supply real semantic vectors, while
the fake implementation keeps unit tests deterministic and offline.
"""

import math
import os
from typing import Protocol

from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI

from contextforge.models import Chunk

load_dotenv()


class Embedder(Protocol):
    """Operations required from any embedding provider."""

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class GeminiEmbedder:
    """Gemini-backed embedding provider for document retrieval."""

    def __init__(self, model : str = "gemini-embedding-001",):
        """Create a Gemini client using the API key from the environment."""
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Could not fetch the API Key")
        self.model = model
        self.client = genai.Client(api_key=self.api_key)


    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Embed document text for storage and later semantic retrieval."""
        if not texts:
            return []

        # Document and query task types are intentionally different, but the
        # resulting vectors remain in a compatible retrieval space.
        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768,
            ),
        )

        # Convert SDK response objects into plain vectors used by our models.
        vectors = [embedding.values for embedding in response.embeddings]
        validate_embeddings(texts, vectors)
        return vectors


    def embed_query(self, text: str) -> list[float]:
        """Embed one user question for comparison with document vectors."""

        if not text.strip():
            raise ValueError("Query text cannot be empty")
        
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768,
            ),
        )
        vector = response.embeddings[0].values
        validate_embeddings([text], [vector])
        return vector


class OpenAIEmbedder:
    """OpenAI-backed embedding provider for document retrieval."""

    def __init__(self, model: str = "text-embedding-3-small"):
        """Create an OpenAI client using the API key from the environment."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Could not fetch the OpenAI API Key")
        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Embed document text for storage and later semantic retrieval."""
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        vectors = [item.embedding for item in response.data]
        validate_embeddings(texts, vectors)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed one user question for comparison with document vectors."""
        if not text.strip():
            raise ValueError("Query text cannot be empty")

        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
        )

        vector = response.data[0].embedding
        validate_embeddings([text], [vector])
        return vector


class FakeEmbedder:
    """Deterministic offline embedder used to test pipeline behavior."""

    def embed_documents(self, texts : list[str]) -> list[list[float]]:
        """Return one predictable three-dimensional vector per text."""
        if not texts:
            return []
        
        vectors = [_fake_vector(text) for text in texts]

        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed one query using the same deterministic text transformation."""
        if not text.strip():
            raise ValueError("Query text cannot be empty")
        
        vectors = self.embed_documents([text])
        validate_embeddings([text], vectors)
        return vectors[0]


def embed_chunks(
    chunks: list[Chunk],
    embedder: Embedder,
) -> list[Chunk]:  
    """Attach provider-generated vectors to chunks in their existing order."""
    
    if not chunks:
        return []
    
    # Batch requests are more efficient for real providers and preserve the
    # one-to-one ordering between chunks and returned vectors.
    texts = [chunk.content for chunk in chunks]

    vectors = embedder.embed_documents(texts)

    # Validate at the pipeline boundary as well as inside known providers so a
    # future custom Embedder cannot silently leave chunks unembedded.
    validate_embeddings(texts, vectors)
    
    for chunk, vector in zip(chunks, vectors):
        chunk.embedding = vector

    return chunks


def _fake_vector(text: str) -> list[float]:
    """Create a stable test vector from simple properties of the input text."""
    return [
        float(len(text)),
        float(len(text.split())),
        float(sum(ord(character) for character in text) % 1000),
    ]


def validate_embeddings(
    texts: list[str],
    vectors: list[list[float]],
) -> None:
    """Validate count, dimensions, and numeric values for a vector batch."""
    # Every input must receive exactly one vector.
    if len(vectors) != len(texts):
        raise ValueError(
            f"Expected {len(texts)} vectors, received {len(vectors)}"
        )

    if not vectors:
        return

    expected_dimension = len(vectors[0])

    if expected_dimension == 0:
        raise ValueError("Embedding vectors cannot be empty")

    for vector in vectors:
        # Cosine similarity requires all vectors to share one dimension.
        if len(vector) != expected_dimension:
            raise ValueError(
                "Embedding vectors have inconsistent dimensions"
            )

        for value in vector:
            # NaN and infinity are floats but cannot produce reliable scores.
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise TypeError("Embedding values must be numeric")
