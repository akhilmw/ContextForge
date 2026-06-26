"""LangChain-backed RAG helpers that preserve ContextForge domain models."""

from pathlib import Path

from contextforge.embedder import Embedder
from contextforge.langchain_adapters import (
    chunk_to_langchain_document,
    langchain_document_to_chunk,
)
from contextforge.llm import LLM
from contextforge.models import Answer, SearchResult
from contextforge.prompt_builder import build_prompt, results_to_sources
from contextforge.retriever import cosine_similarity
from contextforge.store import load_chunks


def retrieve_with_langchain(
    data_dir: Path,
    project_name: str,
    question: str,
    embedder: Embedder,
    top_k: int = 5,
) -> list[SearchResult]:
    """Retrieve chunks through a LangChain Document conversion boundary."""

    if not question.strip():
        raise ValueError("question cannot be empty")

    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError(f"{top_k} is not of type int")

    if top_k <= 0:
        raise ValueError("top k cannot be less than or equal to zero")

    chunks = load_chunks(data_dir, project_name)
    if not chunks:
        return []

    documents = [chunk_to_langchain_document(chunk) for chunk in chunks]

    question_vector = embedder.embed_query(question)

    results: list[SearchResult] = []
    for document in documents:
        chunk = langchain_document_to_chunk(document)
        score = cosine_similarity(question_vector, chunk.embedding)
        results.append(SearchResult(chunk=chunk, score=score))

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_k]


def ask_with_langchain(
    data_dir: Path,
    project_name: str,
    question: str,
    embedder: Embedder,
    llm: LLM,
    top_k: int = 5,
) -> Answer:
    """Retrieve through the LangChain boundary, then generate a grounded answer."""
    results = retrieve_with_langchain(
        data_dir=data_dir,
        project_name=project_name,
        question=question,
        embedder=embedder,
        top_k=top_k,
    )
    sources = results_to_sources(results)
    prompt = build_prompt(question, results)
    response = llm.generate(prompt)

    return Answer(text=response, sources=sources)
