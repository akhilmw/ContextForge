"""Run the full question-answering pipeline for an indexed project."""

from pathlib import Path

from contextforge.embedder import Embedder
from contextforge.llm import LLM
from contextforge.models import Answer
from contextforge.prompt_builder import build_prompt, results_to_sources
from contextforge.retriever import retrieve


def ask_question(
    data_dir: Path,
    project_name: str,
    question: str,
    embedder: Embedder,
    llm: LLM,
    top_k: int = 5,
) -> Answer:
    """Retrieve context, build a grounded prompt, and generate an answer."""

    # Retrieval owns loading the saved index, embedding the question, and ranking
    # matching chunks. The ask layer only coordinates the pipeline.
    results = retrieve(data_dir, project_name, question, embedder, top_k)

    # Prompt construction turns ranked chunks into source-labelled context for
    # the LLM, while sources become structured citation metadata for the caller.
    prompt = build_prompt(question, results)
    sources = results_to_sources(results)

    response = llm.generate(prompt)

    return Answer(text=response, sources=sources)
