"""Build source-grounded prompts from retrieval results."""

from contextforge.models import SearchResult, Source


def results_to_sources(results: list[SearchResult]) -> list[Source]:
    """Convert ranked retrieval results into user-facing citation metadata."""

    label = 1
    sources = []
    for sr in results:
        # Labels stay bracket-free here so presentation layers can choose how
        # to render them, e.g. `S1` in data and `[S1]` in prompts.
        source = Source(f"S{label}", sr.chunk.file_path, sr.chunk.start_line, sr.chunk.end_line, sr.score)
        label += 1
        sources.append(source)

    return sources


def build_prompt(
    question: str,
    results: list[SearchResult],
) -> str:
    """Create a deterministic prompt that constrains the LLM to retrieved sources."""

    if not question.strip():
        raise ValueError("question cannot be empty")

    sources = results_to_sources(results)
    source_blocks = []
    for source, result in zip(sources, results):
        # Build prompt source blocks from SearchResult so chunk content is
        # included while embedding vectors stay out of the prompt.
        block = (
            f"[{source.label}] {source.file_path}:{source.start_line}-{source.end_line}\n"
            f"{result.chunk.content}"
        )
        source_blocks.append(block)

    # Make the no-context case explicit so the model follows the fallback.
    sources_text = "\n\n".join(source_blocks) if source_blocks else "No sources were retrieved."

    prompt = (
        "You are ContextForge, a source-grounded engineering assistant.\n\n"
        "Answer the question using only the provided sources.\n"
        "If the sources do not contain enough evidence, say: \"I do not have enough evidence to answer.\"\n\n"
        f"Question:\n{question}\n\n"
        f"Sources:\n{sources_text}\n\n"
        "Instructions:\n"
        "- Cite sources using [S1], [S2], etc.\n"
        "- Do not cite files that are not listed.\n"
        "- Do not guess beyond the sources."
    )

    return prompt
