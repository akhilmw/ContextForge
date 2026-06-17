# Architecture

## Phase 1 Overview

ContextForge Phase 1 is a local, manual RAG system. It ingests a repository into
a local chunk index, retrieves relevant chunks for a question, and asks an LLM to
answer using only those retrieved sources.

The design keeps each boundary small:

- ingestion modules prepare source-aware chunks
- storage persists embedded chunks
- retrieval returns ranked evidence
- prompt building formats evidence for the LLM
- LLM providers generate final text
- CLIs only parse arguments and print output

## Ingestion Pipeline

```text
Repository path
  -> discovery.discover_files()
  -> loaders.load_documents()
  -> chunker.chunk_documents()
  -> embedder.embed_chunks()
  -> store.save_chunks()
  -> data/projects/<project>/chunks.json
```

Responsibilities:

- `discovery.py` finds supported files and skips ignored directories.
- `loaders.py` reads repository-relative files as UTF-8 text.
- `chunker.py` creates line-based chunks with stable IDs and line ranges.
- `embedder.py` attaches vectors through a fake, Gemini, or OpenAI provider.
- `store.py` writes a project-specific JSON index atomically.
- `ingest.py` coordinates the whole ingestion flow.
- `scripts/ingest_repo.py` exposes ingestion through the terminal.

## Ask Pipeline

```text
Question and project name
  -> retriever.retrieve()
  -> store.load_chunks()
  -> embedder.embed_query()
  -> cosine similarity scoring
  -> prompt_builder.build_prompt()
  -> llm.generate()
  -> models.Answer
```

Responsibilities:

- `retriever.py` embeds the question, scores stored chunks, and returns top-k
  `SearchResult` objects.
- `prompt_builder.py` labels retrieved chunks as `[S1]`, `[S2]`, and builds a
  prompt that tells the LLM not to guess beyond the supplied evidence.
- `llm.py` hides fake and Gemini answer-generation details behind one protocol.
- `ask.py` coordinates retrieval, prompt construction, generation, and source
  conversion.
- `scripts/search.py` prints retrieved chunks for debugging retrieval.
- `scripts/ask.py` runs the full question-answering path from the terminal.

## Evaluation Pipeline

```text
evals/phase_1_questions.json
  -> scripts/eval_retrieval.py
  -> retriever.retrieve()
  -> compare retrieved files with expected_files
  -> PASS/FAIL summary
```

The Phase 1 eval measures retrieval only. It does not grade generated answers.
That separation makes it easier to tell whether a bad answer came from poor
retrieval or from the LLM generation step.

## Core Data Models

- `Document`: one loaded source file.
- `Chunk`: one searchable source section with file path, line range, content,
  and optional embedding.
- `SearchResult`: one retrieved chunk plus its similarity score.
- `Source`: citation metadata shown with an answer.
- `Answer`: final generated text plus sources.
- `IngestionResult`: ingestion summary returned by the orchestration layer.

## Provider Boundaries

`Embedder` and `LLM` are protocols. The rest of the system depends on their
interfaces, not concrete provider classes.

Current providers:

- `FakeEmbedder`: deterministic vectors for tests.
- `GeminiEmbedder`: live Gemini embeddings for real retrieval.
- `OpenAIEmbedder`: live OpenAI embeddings for real retrieval.
- `FakeLLM`: deterministic answer text for tests.
- `GeminiLLM`: live Gemini generation for real answers.
- `OpenAILLM`: live OpenAI generation for real answers.

## Storage Layout

Generated project indexes are stored under:

```text
data/projects/<project-name>/chunks.json
```

The `data/` directory is ignored by git except for `.gitkeep`, because indexes
are generated artifacts.

## Phase 1 Boundaries

Phase 1 intentionally does not include:

- LangChain or LangGraph
- vector databases
- FastAPI
- web UI
- agents
- background jobs
- GitHub ingestion
- log ingestion

These are future architecture layers. Phase 1 keeps the core RAG mechanics
visible and testable.
