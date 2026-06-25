# Phase 3 - LangChain Comparison

## Goal

Rebuild the existing manual ContextForge RAG flow with LangChain and compare
the framework-based implementation against the manual pipeline.

The goal is not to add new retrieval tricks. Phase 2 already improved retrieval
quality. Phase 3 is about understanding what LangChain standardizes, what it
hides, and whether the abstraction improves the project.

## Starting Point

Phase 3 starts from the completed Phase 2 retrieval stack:

```text
Question
  -> semantic candidates
  -> BM25 keyword candidates
  -> Reciprocal Rank Fusion
  -> heuristic reranking
  -> overlap deduplication
  -> source diversity
  -> top-k SearchResults
```

The manual implementation remains the source of truth for behavior. LangChain
should be introduced behind clear adapter boundaries so the public ContextForge
models stay stable.

## Phase Boundaries

Phase 3 includes:

- LangChain document adapters
- LangChain-compatible chunk metadata
- a LangChain retrieval path over the existing local chunk index
- a LangChain ask path using the existing prompt and answer contracts
- side-by-side comparison with the manual pipeline
- deterministic tests using fakes
- documentation of tradeoffs

Phase 3 does not include:

- LangGraph agents
- PostgreSQL or pgvector
- FastAPI
- web UI
- production deployment
- new ingestion sources
- replacing the manual implementation

## Target Architecture

### Adapter Boundary

```text
ContextForge Chunk
  -> LangChain Document
  -> LangChain retriever / chain
  -> ContextForge SearchResult / Answer
```

LangChain objects should stay inside `langchain_adapters.py` and
`langchain_rag.py`. Scripts and app-level callers should continue to use
ContextForge domain models.

### LangChain Ask Pipeline

```text
Question
  -> load stored chunks
  -> convert chunks to LangChain Documents
  -> retrieve relevant documents
  -> build prompt
  -> call chat model
  -> convert sources back to ContextForge Source models
  -> return ContextForge Answer
```

## Folder and File Structure

Expected Phase 3 files:

```text
ContextForge/
  docs/
    phase-notes/
      phase-3.md
    phase-3-comparison.md

  scripts/
    ask_langchain.py
    compare_manual_vs_langchain.py

  src/contextforge/
    langchain_adapters.py
    langchain_rag.py

  tests/
    test_langchain_adapters.py
    test_langchain_rag.py
    test_ask_langchain_script.py
    test_compare_manual_vs_langchain_script.py
```

Do not create a separate `phase3` package. LangChain integration belongs in the
main `contextforge` package behind explicit adapter modules.

## Implementation Order

### Step 1 - Add LangChain Adapters

Files:

- `src/contextforge/langchain_adapters.py`
- `tests/test_langchain_adapters.py`

Tasks:

- Convert `Chunk` to LangChain `Document`.
- Preserve chunk ID, project name, file path, language, line range, and
  embedding in metadata.
- Convert LangChain `Document` back to `Chunk`.
- Raise clear errors when required metadata is missing or malformed.

Checkpoint:

- A `Chunk -> Document -> Chunk` round trip preserves citation metadata.

### Step 2 - Build LangChain Retrieval

Files:

- `src/contextforge/langchain_rag.py`
- `tests/test_langchain_rag.py`

Tasks:

- Load chunks from the existing JSON store.
- Convert chunks to LangChain documents.
- Use a deterministic fake retriever in tests.
- Return ContextForge `SearchResult` objects at the public boundary.

Checkpoint:

- LangChain retrieval can be tested without network calls.

### Step 3 - Build LangChain Ask

Files:

- `src/contextforge/langchain_rag.py`
- `scripts/ask_langchain.py`

Tasks:

- Reuse the existing prompt-building expectations where practical.
- Call the configured LLM through LangChain.
- Return the existing `Answer` and `Source` models.
- Keep source citations intact.

Checkpoint:

- The LangChain ask script returns an answer with file and line citations.

### Step 4 - Compare Manual vs LangChain

Files:

- `scripts/compare_manual_vs_langchain.py`
- `docs/phase-3-comparison.md`

Tasks:

- Run the same questions through the manual and LangChain paths.
- Compare output shape, citation quality, testability, code complexity, and
  dependency cost.
- Document what LangChain simplified and what control it hid.

Checkpoint:

- The final Phase 3 conclusion is based on concrete behavior, not preference.

## Dependency Strategy

Do not install LangChain packages until implementation begins.

When work starts, add only the packages needed for the first adapter/retrieval
step. Keep dependency changes in the same commit as the first code that uses
them.

## Definition of Done

Phase 3 is complete when:

- LangChain adapters preserve ContextForge metadata
- a LangChain retrieval path works with deterministic tests
- a LangChain ask path returns `Answer` and `Source` models
- manual and LangChain paths are compared on the same evaluation set
- docs explain the tradeoffs clearly
- the manual pipeline remains operational

## Expected Learning Outcomes

By the end of Phase 3, you should be able to explain:

- what a LangChain `Document` maps to in the manual pipeline
- what a LangChain retriever abstracts away
- where framework boundaries should live in a RAG codebase
- how to keep domain models independent from framework objects
- what LangChain simplifies
- what control or visibility LangChain reduces
