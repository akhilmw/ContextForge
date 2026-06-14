# Phase 1 - Manual RAG

## Goal

Build a small retrieval-augmented generation system manually so each part of
the pipeline is understood before introducing LangChain or other frameworks.

The system will:

1. Read supported files from a local repository.
2. Split those files into chunks that retain source metadata.
3. Generate and locally store embeddings for each chunk.
4. Retrieve the most relevant chunks for a question.
5. Ask an LLM to answer using only the retrieved context.
6. Return the answer with file and line citations.

## Phase Boundaries

Phase 1 includes:

- Local repository ingestion
- Text file discovery and loading
- Line-based chunking with overlap
- Embedding generation
- Local JSON persistence
- Cosine-similarity retrieval
- Top-k source selection
- Grounded prompt construction
- LLM answer generation
- File and line citations
- Unit tests for deterministic components
- A small retrieval evaluation set

Phase 1 does not include:

- LangChain
- LangGraph
- MCP
- FastAPI
- PostgreSQL or pgvector
- GitHub and log integrations
- Agent routing
- A web interface

## Architecture

### Ingestion pipeline

```text
Repository path
  -> discover supported files
  -> load files as Documents
  -> split Documents into Chunks
  -> generate chunk embeddings
  -> attach embeddings to Chunks
  -> save the project index as local JSON
```

### Ask pipeline

```text
Question and project name
  -> generate a query embedding
  -> load the project's stored Chunks
  -> calculate cosine similarity
  -> select the top-k Chunks
  -> build a grounded prompt with labeled sources
  -> generate an LLM answer
  -> return the answer and source citations
```

Retrieval and generation remain separate. The retriever returns evidence; it
does not generate an answer. The LLM receives selected evidence; it does not
search the repository directly.

## Implementation Order

### Step 1 - Project configuration

Files:

- `pyproject.toml`
- `.env.example`
- `src/contextforge/config.py`

Tasks:

- Choose and document the supported Python version.
- Add only the dependencies required for manual RAG and testing.
- Define environment variables for model credentials and the data directory.
- Define supported extensions, ignored directories, chunk settings, and the
  default retrieval count in one configuration module.

Checkpoint:

- The package can be installed in a local virtual environment.
- Configuration loads without requiring application modules to know how
  environment variables are stored.

### Step 2 - Domain models

File:

- `src/contextforge/models.py`

Define the data passed between modules:

- `Document`: one loaded source file
- `Chunk`: one searchable section with source metadata and an embedding
- `SearchResult`: a retrieved Chunk and its similarity score
- `Source`: citation details derived from a retrieved Chunk
- `Answer`: generated text and the sources used

Required Chunk metadata:

- Stable chunk ID
- Project name
- Repository-relative file path
- Language
- Content
- Start line
- End line
- Embedding

Checkpoint:

- Models can be serialized for JSON storage without losing source metadata.
- No model performs file access, retrieval, or LLM calls.

### Step 3 - File discovery

File:

- `src/contextforge/discovery.py`

Tasks:

- Recursively scan a supplied repository path.
- Include only configured text-based extensions.
- Skip ignored directories such as `.git`, `node_modules`, virtual
  environments, caches, and build output.
- Return deterministic, normalized paths.
- Fail clearly when the repository path does not exist or is not a directory.

Tests:

- `tests/test_discovery.py`

Checkpoint:

- The sample repository returns only the expected files in stable order.

### Step 4 - Document loading

File:

- `src/contextforge/loaders.py`

Tasks:

- Read discovered files as text.
- Preserve repository-relative paths.
- Map known extensions to a language label.
- Skip or report binary, unreadable, and unsupported files safely.
- Return `Document` objects.

Checkpoint:

- Loading does not modify source files.
- One bad file does not silently corrupt the rest of an ingestion run.

### Step 5 - Chunking

File:

- `src/contextforge/chunker.py`

Tasks:

- Implement simple line-based chunking.
- Add configurable overlap between adjacent chunks.
- Preserve exact start and end line numbers.
- Generate stable chunk IDs from project and source metadata.
- Handle short, empty, and final partial chunks explicitly.

Tests:

- `tests/test_chunker.py`

Checkpoint:

- Every non-empty source line appears in at least one Chunk.
- Re-ingesting unchanged content produces the same chunk IDs.

### Step 6 - Embedding boundary

File:

- `src/contextforge/embedder.py`

Tasks:

- Define a small embedding interface.
- Support separate document and query embedding operations.
- Add one concrete provider implementation.
- Validate vector counts and dimensions.
- Keep provider-specific request logic inside this module.

Testing approach:

- Use a deterministic fake embedder in unit tests.
- Do not depend on a network API for discovery, chunking, store, or retrieval
  tests.

Checkpoint:

- A list of chunk texts produces the same number of vectors.
- A question produces one vector with the expected dimension.

### Step 7 - Local project store

File:

- `src/contextforge/store.py`

Tasks:

- Save embedded Chunks to a project-specific JSON index.
- Load Chunks for one requested project.
- Validate required fields and embedding dimensions when loading.
- Define re-ingestion as a full replacement of that project's index for
  Phase 1.
- Write the completed index atomically so a failed ingestion does not leave a
  partially written project file.

Suggested generated path:

```text
data/projects/<project-name>/chunks.json
```

Tests:

- `tests/test_store.py`

Checkpoint:

- Saved data survives a save/load round trip.
- Two projects cannot overwrite or retrieve each other's data.

### Step 8 - Ingestion orchestration

Files:

- `src/contextforge/ingest.py`
- `scripts/ingest_repo.py`

Tasks:

- Connect discovery, loading, chunking, embedding, and storage.
- Keep orchestration in `ingest.py`.
- Keep argument parsing and terminal output in the script.
- Print an ingestion summary with file, chunk, skipped-file, and error counts.

Expected command:

```bash
python scripts/ingest_repo.py --path ../http-go --name http-go
```

Checkpoint:

- The command creates a complete local index for the requested project.
- Running it again replaces that project's old index predictably.

### Step 9 - Similarity retrieval

File:

- `src/contextforge/retriever.py`

Tasks:

- Embed the user's question.
- Load only the requested project index.
- Calculate cosine similarity against every stored Chunk.
- Sort results from highest to lowest score.
- Return no more than the requested `top_k`.
- Handle empty indexes, zero vectors, and dimension mismatches clearly.

Tests:

- `tests/test_retriever.py`

Checkpoint:

- Deterministic fake vectors produce an exactly predictable ranking.

### Step 10 - Grounded prompt construction

File:

- `src/contextforge/prompt_builder.py`

Tasks:

- Label retrieved chunks as `[S1]`, `[S2]`, and so on.
- Include file paths and line ranges with each source.
- Instruct the model to use only the supplied evidence.
- Require source labels for factual claims.
- Require an explicit insufficient-evidence response when context does not
  support an answer.

Checkpoint:

- Prompt construction is deterministic and testable without an LLM call.

### Step 11 - LLM boundary and ask orchestration

Files:

- `src/contextforge/llm.py`
- `src/contextforge/ask.py`
- `scripts/ask.py`

Tasks:

- Keep provider-specific generation logic inside `llm.py`.
- Have `ask.py` coordinate retrieval, prompt construction, generation, and
  source conversion.
- Keep CLI parsing and output formatting in `scripts/ask.py`.
- Print the answer followed by source paths and line ranges.

Expected command:

```bash
python scripts/ask.py \
  "How does RequestFromReader handle partial reads?" \
  --project http-go
```

Checkpoint:

- The answer references only sources returned by retrieval.
- The displayed citations map back to exact stored Chunks.

### Step 12 - Evaluation and documentation

Files:

- `evals/phase_1_questions.json`
- `README.md`
- `docs/architecture.md`
- `docs/learning-log.md`

Tasks:

- Add at least five realistic questions.
- Record expected files for each question.
- Measure whether the expected files appear in the top-k results.
- Document setup, ingestion, asking, example output, and limitations.
- Record chunking and retrieval observations in the learning log.

Checkpoint:

- Evaluation can distinguish retrieval failure from answer-generation failure.

## Test Strategy

Unit tests should cover deterministic behavior:

- File inclusion and exclusion
- Chunk boundaries, overlap, line numbers, and IDs
- Store serialization and project isolation
- Cosine similarity, ordering, and top-k behavior
- Prompt source labels and insufficient-evidence instructions

External embedding and LLM providers should be replaced by fakes in unit
tests. A small optional integration test may exercise real providers when
credentials are available.

## Definition of Done

Phase 1 is complete when:

- A local repository can be ingested from the CLI.
- Supported files become source-aware Chunks.
- Chunks and embeddings are stored in a project-specific local index.
- A question retrieves ranked evidence using cosine similarity.
- The LLM receives only the retrieved evidence.
- The answer includes file and line citations.
- Insufficient evidence produces an honest fallback.
- Core deterministic modules have passing unit tests.
- At least five evaluation questions record expected source files.
- The README and learning log describe the implementation and findings.

## Known Phase 1 Limitations

- Local JSON storage requires loading all project vectors into memory.
- Retrieval is semantic-only, without keyword or hybrid search.
- Line-based chunks do not understand code syntax or symbols.
- Re-ingestion replaces the entire project index.
- There is no API, multi-step agent, external tool access, or production
  observability.
