# Learning Log

## Phase 1 - Manual RAG

Phase 1 built a working retrieval-augmented generation pipeline without using a
RAG framework. The goal was to understand each boundary directly before adding
abstractions.

### What We Built

The completed pipeline can:

- discover supported files in a local repository
- load files as text documents
- split documents into source-aware chunks
- embed chunks with Gemini, OpenAI, or a deterministic fake embedder
- save chunks and embeddings into a local JSON index
- retrieve top-k chunks with cosine similarity
- build a grounded prompt with source labels
- generate an answer with Gemini, OpenAI, or a fake LLM
- return file and line citations
- run repeatable retrieval evals

The main CLI commands are:

```bash
uv run python scripts/ingest_repo.py --path /Users/akhilnair/Desktop/HttpGo --name http-go-eval --data-dir data --embedder gemini
uv run python scripts/search.py --project http-go-eval --data-dir data --question "How does RequestFromReader handle partial reads and incomplete requests?" --top-k 3 --embedder gemini
uv run python scripts/ask.py --project http-go-eval --data-dir data --question "How does RequestFromReader handle partial reads and incomplete requests?" --top-k 3 --embedder gemini --llm gemini
uv run python scripts/eval_retrieval.py --eval-file evals/phase_1_questions.json
```

### Key Lessons

The data models matter because every pipeline stage depends on the same
contract. `Document`, `Chunk`, `SearchResult`, `Source`, and `Answer` keep the
system understandable and make tests easier to write.

Discovery and loading should stay boring. They should not know about embeddings,
retrieval, prompts, or LLMs. Their job is to turn repository files into clean
text inputs.

Chunking is a retrieval-quality decision. Line-based chunks are easy to inspect
and make citations simple, but they do not understand functions, structs, or
semantic code boundaries.

Embedding providers need validation at the boundary. The code validates vector
counts, dimensions, and numeric values so a bad provider response cannot silently
corrupt storage or retrieval.

Fake providers are not shortcuts. They let the pipeline be tested without
network access, API quota, or nondeterministic model output. Gemini and OpenAI
are used for real behavior; fakes are used for reliable tests.

Retrieval and answering are separate responsibilities. Retrieval finds evidence.
Prompt building formats that evidence. The LLM answers only from that supplied
context.

Evals revealed useful behavior. Gemini retrieved valid test files for the header
parsing question because tests directly describe behavior. That is acceptable for
Phase 1, but future evals may need stricter scoring if implementation files are
preferred over tests.

### Phase 1 Eval Result

The live Gemini retrieval eval against `/Users/akhilnair/Desktop/HttpGo` passed:

```text
Summary: 6/6 passed
```

This means each eval question retrieved at least one expected source file in the
top 3 results.

### Known Limitations

- JSON storage loads all chunks into memory.
- Retrieval is semantic-only, not hybrid keyword plus vector search.
- Results can include duplicate chunks from the same file.
- Line-based chunking can split logical code units.
- There is no reranking step.
- There is no API server, UI, or background ingestion worker.
- Live Gemini evals require a valid API key and can change slightly over time.

### Phase 2 Direction

Phase 2 should improve retrieval quality before adding agent behavior:

- dedupe repeated files or overlapping chunks in retrieval results
- add better chunking around code symbols
- add hybrid keyword plus vector retrieval
- add structured eval metrics beyond pass/fail
- consider reranking after initial top-k retrieval
