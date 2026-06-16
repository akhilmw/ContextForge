# ContextForge

ContextForge is a learning-first RAG project for asking source-grounded
questions about a local codebase. Phase 1 intentionally builds the pipeline
manually before introducing frameworks.

## Phase 1 Status

Phase 1 supports:

- local repository ingestion
- file discovery and UTF-8 loading
- line-based chunking with source metadata
- Gemini or fake embeddings
- local JSON chunk indexes
- cosine-similarity retrieval
- grounded prompt construction
- Gemini or fake LLM answers
- source citations with file paths and line ranges
- repeatable retrieval evals

## Setup

Install dependencies:

```bash
uv sync
```

Create a local `.env` file:

```bash
cp .env.example .env
```

Set your Gemini key:

```bash
GEMINI_API_KEY=your_key_here
```

## Ingest A Repository

Example using the HttpGo repo:

```bash
uv run python scripts/ingest_repo.py \
  --path /Users/akhilnair/Desktop/HttpGo \
  --name http-go-eval \
  --data-dir data \
  --embedder gemini
```

This creates:

```text
data/projects/http-go-eval/chunks.json
```

## Search Retrieved Sources

Use this when you want to inspect retrieval without asking an LLM:

```bash
uv run python scripts/search.py \
  --project http-go-eval \
  --data-dir data \
  --question "How does RequestFromReader handle partial reads and incomplete requests?" \
  --top-k 3 \
  --embedder gemini
```

## Ask A Question

Use this for the full RAG flow:

```bash
uv run python scripts/ask.py \
  --project http-go-eval \
  --data-dir data \
  --question "How does RequestFromReader handle partial reads and incomplete requests?" \
  --top-k 3 \
  --embedder gemini \
  --llm gemini
```

The output includes the generated answer followed by source citations.

## Run Retrieval Evals

Run against an existing ingested index:

```bash
uv run python scripts/eval_retrieval.py \
  --eval-file evals/phase_1_questions.json
```

Re-ingest the eval repo first:

```bash
uv run python scripts/eval_retrieval.py \
  --eval-file evals/phase_1_questions.json \
  --ingest
```

The Phase 1 eval checks whether each question retrieves at least one expected
source file in the top-k results.

## Offline Mode

Most tests use fake providers so they do not need network access or Gemini
quota:

```bash
uv run pytest -q
```

The fake embedder is deterministic but not semantically meaningful. Use Gemini
for realistic retrieval behavior.

## Current Limitations

- JSON storage loads all project chunks into memory.
- Retrieval is semantic-only, not hybrid keyword plus vector search.
- Chunks are line-based and do not understand code symbols.
- Re-ingestion replaces the whole project index.
- There is no API server or web UI yet.
