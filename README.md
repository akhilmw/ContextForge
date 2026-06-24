# ContextForge

ContextForge is a learning-first RAG project for asking source-grounded
questions about a local codebase. The project intentionally builds the core
pipeline manually before introducing frameworks.

## Phase 1 Status

Phase 1 supports:

- local repository ingestion
- file discovery and UTF-8 loading
- line-based chunking with source metadata
- Gemini, OpenAI, or fake embeddings
- local JSON chunk indexes
- cosine-similarity retrieval
- grounded prompt construction
- Gemini, OpenAI, or fake LLM answers
- source citations with file paths and line ranges
- repeatable retrieval evals

## Phase 2 Status

Phase 2 focuses on retrieval quality and measurement. It adds:

- Hit@K, Hit Rate@K, reciprocal rank, and MRR metrics
- exact chunk-ID deduplication
- same-file line-overlap deduplication
- candidate overfetching before final top-k selection
- source diversity limits
- manual BM25 keyword retrieval
- Reciprocal Rank Fusion for semantic plus keyword retrieval
- local heuristic reranking
- side-by-side retrieval comparison scripts

Measured on the HttpGo eval set with OpenAI embeddings and top 3 retrieval:

| Strategy | Hits | Hit Rate@3 | MRR |
|---|---:|---:|---:|
| Semantic | 4/6 | 0.6667 | 0.5000 |
| Diverse semantic | 5/6 | 0.8333 | 0.5833 |
| BM25 keyword | 2/6 | 0.3333 | 0.1667 |
| Hybrid RRF | 6/6 | 1.0000 | 0.6667 |
| Hybrid + heuristic reranker | 6/6 | 1.0000 | 0.9167 |

LangChain is deferred to a later phase so the retrieval work can stand on its
own and be compared cleanly.

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
OPENAI_API_KEY=your_key_here
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

Use OpenAI instead:

```bash
uv run python scripts/ingest_repo.py \
  --path /Users/akhilnair/Desktop/HttpGo \
  --name http-go-openai \
  --data-dir data \
  --embedder openai
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

OpenAI version:

```bash
uv run python scripts/ask.py \
  --project http-go-openai \
  --data-dir data \
  --question "How does RequestFromReader handle partial reads and incomplete requests?" \
  --top-k 3 \
  --embedder openai \
  --llm openai
```

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

Run the Phase 2 semantic baseline:

```bash
uv run python scripts/eval_retrieval.py \
  --eval-file evals/phase_2_questions.json \
  --strategy semantic
```

Run hybrid retrieval with heuristic reranking:

```bash
uv run python scripts/eval_retrieval.py \
  --eval-file evals/phase_2_questions.json \
  --strategy hybrid-reranked \
  --candidate-k 15 \
  --rank-constant 60 \
  --overlap-threshold 0.25 \
  --max-per-file 1 \
  --bm25-k1 1.5 \
  --bm25-b 0.75
```

Compare all Phase 2 retrieval strategies:

```bash
uv run python scripts/compare_retrievers.py \
  --eval-file evals/phase_2_questions.json \
  --candidate-k 15 \
  --rank-constant 60 \
  --overlap-threshold 0.25 \
  --max-per-file 1 \
  --bm25-k1 1.5 \
  --bm25-b 0.75
```

## Offline Mode

Most tests use fake providers so they do not need network access or Gemini
quota:

```bash
uv run pytest -q
```

The fake embedder is deterministic but not semantically meaningful. Use Gemini
or OpenAI for realistic retrieval behavior.

## Current Limitations

- JSON storage loads all project chunks into memory.
- Chunks are line-based and do not understand code symbols.
- Re-ingestion replaces the whole project index.
- API-backed reranking is not implemented yet.
- LangChain integration is deferred to a later phase.
- There is no API server or web UI yet.
