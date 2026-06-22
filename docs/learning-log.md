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

## Phase 2 - Retrieval Quality

### Evaluation Foundation

Phase 2 started by building a better scoreboard before changing retrieval. A
pass/fail metric only reveals whether an expected source appeared somewhere in
the result limit. It does not distinguish a relevant source at rank 1 from one
at rank 3.

The evaluation module now calculates:

- first relevant rank for each question
- reciprocal rank for each question
- Hit@K and aggregate Hit Rate@K
- mean reciprocal rank across an evaluation run

The metric functions are provider-independent and deterministic. They operate
on ranked file paths and expected file paths, so they can compare semantic,
keyword, hybrid, reranked, and LangChain retrieval using the same contract.

### OpenAI Baseline

The unchanged Phase 1 semantic retriever produced this baseline on the six
HttpGo questions at top-k 3:

```text
Summary: 4/6 passed
Hit Rate@3: 0.6667
MRR: 0.5000
```

MRR added information that pass/fail did not: two successful questions found
their first expected file at rank 2 rather than rank 1. The failures also made
the next retrieval problems concrete: repeated README chunks, repeated chunks
from one file, and test files outranking implementation files.

The next experiments can now be accepted or rejected using measured changes
rather than intuition.

### Overlap Deduplication Experiment

The first improvement overfetched semantic candidates before removing exact
chunk IDs and overlapping line ranges. Overfetching matters because filtering
only the original top 3 could leave fewer than three final results.

Two overlap thresholds were evaluated:

```text
Threshold 0.50: Hit Rate@3 0.6667, MRR 0.5000
Threshold 0.25: Hit Rate@3 0.6667, MRR 0.5000
```

Neither improved the `4/6` baseline. This was still useful evidence. Adjacent
20-line chunks share 5 lines, giving an overlap ratio of 0.25, but the repeated
README results were generally separate sections rather than overlapping chunks.

The distinction is important: overlap deduplication asks whether two chunks
repeat the same source lines, while source diversity asks whether one file is
consuming too many result positions. The next experiment targets source
diversity rather than weakening the overlap threshold further.

### Source Diversity Experiment

Source diversity limits how many chunks from one file may occupy the final
ranking. Unlike overlap deduplication, it can promote another file even when
the original same-file chunks contain completely different source lines.

Results:

```text
Max per file 1: 5/6, Hit Rate@3 0.8333, MRR 0.5833
Max per file 2: 5/6, Hit Rate@3 0.8333, MRR 0.5556
```

The limit of one performed best and recovered the `httpbin-proxy-streaming`
case. This demonstrates the tradeoff: semantic score alone can over-allocate a
small context budget to one source, while a diversity constraint increases
evidence coverage.

The remaining `chunked-response` failure suggests that post-processing alone is
not enough. BM25 keyword retrieval is the next experiment because exact code
terms and implementation paths may be underweighted by embeddings.

### Manual BM25 Experiment

BM25 was implemented manually to make its inputs visible:

- term frequency describes one term inside one chunk
- document length describes the current chunk
- document frequency, average length, and total documents describe the corpus
- `k1` controls term-frequency saturation
- `b` controls document-length normalization

The tokenizer preserves exact identifiers such as `RequestFromReader` and also
adds components such as `request`, `from`, and `reader`. File paths are indexed
with chunk content so source names can participate in retrieval.

Keyword-only evaluation produced:

```text
Summary: 2/6 passed
Hit Rate@3: 0.3333
MRR: 0.1667
```

BM25 performed worse than semantic retrieval on the full natural-language
questions. Exact matching alone was not sufficient because generic query terms
could accumulate across README and test chunks, and identifiers appearing in
many tests were less rare than expected.

The lesson is not that BM25 is unusable. Its raw scores answer a different
question from cosine similarity. Hybrid retrieval should combine their ranking
positions, not add incomparable score scales directly.

### Hybrid RRF Experiment

Reciprocal Rank Fusion combines result positions using
`1 / (rank_constant + rank)`. It avoids adding cosine similarity and BM25
scores, whose numeric scales have unrelated meanings.

The hybrid pipeline produced:

```text
Summary: 6/6 passed
Hit Rate@3: 1.0000
MRR: 0.6667
```

This is the strongest result so far. BM25 passed only `2/6` by itself, yet its
ranking evidence helped hybrid retrieval recover the final failed question.
A component does not need to be the best standalone system to add useful,
independent signal to an ensemble.

Hit Rate@3 is now perfect on the current set, but MRR shows that relevant
sources are not consistently first. The next retrieval-quality problem is
second-stage ordering, which can be explored with reranking.
