# Phase 2 Retrieval Comparison

This document will record the Phase 1 baseline and each Phase 2 retrieval
experiment. Results should be added only after running the same evaluation set
with documented provider, index, and top-k settings.

## Baseline - Manual Semantic Retrieval

Recorded on June 19, 2026 using the unchanged Phase 1 cosine-similarity
retriever.

Configuration:

- Repository: `/Users/akhilnair/Desktop/HttpGo`
- Project index: `http-go-openai`
- Embedder: OpenAI
- Retrieval limit: top 3
- Evaluation cases: 6

Results:

```text
Summary: 4/6 passed
Hit Rate@3: 0.6667
MRR: 0.5000
```

The four successful cases had first relevant ranks `[1, 1, 2, 2]`. The two
failed cases had no relevant source in the top three results.

### Failed Cases

```text
FAIL chunked-response
  expected any: ['internal/response/response.go']
  got: ['httpserver/main.go', 'README.md', 'internal/response/response_test.go']

FAIL httpbin-proxy-streaming
  expected any: ['httpserver/main.go', 'internal/response/response.go']
  got: ['README.md', 'README.md', 'README.md']
```

### Baseline Observations

- Repeated chunks from one file can consume multiple top-k positions.
- README content can outrank implementation files for broad architectural
  questions.
- Test files can rank above the implementation because they contain direct
  behavioral language.
- A 4/6 pass count alone hides ranking quality. MRR shows that two successful
  cases still placed their first relevant source at rank 2.

These observations define the retrieval problems Phase 2 experiments should
address. Future results must use the same six questions, OpenAI index, and top-k
value unless the changed configuration is explicitly documented.

## Experiment 1 - Overlap Deduplication

The first retrieval experiment overfetched 15 semantic candidates, removed
exact chunk IDs and highly overlapping same-file line ranges, then returned the
final top 3 results.

Results:

| Strategy | Candidate K | Threshold | Hit Rate@3 | MRR |
|---|---:|---:|---:|---:|
| Semantic baseline | 3 | N/A | 0.6667 | 0.5000 |
| Deduplicated | 15 | 0.50 | 0.6667 | 0.5000 |
| Deduplicated | 15 | 0.25 | 0.6667 | 0.5000 |

Both thresholds produced the same `4/6` result as the baseline. The project is
chunked into 20-line windows with 5 lines of overlap, so adjacent full chunks
have an overlap ratio of `5/20 = 0.25`. Testing at `0.25` therefore exercised
the actual configured chunk overlap boundary.

The repeated README results were mostly different, non-overlapping sections.
Overlap deduplication correctly retained them, which means this failure is not
an exact-duplicate or line-overlap problem. The next experiment should limit
how many initial results one file can occupy while still allowing a configurable
number of useful same-file chunks.

## Experiment 2 - Source Diversity

The second experiment kept the same 15 semantic candidates and 0.25 overlap
threshold, then limited how many chunks from one file could occupy the final
ranking.

Results:

| Strategy | Max per File | Hit Rate@3 | MRR |
|---|---:|---:|---:|
| Semantic baseline | Unlimited | 0.6667 | 0.5000 |
| Diverse | 1 | **0.8333** | **0.5833** |
| Diverse | 2 | 0.8333 | 0.5556 |

Both limits improved the result from `4/6` to `5/6`. Limiting each file to one
result recovered `httpbin-proxy-streaming` by promoting `httpserver/main.go`
into the top three. It also produced the stronger MRR because relevant sources
appeared earlier on average.

`chunked-response` still failed. Source diversity solved one ranking-capacity
problem, but it cannot make a weakly ranked exact implementation file relevant.
The next experiment will add BM25 keyword retrieval before hybrid rank fusion.

## Experiment 3 - BM25 Keyword Retrieval

The third experiment implemented BM25 manually and evaluated it as a standalone
retriever. Chunk paths and content were tokenized together, preserving complete
code identifiers while also splitting snake_case and CamelCase components.

Configuration:

- BM25 `k1`: 1.5
- BM25 `b`: 0.75
- Retrieval limit: top 3
- Embedding API calls: none

Results:

| Strategy | Hit Rate@3 | MRR |
|---|---:|---:|
| Semantic baseline | 0.6667 | 0.5000 |
| Diverse semantic | **0.8333** | **0.5833** |
| BM25 keyword | 0.3333 | 0.1667 |

Keyword-only retrieval passed `2/6` questions. For the partial-read question,
the `RequestFromReader` implementation chunk ranked 14 even though it contained
the exact identifier. The identifier also appears in tests, reducing its IDF,
while generic query terms accumulated score in other chunks.

This result does not justify discarding BM25. It shows that BM25 and semantic
retrieval solve different ranking problems. The next experiment will combine
their rank positions with Reciprocal Rank Fusion instead of adding their
incompatible raw scores.

## Experiment 4 - Hybrid Reciprocal Rank Fusion

The fourth experiment retrieved 15 semantic and 15 BM25 candidates, combined
their rank positions with Reciprocal Rank Fusion, then applied overlap removal
and one-result-per-file source diversity before selecting the final top 3.

Configuration:

- Candidate count per retriever: 15
- RRF rank constant: 60
- Overlap threshold: 0.25
- Maximum results per file: 1
- BM25 `k1`: 1.5
- BM25 `b`: 0.75

Results:

| Strategy | Hit Rate@3 | MRR |
|---|---:|---:|
| Semantic baseline | 0.6667 | 0.5000 |
| Diverse semantic | 0.8333 | 0.5833 |
| BM25 keyword | 0.3333 | 0.1667 |
| Hybrid RRF | **1.0000** | **0.6667** |

Hybrid retrieval passed all `6/6` questions. It recovered `chunked-response`
with `internal/response/response.go` at rank 1. BM25 was weak alone, but its
independent ranking evidence changed the fused ordering enough to solve the
remaining semantic failure.

MRR remains below 1.0 because four successful questions placed their first
expected source at rank 2. The next quality step should target ordering within
the fused shortlist rather than adding more first-stage candidate generators.

## Experiment 5 - Hybrid With Heuristic Reranking

The fifth experiment added a local second-stage reranker after RRF and before
overlap removal, source diversity, and final top-k selection. The reranker does
not call an external API. It reorders the fused candidate set using lightweight
signals:

- question-term coverage in chunk content
- question-term matches in the file path
- a small preference for implementation files over tests and docs
- original fused rank as a tie-breaker

Comparison command:

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

Results:

| Strategy | Hits | Hit Rate@3 | MRR |
|---|---:|---:|---:|
| Semantic baseline | 4/6 | 0.6667 | 0.5000 |
| Diverse semantic | 5/6 | 0.8333 | 0.5833 |
| BM25 keyword | 2/6 | 0.3333 | 0.1667 |
| Hybrid RRF | 6/6 | 1.0000 | 0.6667 |
| Hybrid + heuristic reranker | 6/6 | 1.0000 | 0.9167 |

The reranker did not change the pass count because hybrid retrieval already
found relevant files for all six questions. It improved MRR by moving relevant
files earlier in the top-three list. That is the intended job of reranking:
improve ordering quality after candidate generation has already found useful
evidence.
