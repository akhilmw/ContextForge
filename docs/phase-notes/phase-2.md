# Phase 2 - Retrieval Quality

## Goal

Improve ContextForge retrieval quality with measurable experiments while keeping
the manual Phase 1 RAG pipeline intact.

Phase 1 proved the end-to-end flow: ingestion, chunking, embeddings, retrieval,
prompt construction, and answering. Phase 2 focused on the retrieval layer
itself:

- better evaluation metrics
- less redundant retrieval output
- keyword retrieval alongside semantic retrieval
- hybrid rank fusion
- second-stage reranking

LangChain is intentionally deferred to Phase 3. Retrieval quality is a large
enough topic to stand on its own, and the manual implementation gives us a
clear baseline before introducing a framework.

## Starting Baseline

The initial OpenAI embedding baseline used semantic cosine-similarity retrieval
only.

Configuration:

- Eval target: HttpGo
- Eval file: `evals/phase_2_questions.json`
- Project index: `http-go-openai`
- Top K: 3
- Embedding model: OpenAI

Baseline result:

```text
Summary: 4/6 passed
Hit Rate@3: 0.6667
MRR: 0.5000
```

The baseline exposed three retrieval problems:

- repeated chunks from one file could consume multiple top-k positions
- README and test chunks sometimes outranked implementation files
- pass/fail alone hid whether relevant files appeared early or late

## Completed Work

Phase 2 added:

- `EvaluationCaseResult` and `EvaluationSummary`
- first relevant rank
- reciprocal rank
- mean reciprocal rank
- Hit@K and Hit Rate@K
- exact chunk-ID deduplication
- same-file line-overlap measurement and deduplication
- candidate overfetching before final top-k selection
- configurable per-file source diversity
- deterministic code-aware tokenization
- manual BM25 keyword retrieval
- Reciprocal Rank Fusion for semantic plus keyword retrieval
- reranker protocol and fake reranker
- local heuristic reranker
- strategy support in `scripts/eval_retrieval.py`
- compact comparison table in `scripts/compare_retrievers.py`

## Final Results

Command:

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

| Strategy | Hits | Hit Rate@3 | MRR |
|---|---:|---:|---:|
| Semantic | 4/6 | 0.6667 | 0.5000 |
| Diverse semantic | 5/6 | 0.8333 | 0.5833 |
| BM25 keyword | 2/6 | 0.3333 | 0.1667 |
| Hybrid RRF | 6/6 | 1.0000 | 0.6667 |
| Hybrid + heuristic reranker | 6/6 | 1.0000 | 0.9167 |

The final retrieval stack improved Hit Rate@3 from `0.6667` to `1.0000` and
MRR from `0.5000` to `0.9167`.

The key result is not just that the right files appeared in the top three. The
reranker moved relevant files earlier in the result list, which is why MRR
improved after hybrid retrieval had already reached `6/6`.

## Final Manual Retrieval Pipeline

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

## Folder and File Structure

Phase 2 primarily uses these files:

```text
ContextForge/
  docs/
    phase-notes/
      phase-2.md
    phase-2-comparison.md

  evals/
    phase_2_questions.json

  scripts/
    eval_retrieval.py
    compare_retrievers.py

  src/contextforge/
    evaluation.py
    keyword_retriever.py
    ranking.py
    reranker.py
    retriever.py

  tests/
    test_evaluation.py
    test_keyword_retriever.py
    test_ranking.py
    test_reranker.py
    test_retriever.py
    test_eval_retrieval_script.py
    test_compare_retrievers_script.py
```

## Core Contracts

- Evaluation accepts ranked source paths and expected source paths, then returns
  deterministic metrics.
- Ranking logic accepts scored candidates and does not access files or call an
  embedding API.
- Keyword retrieval returns the same `SearchResult` domain model as semantic
  retrieval.
- Hybrid retrieval combines rank positions rather than incompatible raw scores.
- A reranker accepts a question and candidate results, then returns the same
  candidates in a new deterministic order.
- Rerankers must not introduce, drop, or duplicate candidates.

## Experiment Summary

### 1. Evaluation Metrics

Phase 1 reported pass/fail. Phase 2 added rank-aware metrics so we can measure
whether the first relevant result appears early.

MRR became important because two strategies can both pass, while one returns
the useful source at rank 1 and another returns it at rank 3.

### 2. Overlap Deduplication

Overlap deduplication removed exact duplicate chunk IDs and highly overlapping
same-file line ranges.

This did not improve the eval score by itself. That was useful evidence: the
main failure was not exact duplication, but ranking capacity and source
dominance.

### 3. Source Diversity

Source diversity limited how many chunks from one file could occupy the final
result set.

This improved the OpenAI eval from `4/6` to `5/6` by preventing README chunks
from consuming the full top-three result budget.

### 4. BM25 Keyword Retrieval

BM25 keyword retrieval was implemented manually to understand sparse retrieval
instead of treating it as a black box.

BM25 alone was worse than semantic retrieval on this eval (`2/6`), but it
provided different ranking evidence that became useful in hybrid retrieval.

### 5. Hybrid Reciprocal Rank Fusion

Hybrid retrieval combined semantic and BM25 rankings with Reciprocal Rank
Fusion.

RRF was chosen because semantic similarity scores and BM25 scores are not on
the same scale. Rank positions are safer to combine than raw scores.

Hybrid retrieval reached `6/6`, recovering the final missed implementation
file.

### 6. Heuristic Reranking

The heuristic reranker reordered fused candidates using:

- question-term coverage in chunk content
- question-term matches in file paths
- small implementation-file preference over docs and tests
- original rank as a tie-breaker

This kept Hit Rate@3 at `1.0000` while improving MRR to `0.9167`.

## Deferred Work

The following are intentionally not part of Phase 2:

- LangChain RAG implementation
- LangGraph agents
- PostgreSQL or pgvector
- API-backed reranking
- GitHub, database, or log ingestion
- FastAPI server
- web UI
- production authentication, deployment, or observability

## Definition of Done

Phase 2 is complete because:

- the semantic baseline is recorded and reproducible
- evaluation reports Hit Rate@K and MRR
- redundant retrieval results are handled predictably
- keyword and semantic retrieval are both implemented
- hybrid retrieval is evaluated against semantic and keyword baselines
- second-stage reranking is evaluated
- results are documented in `README.md` and `docs/phase-2-comparison.md`
- deterministic tests cover the retrieval-quality modules

## Expected Learning Outcomes

By the end of Phase 2, you should be able to explain:

- why Hit Rate@K and MRR measure different retrieval behavior
- why semantic retrieval can miss exact identifiers
- why BM25 can be weak alone but useful in hybrid retrieval
- why repeated high-scoring chunks can reduce answer quality
- why RRF combines rankings instead of raw scores
- how first-stage retrieval differs from second-stage reranking
- why rerankers operate on a shortlist instead of the whole index
- why retrieval evaluation should drive RAG architecture decisions
