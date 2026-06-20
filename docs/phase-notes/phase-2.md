# Phase 2 - Retrieval Quality and LangChain

## Goal

Improve ContextForge's retrieval quality, then rebuild the RAG pipeline with
LangChain and compare it with the manual Phase 1 implementation.

Phase 1 made every RAG step visible. Phase 2 should preserve that knowledge:
LangChain is introduced as another implementation behind the same application
boundaries, not as a replacement for understanding retrieval.

The main success criterion is measurable retrieval quality. A framework change
by itself does not count as an improvement.

## Starting Baseline

Record the existing Phase 1 results before changing retrieval:

- Gemini embeddings: 6/6 retrieval questions passed at top-k 3.
- OpenAI embeddings: 4/6 retrieval questions passed at top-k 3.
- Retrieval uses semantic similarity only.
- Chunks are fixed line ranges with overlap.
- Results may contain overlapping chunks or several chunks from one file.
- Evaluation currently reports only per-question pass or fail.

These results are the baseline. Every retrieval experiment should be compared
against the same questions, provider, project index, and top-k value.

Measured OpenAI baseline:

```text
Summary: 4/6 passed
Hit Rate@3: 0.6667
MRR: 0.5000
```

## Implementation Progress

Completed:

- Phase 2 folder and file scaffolding
- Reproducible six-question OpenAI baseline
- First relevant rank and reciprocal-rank metrics
- Hit@K and Hit Rate@K metrics
- Mean reciprocal rank
- Per-question and aggregate evaluation result models
- Metric output in the existing retrieval evaluator

Next:

- Build the retrieval comparison runner
- Reduce duplicate and overlapping results

## Phase Boundaries

Phase 2 includes:

- A larger and more descriptive retrieval evaluation set
- Hit rate, hit rate at k, reciprocal rank, and mean reciprocal rank
- Retrieval result deduplication and source diversity
- Neighbor expansion for adjacent source chunks
- Keyword retrieval alongside semantic retrieval
- Hybrid score fusion
- Second-stage reranking of a broader candidate set
- A LangChain RAG implementation using existing provider boundaries where
  practical
- Side-by-side manual and LangChain evaluation
- Comparison notes covering behavior, complexity, and tradeoffs
- Unit tests using deterministic fakes

Phase 2 does not include:

- PostgreSQL or pgvector
- LangGraph agents
- MCP servers
- GitHub, database, or log ingestion
- Background jobs
- A web interface
- Replacing the local JSON project store
- Production authentication, deployment, or observability

FastAPI is deferred until the retrieval and LangChain boundaries are stable.
Otherwise the API would expose contracts that are still changing.

## Target Architecture

### Ingestion pipeline

The Phase 1 ingestion pipeline remains the source of stored chunks:

```text
Repository
  -> discovery
  -> loading
  -> line chunking
  -> embeddings
  -> local JSON project index
```

Phase 2 may add metadata required by improved retrieval, but it should not
rewrite working ingestion components without an evaluation-backed reason.

### Manual retrieval pipeline

```text
Question
  -> semantic candidates
  -> keyword candidates
  -> score normalization and fusion
  -> second-stage reranking
  -> duplicate removal and source diversification
  -> optional neighboring chunks
  -> ranked SearchResults
```

Each stage should be independently testable. Candidate generation finds likely
evidence; ranking combines signals; result processing controls redundancy and
context coverage.

### LangChain ask pipeline

```text
Question
  -> LangChain Documents and retriever
  -> prompt template
  -> chat model
  -> parsed answer
  -> ContextForge Answer and Source models
```

LangChain-specific objects must remain inside `langchain_rag.py` and
`langchain_adapters.py`. Scripts and callers should receive ContextForge domain
models rather than framework-specific objects.

### Evaluation pipeline

```text
Phase 2 question set
  -> selected retrieval implementation
  -> ranked results
  -> expected-source comparison
  -> per-question rank metrics
  -> aggregate comparison report
```

Generation evaluation is not part of this phase. Retrieval should be measured
separately so an LLM answer cannot hide missing evidence.

## Folder and File Structure

Existing Phase 1 files stay in place. Phase 2 adds the following files:

```text
ContextForge/
  docs/
    phase-notes/
      phase-2.md                 # Scope, architecture, and learning plan
    phase-2-comparison.md        # Results and manual-vs-LangChain findings

  evals/
    phase_2_questions.json       # Expanded retrieval evaluation cases

  scripts/
    compare_retrievers.py        # Run implementations against one eval set

  src/contextforge/
    evaluation.py                # Metric calculation and report models
    keyword_retriever.py         # Keyword candidate generation
    ranking.py                   # Fusion, deduplication, and diversity logic
    reranker.py                  # Second-stage candidate reranking
    context_expander.py          # Add neighboring chunks to selected evidence
    langchain_adapters.py        # Domain model and LangChain conversions
    langchain_rag.py             # LangChain ingestion/retrieval/ask composition

  tests/
    test_evaluation.py
    test_keyword_retriever.py
    test_ranking.py
    test_reranker.py
    test_context_expander.py
    test_langchain_adapters.py
    test_langchain_rag.py
    test_compare_retrievers_script.py
```

Phase 1 modules that Phase 2 will reuse or extend:

```text
src/contextforge/models.py       # Domain models remain framework-independent
src/contextforge/retriever.py    # Existing semantic retrieval baseline
src/contextforge/store.py        # Existing local project index
src/contextforge/embedder.py     # Existing embedding provider protocol
src/contextforge/prompt_builder.py
src/contextforge/llm.py
src/contextforge/ask.py
scripts/eval_retrieval.py        # Existing baseline eval runner
```

Do not create a separate `phase2` package. Retrieval concepts belong in the
main `contextforge` package and should remain usable after Phase 2 ends.

## Core Contracts

The exact types should be designed during implementation, but these behavioral
contracts should guide the modules:

- A retriever accepts a project, question, and result limit, then returns
  ranked `SearchResult` objects.
- Ranking logic accepts scored candidates and does not access files or call an
  embedding API.
- A reranker accepts a question and candidate results, then returns the same
  candidates in a new deterministic order without retrieving additional data.
- Context expansion accepts selected results and stored chunks, then returns
  ordered evidence without duplicate chunk IDs.
- Evaluation accepts ranked source paths and expected source paths, then
  returns deterministic metrics.
- LangChain adapters perform conversion only; they do not retrieve or generate.
- The LangChain pipeline returns existing `Answer` and `Source` models.

## Implementation Order

### Step 1 - Freeze and describe the baseline

Files:

- `evals/phase_2_questions.json`
- `docs/phase-2-comparison.md`

Tasks:

- Copy the useful Phase 1 questions into the Phase 2 schema.
- Add questions that expose the observed failures: function bodies split across
  chunks, README dominance, test-file dominance, and repeated files.
- Record embedding provider, top-k, and index configuration with each run.
- Run the unchanged Phase 1 retriever and record its results before edits.

Checkpoint:

- The baseline can be reproduced from one command.
- Each question has at least one explicit expected source path.

### Step 2 - Build evaluation metrics

File:

- `src/contextforge/evaluation.py`

Tasks:

- Represent one evaluation case and one case result.
- Calculate whether an expected source appears in the results.
- Calculate the rank of the first relevant source.
- Calculate reciprocal rank and aggregate mean reciprocal rank.
- Report hit rate at configurable k values.
- Keep metric calculation independent from providers and network calls.

Tests:

- `tests/test_evaluation.py`

Checkpoint:

- Hand-written ranked paths produce predictable metrics.
- Missing expected files produce rank `None` and reciprocal rank `0`.

### Step 3 - Add the comparison runner

File:

- `scripts/compare_retrievers.py`

Tasks:

- Load one evaluation file.
- Run a named retrieval implementation with the same cases and top-k.
- Print per-question rank and aggregate hit-rate/MRR results.
- Make failures return a useful process exit code.
- Avoid live API calls in script unit tests.

Tests:

- `tests/test_compare_retrievers_script.py`

Checkpoint:

- Manual semantic retrieval can be recorded as the first reproducible baseline.

### Step 4 - Reduce redundant results

File:

- `src/contextforge/ranking.py`

Tasks:

- Remove duplicate chunk IDs.
- Detect strongly overlapping line ranges from the same file.
- Add a configurable maximum number of initial results per file.
- Preserve deterministic ordering when scores tie.
- Measure changes rather than assuming greater diversity is always better.

Tests:

- `tests/test_ranking.py`

Checkpoint:

- Repeated overlapping chunks do not consume the full context budget.
- Relevant same-file chunks are not removed merely because paths match.

### Step 5 - Add neighboring context

File:

- `src/contextforge/context_expander.py`

Tasks:

- Find the previous and next stored chunks from the same file.
- Expand only after initial ranking so neighbors do not compete as search hits.
- Deduplicate neighbors already selected by retrieval.
- Preserve source order for prompt readability.
- Enforce a context budget.

Tests:

- `tests/test_context_expander.py`

Checkpoint:

- A hit at the start of a function can include its adjacent continuation.
- Expansion never crosses into another file.

### Step 6 - Add keyword retrieval

File:

- `src/contextforge/keyword_retriever.py`

Tasks:

- Tokenize questions and chunks deterministically.
- Score exact identifiers, file names, and content terms.
- Avoid introducing a search service or database in this phase.
- Return the same `SearchResult` domain model as semantic retrieval.
- Handle empty questions and indexes explicitly.

Tests:

- `tests/test_keyword_retriever.py`

Checkpoint:

- Exact code identifiers can be found even when semantic retrieval misses them.

### Step 7 - Build hybrid ranking

File:

- `src/contextforge/ranking.py`

Tasks:

- Combine semantic and keyword candidate rankings.
- Start with reciprocal rank fusion because provider score scales differ.
- Make fusion weights and result limits explicit configuration.
- Apply deduplication and diversity after fusion.
- Compare semantic-only, keyword-only, and hybrid metrics.

Checkpoint:

- Hybrid retrieval is kept only if it improves the evaluation or solves a
  documented failure without unacceptable regressions.

### Step 8 - Add second-stage reranking

File:

- `src/contextforge/reranker.py`

Tasks:

- Define a small reranker protocol that is independent of any provider.
- Retrieve a broader candidate set, such as 15-20 chunks, before reranking.
- Return only the final top-k results after reranking.
- Begin with a deterministic fake or heuristic reranker for contract tests.
- Optionally add one provider-based reranker after the local contract works.
- Preserve chunk metadata and prevent a reranker from introducing new chunks.
- Apply deduplication and source diversity after reranking.
- Apply neighboring-context expansion only after final results are selected.
- Compare retrieval quality, latency, and API cost with reranking disabled and
  enabled.

Tests:

- `tests/test_reranker.py`

Checkpoint:

- The reranker can change candidate order without changing candidate identity.
- Invalid provider responses, duplicate results, and missing candidates fail
  clearly.
- Reranking is kept only when its measured benefit justifies added latency and
  cost.

### Step 9 - Introduce LangChain adapters

File:

- `src/contextforge/langchain_adapters.py`

Tasks:

- Convert ContextForge `Document` and `Chunk` models to LangChain documents.
- Preserve chunk IDs, project names, paths, languages, and line ranges as
  metadata.
- Convert retrieved LangChain documents back into `SearchResult` objects.
- Raise clear errors when required metadata is missing.

Tests:

- `tests/test_langchain_adapters.py`

Checkpoint:

- A domain object survives a conversion round trip without losing citations.

### Step 10 - Build the LangChain RAG path

File:

- `src/contextforge/langchain_rag.py`

Tasks:

- Compose loading, splitting, embeddings, retrieval, prompting, and generation
  using focused LangChain components.
- Keep provider selection outside the core chain.
- Return ContextForge domain models at the public boundary.
- Keep the manual Phase 1 path operational for comparison.
- Do not add LangGraph; this is a deterministic pipeline, not an agent.

Tests:

- `tests/test_langchain_rag.py`

Checkpoint:

- The LangChain path answers through fakes without network access.
- Citations retain exact file and line metadata.

### Step 11 - Compare and document

File:

- `docs/phase-2-comparison.md`

Tasks:

- Run manual semantic, manual hybrid, and LangChain retrieval consistently.
- Record hit rate at k, MRR, latency observations, and failure examples.
- Compare code visibility, control, testability, dependency cost, and extension
  effort.
- Document which implementation becomes the default and why.
- Update `README.md`, `docs/architecture.md`, and `docs/learning-log.md`.

Checkpoint:

- The final choice is supported by measurements and concrete tradeoffs.

## Dependency Strategy

Do not install all LangChain integrations at the start. Add dependencies only
when the relevant implementation step begins.

Expected minimum packages should be verified against current LangChain
documentation at that time. Likely categories are:

- LangChain core abstractions
- The provider integration actually used in the experiment
- A reranker integration only if the provider experiment is retained
- A local/in-memory vector-store integration only if required

Pin compatible versions through `uv`, and commit `uv.lock` changes with the
step that first uses them.

## Test Strategy

- Unit tests use deterministic chunks, vectors, rankings, and fake models.
- Existing Phase 1 tests must continue passing.
- Live provider tests remain optional integration tests.
- Retrieval experiments must use the same evaluation cases and settings.
- Each bug found in evaluation should become a small regression test when its
  expected behavior is deterministic.

## Learning Order

Implement one small contract at a time:

1. Learn why a metric is needed before coding it.
2. Write examples by hand and predict their output.
3. Implement the smallest function that satisfies the contract.
4. Add edge-case tests.
5. Run the retrieval evaluation.
6. Record what changed and why.

For LangChain, map every abstraction back to its manual Phase 1 equivalent.
For example, a LangChain retriever should be understood in terms of query
embedding, candidate scoring, top-k selection, and returned metadata.

## Definition of Done

Phase 2 is complete when:

- The Phase 1 baseline is recorded and reproducible.
- Evaluation reports hit rate at k and MRR, not only pass/fail.
- Redundant retrieval results are handled predictably.
- Neighbor expansion can recover adjacent source context.
- Keyword and hybrid retrieval are evaluated against semantic retrieval.
- Second-stage reranking is evaluated for quality, latency, and cost.
- A LangChain RAG path works without removing the manual path.
- Both paths return ContextForge answers with file and line citations.
- Core behavior has deterministic unit tests.
- Manual and LangChain approaches are compared using the same evaluation set.
- Architecture, README, comparison notes, and learning log are updated.

## Expected Learning Outcomes

By the end of Phase 2, you should be able to explain:

- Why hit rate and MRR reveal different retrieval behavior.
- Why semantic retrieval can miss exact identifiers.
- Why repeated high-scoring chunks can reduce answer quality.
- How neighbor expansion differs from retrieving more chunks.
- How hybrid retrieval combines different relevance signals.
- How first-stage retrieval differs from second-stage reranking.
- Why rerankers usually process a shortlist rather than the entire index.
- Which LangChain abstractions correspond to the manual Phase 1 modules.
- What LangChain simplifies and what control it hides.
- Why evaluation should drive architecture choices in a RAG system.
