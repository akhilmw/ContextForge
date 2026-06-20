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
