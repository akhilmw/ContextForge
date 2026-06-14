# ContextForge — Project Master Plan

## Project Goal

Build **ContextForge**, an MCP-powered developer operations agent.

The main goal is **learning**, not just shipping. The project should teach RAG, LangChain, LangGraph, MCP servers, LangSmith, vector databases, tool-calling, agent orchestration, and production-style AI backend design through one serious flagship project.

This should be built in small milestones so each phase teaches one major concept clearly. The final project should be resume-worthy, but every phase should also produce learning notes that can be shared publicly, such as on LinkedIn.

---

## One-Line Description

ContextForge is an MCP-powered developer operations agent that helps engineers investigate backend issues by combining codebase RAG, GitHub analysis, database inspection, log search, and LangGraph-based tool orchestration.

---

## What Problem It Solves

When backend engineers debug or understand a system, useful context is scattered across many places:

* Code is in a local repo or GitHub.
* Architecture notes are in docs.
* Runtime errors are in logs.
* Database structure is in Postgres.
* Recent changes are in commits and PRs.
* Past decisions are in README/design docs/runbooks.

ContextForge helps answer questions like:

* “How does this part of the codebase work?”
* “Why might this endpoint be failing?”
* “What changed recently that could have caused this issue?”
* “Which files, logs, docs, or database tables are relevant?”
* “Can you generate a root-cause style explanation with sources?”

The project is essentially a mini **AI SRE / backend debugging assistant**.

---

## What the Project Is NOT

This is not just:

* A PDF chatbot.
* A generic ChatGPT clone.
* A random LangChain demo.
* A model-training project.
* A UI-first project.

This is a backend/AI systems project focused on:

* RAG over real engineering context.
* Agentic workflows.
* MCP tool servers.
* Safe tool execution.
* Tracing and evaluation.
* Production-style backend design.

---

## Final Mental Model

ChatGPT alone:

> “I can guess based on what you paste.”

ContextForge:

> “I can search your codebase, inspect docs, check logs, query safe database metadata, look at commits, and then give a grounded answer.”

---

## Final Architecture

```text
User
 |
CLI / FastAPI API / Optional UI
 |
LangGraph Agent Orchestrator
 |
MCP Client Layer
 |
 ------------------------------------------------
 |              |              |                |
Docs/RAG MCP   GitHub MCP     Postgres MCP     Logs MCP
Server         Server         Server           Server
 |
PostgreSQL + pgvector
```

---

## Core Technologies

Use these technologies across phases:

* Python
* FastAPI
* PostgreSQL
* pgvector
* LangChain
* LangGraph
* MCP
* LangSmith
* Docker / Docker Compose
* Optional: Go for later backend integration
* Optional: React/Next.js for final UI demo

---

## Final Repo Structure

```text
contextforge/
  README.md
  docs/
    architecture.md
    learning-log.md
    phase-notes/
      phase-0.md
      phase-1.md
      phase-2.md
      phase-3.md
      phase-4.md
      phase-5.md
      phase-6.md
      phase-7.md
      phase-8.md

  apps/
    api/
      main.py
    cli/
      main.py

  services/
    rag-service/
      ingest_repo.py
      ingest_docs.py
      ingest_logs.py
      chunker.py
      embedder.py
      retriever.py
      ask.py

    mcp-docs-server/
      server.py

    mcp-github-server/
      server.py

    mcp-postgres-server/
      server.py

    mcp-logs-server/
      server.py

  infra/
    docker-compose.yml

  evals/
    questions.json

  sample-data/
    docs/
    logs/
```

---

## Input Sources

The project should support multiple input types over time.

### 1. Codebase Input

User provides a local repo path:

```bash
contextforge ingest repo --path ../http-go --name http-go
```

The system scans source files such as:

* `.go`
* `.py`
* `.md`
* `.yaml`
* `.yml`
* `.json`
* `.sql`
* `.txt`

Ignore files/folders like:

* `.git/`
* `node_modules/`
* `venv/`
* `__pycache__/`
* `dist/`
* `build/`
* `.env`

### 2. Documentation Input

User provides a docs folder:

```bash
contextforge ingest docs --path ./docs --name http-go-docs
```

Docs may include:

* `README.md`
* `architecture.md`
* `design-doc.md`
* `learning-notes.md`
* `runbook.md`

### 3. Logs Input

For learning, use sample/generated logs first.

```bash
contextforge ingest logs --path ./sample-data/logs --service http-go
```

Example logs:

```text
2026-06-13T10:14:22Z ERROR service=http-go route=/httpbin/stream/3 request_id=req-123 upstream timeout after 5000ms
2026-06-13T10:14:24Z WARN service=http-go route=/httpbin/stream/3 request_id=req-124 client closed connection during chunked response
```

### 4. Database Input

Later phase only. Use a Postgres connection string and expose safe readonly database tools.

### 5. GitHub Input

Later phase only. Use repo URL/token to analyze commits, PRs, changed files, and issues.

---

## CLI Vision

The CLI should eventually support:

```bash
contextforge init

contextforge ingest repo --path ../http-go --name http-go

contextforge ingest docs --path ../http-go/docs --name http-go-docs

contextforge ingest logs --path ./sample-data/logs --service http-go

contextforge ask "How does RequestFromReader handle partial reads?"

contextforge ask "Why might /httpbin/stream/3 fail under high concurrency?"
```

---

## API Vision

FastAPI endpoints can eventually include:

```http
POST /ingest/repo
POST /ingest/docs
POST /ingest/logs
POST /ask
GET /sources/{chunk_id}
```

Example `/ask` request:

```json
{
  "question": "How does chunked transfer encoding work in this server?",
  "repo_name": "http-go"
}
```

Example response:

```json
{
  "answer": "The server writes chunked responses by...",
  "sources": [
    {
      "file_path": "internal/response/writer.go",
      "score": 0.84
    },
    {
      "file_path": "README.md",
      "score": 0.77
    }
  ]
}
```

---

# Development Phases

## Phase 0 — Project Setup and Architecture

### Goal

Set up the project structure and define the architecture clearly.

### Learn

* What is RAG?
* What is an agent?
* What is MCP?
* What is LangChain?
* What is LangGraph?
* What is LangSmith?
* How do all of these fit together?

### Build

* Repo structure.
* README.md.
* `docs/architecture.md`.
* `docs/learning-log.md`.
* Docker Compose skeleton.

### Deliverables

* Project folder structure.
* Initial README.
* Architecture document.
* Learning log file.

### LinkedIn Learning Theme

“I’m starting a learning-first project to understand RAG, MCP, LangGraph, and AI agents by building a developer operations assistant.”

---

## Phase 1 — Manual RAG From Scratch

### Goal

Before using LangChain, build RAG manually once.

### Why

This is important because the goal is to understand what LangChain later abstracts away.

### Learn

* Documents.
* Chunking.
* Embeddings.
* Vector similarity.
* Top-k retrieval.
* Context stuffing.
* Prompt construction.
* Grounded answers.
* Source citations.

### Build

A simple Python service/script that can:

1. Load files from a local codebase.
2. Chunk code and markdown files.
3. Create embeddings.
4. Store chunks locally at first.
5. Retrieve top-k chunks for a question.
6. Send retrieved context to an LLM.
7. Return an answer with source files.

### Initial Target Codebase

Use the existing Go HTTP server project as the first codebase.

Example question:

```text
How does RequestFromReader handle partial reads?
```

Expected behavior:

The system should retrieve the relevant parser/request code and tests, then explain the answer based on actual files.

### Deliverables

* `chunker.py`
* `embedder.py`
* `retriever.py`
* `ingest_repo.py`
* `ask.py`
* 5 sample questions.
* Answers with source file references.
* Learning notes.

### Week 1 Learning Outcomes

By the end of this phase, be able to explain:

* What RAG is.
* Why chunking matters.
* What embeddings are.
* How vector similarity search works.
* What top-k retrieval means.
* Why source grounding matters.
* Why bad retrieval causes hallucination.

### LinkedIn Learning Theme

“Built RAG from scratch before using frameworks.”

Key learning:

> RAG is not magic. It is documents → chunks → embeddings → similarity search → prompt with retrieved context.

---

## Phase 2 — LangChain RAG Pipeline

### Goal

Rebuild the manual RAG pipeline using LangChain.

### Learn

* Document loaders.
* Text splitters.
* Retrievers.
* Prompt templates.
* Chains.
* Output parsers.
* Vector store integrations.

### Build

Convert manual RAG into a LangChain-based flow:

```text
loader -> splitter -> embeddings -> vector store -> retriever -> prompt -> LLM -> answer
```

### Deliverables

* `langchain_rag.py`
* Clean FastAPI endpoint: `/ask`
* Comparison notes: manual RAG vs LangChain RAG.

### LinkedIn Learning Theme

“Rebuilt my manual RAG pipeline using LangChain.”

Key learning:

> Manual RAG helped me understand the internals. LangChain helped standardize the pieces.

---

## Phase 3 — PostgreSQL + pgvector Retrieval

### Goal

Move from local/in-memory vector search to PostgreSQL + pgvector.

### Learn

* PostgreSQL pgvector.
* Embedding storage.
* Metadata filtering.
* Top-k search.
* Hybrid search basics.
* Retrieval quality.
* Chunk size tradeoffs.

### Build

Create a Postgres schema for chunks.

Example fields:

```text
chunk_id
repo_name
source_type
file_path
language
content
embedding
created_at
```

Support metadata filters:

```text
repo_name = http-go
file_type = .go
folder = internal/request
source_type = code/docs/logs
```

### Deliverables

* Docker Compose with Postgres + pgvector.
* Database schema.
* Ingestion script writing to Postgres.
* Retrieval API.
* Source-aware answers.
* Basic evaluation set of 10 questions.

### Example Eval Questions

1. How are headers parsed?
2. How does the server detect end of headers?
3. What happens for invalid request lines?
4. How is chunked transfer encoding handled?
5. How does the handler route requests?
6. How does the server handle connection close?
7. Where is the request body parsed?
8. What tests exist for malformed headers?
9. What does the response writer do?
10. How are upstream proxy responses handled?

### LinkedIn Learning Theme

“Moved my RAG system from local search to PostgreSQL + pgvector.”

Key learning:

> Retrieval quality depends heavily on chunking, metadata, and top-k selection, not just the LLM.

---

## Phase 4 — LangGraph Orchestration

### Goal

Turn the RAG app into a stateful agent workflow.

### Learn

* State.
* Nodes.
* Edges.
* Conditional routing.
* Tool calls.
* Fallbacks.
* Retries.
* Agent loops.

### Build

Create a LangGraph workflow:

```text
User question
  -> classify intent
  -> route to code RAG / docs RAG / logs search / general answer
  -> retrieve context
  -> check confidence
  -> answer with sources
```

Example graph:

```text
START
  |
classify_question
  |
  |-- code_question -> retrieve_code_context
  |-- docs_question -> retrieve_docs_context
  |-- debugging_question -> retrieve_code + retrieve_logs
  |
generate_answer
  |
END
```

### Deliverables

* LangGraph workflow.
* State schema.
* Routing logic.
* Confidence check.
* Fallback answer when context is weak.

### LinkedIn Learning Theme

“Added LangGraph to make my RAG app stateful and agentic.”

Key learning:

> A chain is usually linear. A graph lets the agent branch, retry, and make decisions based on state.

---

## Phase 5 — First Custom MCP Server

### Goal

Build the first custom MCP server.

Start with the Docs/RAG MCP Server.

### Learn

* MCP server.
* MCP client.
* MCP tools.
* MCP resources.
* MCP prompts.
* Tool schemas.
* Standardized AI tool access.

### Build

Create an MCP server that exposes the RAG system as tools.

Example tools:

```text
docs.search(query, top_k)
docs.get_chunk(chunk_id)
docs.list_repos()
docs.explain_file(file_path)
```

Instead of the agent directly calling a local RAG function, it should call the MCP tool.

### Deliverables

* `mcp-docs-server`
* Tool definitions.
* MCP client integration.
* LangGraph node that calls `docs.search`.

### LinkedIn Learning Theme

“Built my first custom MCP server.”

Key learning:

> MCP is not an agent framework. It is a standard way to expose external capabilities as tools/resources that agents can use.

---

## Phase 6 — Multiple MCP Servers

### Goal

Make the system modular with multiple MCP servers.

### Learn

* Multi-tool agent design.
* Tool routing.
* Permission boundaries.
* Tool failures.
* Timeouts.
* Audit logs.
* Service separation.

### Build

Add more MCP servers.

### GitHub MCP Server

Tools:

```text
github.get_recent_commits
github.get_pr_diff
github.get_changed_files
github.search_issues
```

### Postgres MCP Server

Tools:

```text
postgres.list_tables
postgres.describe_schema
postgres.safe_query
postgres.explain_query
```

Important: keep Postgres tools readonly by default.

Safety requirements:

* Block `DELETE`.
* Block `UPDATE`.
* Block `INSERT`.
* Block `DROP`.
* Add query timeout.
* Add row limit.
* Log every query.

### Logs MCP Server

Tools:

```text
logs.search_errors
logs.get_recent_logs
logs.find_by_request_id
```

### Deliverables

* 3–4 MCP servers.
* Tool registry.
* LangGraph tool router.
* Audit log table.
* Timeout/retry handling.

### LinkedIn Learning Theme

“Moved from one MCP server to a multi-server agent system.”

Key learning:

> The hard part was not calling tools. The hard part was deciding which tool should be called, how to handle failures, and how to keep tool execution safe.

---

## Phase 7 — LangSmith Tracing and Evaluation

### Goal

Add observability for the agent.

### Learn

* LLM tracing.
* Tool call traces.
* Latency tracking.
* Prompt debugging.
* Retrieval evaluation.
* Agent debugging.
* Dataset-based evals.

### Build

Track:

* Which LangGraph node ran.
* Which MCP tool was called.
* Tool input/output.
* Retrieved chunks.
* LLM prompt.
* LLM answer.
* Latency.
* Errors.
* Retries.

Create an eval dataset:

```text
20 questions
expected source files
expected answer traits
```

### Deliverables

* LangSmith traces.
* Eval dataset.
* Before/after prompt comparison.
* Latency report.
* Retrieval quality notes.

### LinkedIn Learning Theme

“Added LangSmith tracing to debug my agent workflow.”

Key learning:

> Agent observability matters because without traces, it is hard to know whether the model failed, retrieval failed, or the wrong tool was called.

---

## Phase 8 — Final Demo, README, Resume, LinkedIn Wrap-Up

### Goal

Make the project presentable.

### Build Final Demo Scenarios

1. Ask a codebase question.
2. Debug a fake incident.
3. Analyze a PR.
4. Inspect a database schema.
5. Generate a root-cause report.

### Optional UI

Build a simple UI only at the end.

Suggested UI layout:

```text
Left side:
- Repo selector
- Source type filter: code/docs/logs/db

Main area:
- Chat box
- Agent answer

Right side:
- Sources used
- MCP tools called
- Trace steps
```

### Best Demo GIF

Show this flow:

```text
Question:
"Why might /httpbin/stream/3 fail under high concurrency?"

Agent steps:
1. Searching codebase...
2. Found handler in handlers/proxy.go
3. Searching logs...
4. Found upstream timeout + client closed connection
5. Searching docs...
6. Found chunked response notes
7. Generated RCA
```

Final answer should show:

```text
Likely issue:
The streaming proxy path depends on upstream response streaming and chunked transfer encoding. Logs show upstream timeouts and client disconnects. The handler should ensure upstream body closure, timeout propagation, and connection-close behavior are handled correctly.

Sources:
- handlers/proxy.go
- internal/response/writer.go
- logs/http-go-errors.log
```

### Deliverables

* Final README.
* Architecture diagram.
* Demo GIF.
* Sample traces.
* Sample questions.
* Resume bullets.
* LinkedIn launch post.

### Final README Sections

* What is ContextForge?
* Why I built it.
* Architecture.
* Tech stack.
* MCP servers.
* LangGraph workflow.
* RAG pipeline.
* LangSmith tracing.
* Demo.
* Learning notes.
* Future work.

---

# Weekly Schedule

```text
Week 1: Phase 0 + Phase 1
Output: Manual RAG over your Go server

Week 2: Phase 2
Output: LangChain RAG API

Week 3: Phase 3
Output: pgvector retrieval

Week 4: Phase 4
Output: LangGraph workflow

Week 5: Phase 5
Output: First MCP server

Week 6: Phase 6
Output: Multiple MCP servers

Week 7: Phase 7
Output: LangSmith tracing/evals

Week 8: Phase 8
Output: README, demo, resume bullets, LinkedIn wrap-up
```

---

# First Immediate Task

Start with Phase 0 and Phase 1.

Do not build UI yet.

Do not add MCP yet.

Do not add LangGraph yet.

First build manual RAG over a local codebase.

## Phase 0 Immediate Deliverables

Create:

```text
README.md
docs/architecture.md
docs/learning-log.md
infra/docker-compose.yml
services/rag-service/
```

## Phase 1 Immediate Deliverables

Create:

```text
services/rag-service/chunker.py
services/rag-service/embedder.py
services/rag-service/retriever.py
services/rag-service/ingest_repo.py
services/rag-service/ask.py
```

Initial CLI usage should look like:

```bash
python services/rag-service/ingest_repo.py --path ../http-go --name http-go

python services/rag-service/ask.py "How does RequestFromReader handle partial reads?"
```

For Phase 1, it is acceptable to use local JSON files for storage before moving to Postgres in Phase 3.

---

# Important Engineering Principles

Follow these principles throughout the project:

1. Learning-first. Prefer clear code over clever abstractions.
2. Build manually first, then introduce frameworks.
3. Every phase should have notes explaining what was learned.
4. Keep every answer grounded in source files/logs/docs.
5. Always return sources with answers.
6. Add observability and evals before calling the project finished.
7. Treat MCP tools as production-style boundaries.
8. For database tools, default to readonly and safe behavior.
9. Build CLI/API first; UI comes last.
10. Keep the final resume story grounded and explainable.

---

# Final Resume Direction

Final project can be described like this:

**ContextForge | Python, FastAPI, PostgreSQL, pgvector, MCP, LangGraph, LangSmith, Docker**

* Built an MCP-powered developer operations agent that connected code search, documentation, GitHub, PostgreSQL, and logs through modular tool servers.
* Implemented a LangGraph orchestration layer that routed user requests across RAG retrieval, database inspection, GitHub analysis, and log investigation workflows.
* Integrated pgvector-backed semantic search and LangSmith tracing to monitor retrieval quality, tool calls, latency, retries, and agent execution paths.

---

# Codex Instructions

When implementing this project, proceed phase by phase.

Do not jump ahead.

For each phase:

1. Create/modify only the files needed for that phase.
2. Keep the code simple and readable. (Also before writing any code, ask me!)
3. Add comments where concepts are important for learning.
4. Update `docs/learning-log.md` with what was learned.
5. Add example commands to the README.
6. Include sample outputs wherever possible.
7. Avoid overengineering early phases.
8. Preserve the learning progression:

   * Manual RAG first.
   * LangChain second.
   * pgvector third.
   * LangGraph fourth.
   * MCP fifth.
   * LangSmith later.
   * UI last.