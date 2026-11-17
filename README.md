# AI-Powered Knowledge Assistant

A conversational, context-aware **Retrieval-Augmented Generation (RAG)** platform that answers questions from
enterprise documents with cited, verified responses.

| | |
|---|---|
| **Language** | Python 3.10 – 3.13 (3.11 / 3.12 recommended) |
| **Core stack** | LangChain · LangGraph · OpenAI APIs · Pinecone · SQL (SQLite) |
| **Interfaces** | REST API (FastAPI) · Chat UI (Streamlit) · CLI · MCP server |
| **Quality** | Automated AI testing / assurance suite with a golden evaluation set |

---

## Table of contents
1. [Features](#features)
2. [How it works](#how-it-works)
3. [Project structure](#project-structure)
4. [Quick start](#quick-start)
5. [Using the application](#using-the-application)
6. [Operating modes](#operating-modes)
7. [Testing and AI assurance](#testing-and-ai-assurance)
8. [Documentation index](#documentation-index)
9. [Known limitations](#known-limitations)
10. [Roadmap](#roadmap)

---

## Features

- **RAG architecture**: LLM answers are generated *only* from retrieved enterprise document chunks, with inline `[n]` citations.
- **Vector retrieval**: semantic search over Pinecone (or a persistent in-memory store for offline use), with a relevance threshold.
- **LangGraph workflows**: multi-step reasoning with intent routing, history-aware query rewriting, answer validation, a retry loop and explicit state management.
- **Conversational memory**: per-session history, so follow-ups such as "what about the API limit?" resolve correctly.
- **Document processing pipeline**: PDF / TXT / Markdown loading, chunking, embedding generation, idempotent upserts, and cleanup of stale vectors when files change.
- **Hallucination guard**: every answer is fact-checked against its context. Unverified answers are retried with wider retrieval, and then flagged.
- **SQL persistence**: document registry and a full interaction audit log (question, answer, route, groundedness, sources, trace, latency).
- **MCP support**: the knowledge base is exposed as MCP tools for Claude Desktop, IDEs and other agents.
- **AI testing / assurance**: tests for retrieval quality, response relevance, workflow behaviour and failure paths.
- **Runs offline**: mock LLM + local embeddings + in-memory store means no API keys are needed to try everything.

## How it works

```
 Documents ──▶ load ─▶ chunk ─▶ embed ─▶ upsert ──▶ Pinecone / Memory store
                                    │
                                    └── registry (SQLite): skip unchanged, replace changed

 Question ─▶ classify ─┬─ chitchat ──────────────────────────────┐
                       └─ rewrite ─▶ retrieve ─┬─ generate ─▶ validate ─┬─▶ finalize ─▶ answer + sources
                                      ▲        └─ no_answer ────────────┤
                                      └────────── retry (wider k) ◀─────┘  (not grounded)
```

Full details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Project structure

```
ai-knowledge-assistant/
├── app/
│   ├── config.py               # Settings dataclass, env loading, validation
│   ├── db.py                   # SQLite: document registry + interaction log
│   ├── embeddings.py           # OpenAI embeddings / local hashing embeddings
│   ├── llm.py                  # ChatOpenAI / deterministic mock LLM
│   ├── prompts.py              # Prompt templates for every workflow step
│   ├── vectorstore.py          # Pinecone (SDK) + in-memory vector stores
│   ├── ingestion/pipeline.py   # load → chunk → embed → upsert
│   ├── graph/
│   │   ├── state.py            # LangGraph state schema
│   │   └── workflow.py         # Nodes, routing and graph assembly
│   ├── assistant.py            # KnowledgeAssistant facade (ask / ingest)
│   ├── evaluation.py           # Retrieval / answer quality metrics
│   ├── api.py                  # FastAPI application
│   └── mcp_server.py           # MCP tools
├── scripts/                    # ingest.py · chat_cli.py · evaluate.py
├── ui/streamlit_app.py         # Chat UI
├── data/documents/             # Sample knowledge base (+ uploads/)
├── tests/                      # pytest suite + golden/golden_qa.json
├── docs/                       # Detailed documentation
├── requirements.txt · .env.example · Dockerfile · docker-compose.yml · pytest.ini
```

## Quick start

### Prerequisites
- Python **3.10 – 3.13** (3.14 is not yet supported by all dependencies)
- (Optional, production mode) an OpenAI API key and a Pinecone API key

### 1. Install

**Windows (PowerShell)**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

**macOS / Linux**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. Verify
```bash
pytest                      # 20 tests should pass
python scripts/evaluate.py  # prints metrics and PASSED
```

### 3. Run
```bash
uvicorn app.api:app --reload        # API   → http://127.0.0.1:8000  (redirects to /docs)
streamlit run ui/streamlit_app.py   # UI    → http://localhost:8501   (second terminal)
python scripts/chat_cli.py          # terminal chat
```

The default `.env` runs in **offline demo mode** and needs no API keys.

## Using the application

**Swagger UI** at `http://127.0.0.1:8000/docs`, or:

```bash
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
  -d '{"question": "How many days of annual leave do employees get?", "session_id": "demo"}'
```

Add your own documents:
```bash
python scripts/ingest.py path/to/folder_or_file     # CLI
# or POST /ingest (multipart) / the Streamlit sidebar uploader
```

Supported types: `.pdf`, `.txt`, `.md`. Re-ingesting is safe: unchanged files are skipped and changed files replace their old vectors.

### MCP
```bash
python -m app.mcp_server
```
Claude Desktop configuration:
```json
{"mcpServers": {"knowledge-assistant": {
  "command": "C:/path/to/.venv/Scripts/python.exe",
  "args": ["-m", "app.mcp_server"],
  "cwd": "C:/path/to/ai-knowledge-assistant"}}}
```
Tools: `search_knowledge_base(query, top_k)` and `ask_knowledge_assistant(question, session_id)`.

## Operating modes

| Mode | LLM | Embeddings | Vector DB | Use for |
|---|---|---|---|---|
| **Offline** (default) | mock | local hashing | in-memory (JSON file) | demos, CI, development |
| **Production** | OpenAI | OpenAI | Pinecone | real answer quality |

Switch in `.env`:
```
LLM_BACKEND=openai
EMBEDDING_BACKEND=openai
VECTOR_BACKEND=pinecone
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk_...
MIN_RELEVANCE=0.25
```
The Pinecone serverless index (cosine, 1536 dims) is created automatically. See [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

> **Note:** the offline mock LLM returns the top retrieved passage; it demonstrates the pipeline but not real answer quality.

## Testing and AI assurance

```bash
pytest                       # full suite
pytest tests/test_workflow.py -v
python scripts/evaluate.py   # writes reports/eval_report.json
```

| Area | What is verified |
|---|---|
| Ingestion | chunk metadata, idempotency, changed-file replacement, empty/unsupported files |
| Retrieval quality | hit-rate on golden questions; out-of-scope queries fall below the relevance threshold |
| Workflow behaviour | chitchat skips retrieval, full RAG path order, no-answer path, retry loop, disclaimer after exhausted retries, per-session memory, SQL logging |
| Response quality | keyword coverage, groundedness rate, refusal accuracy |
| API | health, chat, validation errors, upload + query, unsupported uploads, root redirect |

Add your own questions to `tests/golden/golden_qa.json` (set `"expected_source": null` for questions that should be refused). Details: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

## Documentation index

| Document | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Components, workflow nodes, state, data model, design decisions |
| [docs/API.md](docs/API.md) | Every endpoint with request / response examples and errors |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | All environment variables, modes, tuning guidance |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Dev setup, testing, evaluation, extending the system |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Docker, Compose, systemd, cloud, production hardening |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Startup and runtime problems with fixes |

## Known limitations

- The API has **no authentication** yet. Add it before exposing it to a network (see deployment doc).
- Conversation memory uses LangGraph's in-process `MemorySaver`; it is lost on restart and not shared between workers.
- The in-memory vector store is for development only and is not shared between processes.
- Documents are keyed by **file name**; two files with the same name in different folders collide.
- Scanned (image-only) PDFs yield no text because there is no OCR step.

## Roadmap

Auth and rate limiting, a persistent checkpointer (Postgres), streaming responses, OCR and DOCX loaders,
hybrid (keyword + vector) search with reranking, per-user document access control.
"# ai-knowledge-assistant" 
