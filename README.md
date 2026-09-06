# Fieldguide AI

Fieldguide AI is an experimental local knowledge assistant that turns Markdown corpora and structured CSV data into searchable, model-assisted workflows.

> **Status:** Active prototype. Markdown ingestion now uses vectorstore-ai with a Postgres catalog and optional Chroma, NumPy, or FAISS vectors. Chat still reads legacy indexes; connecting chat to the shared retriever is a later phase. A separate dataframe-tool agent is also available.

## Quickstart

### Prerequisites

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key for embeddings and OpenAI chat; an Anthropic API key for Anthropic chat
- Postgres for corpus indexing; chunk preview and plain chat do not need a database

Install the project from the repository root:

```bash
uv sync
```

The lockfile resolves `model-runtime` from the `v0.2.1` Git tag and `vectorstore-ai[chroma,faiss,postgres]` from `v0.1.0`. Chroma, FAISS, and the Postgres driver are installed through the `vectorstore-ai` extras.

`uv sync` also installs the default `dev` dependency group, including Ruff, mypy, and pytest. Use `uv sync --no-dev` for a runtime-only environment. Development tools use a dependency group rather than an optional package extra, so the orchestrator's `uv sync --locked` retains the tools required by its checks.

If you do not already have a local `.env` file, copy the example:

```bash
cp .env.example .env
```

Fill in the provider keys you need:

```dotenv
OPENAI_API_KEY=your-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

`OPENAI_API_KEY` is required for embeddings and OpenAI chat. Set `ANTHROPIC_API_KEY` when using Anthropic chat. Do not commit `.env` or real credentials.

Start the guided CLI:

```bash
uv run fieldguide
```

The styled wizard prompts for an LLM provider and model, an optional vector store, a system prompt, and optional Markdown ingestion. Choose `none (plain chat)` to chat without retrieval. Selecting ingestion writes the shared catalog/index and ends the wizard after reporting counts. When ingestion is declined, chat can retrieve from an existing legacy index and display its document and section sources.

### Local Postgres for the retrieval migration

Both CLI ingestion entry points use the shared pipeline and read `POSTGRES_CONNECTIONSTRING`. All indexing modes require a catalog, including dense mode. Catalog-backed chat and CLI retrieval-mode selection remain future work. Offline tests substitute SQLite for Postgres.

With Docker and the Docker Compose plugin installed, start the database and wait for its health check:

```bash
docker compose up -d --wait postgres
```

The service runs Postgres 17, listens on `127.0.0.1:5432`, and persists data in the `postgres_data` named volume. The development database, user, and password default to `fieldguide`. `.env.example` supplies the matching `POSTGRES_CONNECTIONSTRING` for vectorstore-ai's document catalog. If you change `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, or `POSTGRES_PORT`, update the connection string to match. Database credentials are applied when the volume is first initialized.

Explicitly initialize the catalog before the first ingestion:

```bash
uv run python main.py --init-catalog
```

This action creates the catalog schema without constructing chat or embedding clients. It can be repeated and is separate from `--index-corpus` and `--chunk-corpus`; normal ingestion never creates the schema.

Stop the service while keeping its data:

```bash
docker compose down
```

## What it does

- Provides a rich menu-driven CLI with startup and mid-chat configuration.
- Loads Markdown files with optional frontmatter and splits them into section-aware, overlapping chunks.
- Indexes source documents and chunks in Postgres, with optional OpenAI vectors persisted through vectorstore-ai in Chroma or NumPy/FAISS directories.
- Retrieves the nearest chunks for each question, keeps injected context out of visible history, and displays the sources used.
- Maintains stateful OpenAI or Anthropic chat history with commands for inspection, reset, and live reconfiguration.
- Includes a separate LangChain dataframe agent with explicit tools for inspecting, searching, filtering, and aggregating CSV datasets.

## Problem and target user

Operational knowledge often lives across runbooks, policies, product documentation, FAQs, and tabular exports. Finding the relevant source and turning it into a defensible answer becomes slow and inconsistent as the corpus grows.

- **Primary user:** A forward-deployed engineer, support operator, or operations specialist working with customer and company knowledge.
- **Job to be done:** Ingest local knowledge, inspect or search it consistently, and use a model-assisted interface to answer follow-up questions.
- **Secondary user:** A developer extending provider, vector-store, ingestion, or dataframe-tool behavior.

## Why model-assisted behavior

Loading, chunking, validation, persistence, and dataframe operations are deterministic and remain ordinary Python pipelines. A model is useful where the user asks open-ended questions, conversational context must be maintained, or the appropriate dataframe tool and arguments must be selected dynamically.

The project does not treat every step as agentic. Deterministic work stays explicit and testable; model-driven work is isolated behind provider and tool interfaces. The shared `model-runtime` package owns asynchronous OpenAI and Anthropic chat calls, routing, retries, timeouts, usage accounting, conversation state, generation telemetry, and the guarded synchronous bridge used by Fieldguide's CLI. Credentials, vector stores, and filesystem loading enter through explicit composition boundaries so core conversation and indexing behavior can be tested without network or disk access.

## Usage

### Interactive wizard

```bash
uv run fieldguide
```

Available chat commands:

| Command | Effect |
| --- | --- |
| `:help` | Show all available commands. |
| `:config` | Show the current provider, model, vector store, top-k, and system prompt. |
| `:provider` | Select a new provider and model while preserving conversation history. |
| `:model` | Select another current-provider model while preserving history. |
| `:store` | Change the vector store or switch to plain chat. |
| `:system` | Change the system prompt. |
| `:history` | Print the current system, user, and assistant messages. |
| `:clear` | Clear conversation turn history; the configured system prompt is kept. |
| `:quit`, `:q`, `:exit` | Exit the chat session. |

Pressing Ctrl+C, Ctrl+D, or cancelling a wizard question exits cleanly.

For wizard ingestion, initialize Postgres first and enter a fresh directory as the vector-store path. The path defaults still describe legacy chat storage (including `numpy_index.npz`); shared NumPy/FAISS ingestion uses directories. The wizard ends after ingestion because its chat path has not yet migrated to the shared retriever.

### Flag-based CLI

The original argparse interface remains available through `main.py`.

Inspect all options:

```bash
uv run python main.py --help
```

Preview Markdown chunks without credentials, a database, or model calls:

```bash
uv run python main.py --chunk-corpus path/to/corpus --chunk-limit 5
```

Add `--chunk-details` to emit full `CatalogChunk` JSON, including `chunk_id`, `doc_id`, `text`, `chunk_index`, `section_path`, `content_hash`, and `active`. The compact view also shows document type and source path. Preview and ingestion use the same shared adapter and section chunker; `--chunk-max-words` must be greater than the default 75-word overlap.

After setting `POSTGRES_CONNECTIONSTRING` and running `--init-catalog`, index a corpus in Chroma:

```bash
uv run python main.py \
  --index-corpus path/to/corpus \
  --vector-store chroma \
  --store-path data/retrieval/chroma
```

Use the lightweight NumPy store instead:

```bash
uv run python main.py \
  --index-corpus path/to/corpus \
  --vector-store numpy \
  --store-path data/retrieval/numpy
```

Use FAISS for a compact, high-performance local cosine index:

```bash
uv run python main.py \
  --index-corpus path/to/corpus \
  --vector-store faiss \
  --store-path data/retrieval/faiss
```

`--index-corpus` uses dense ingestion and respects `--vector-store`, `--store-path`, `--collection-name`, and `--embedding-model`. With no path, indexing defaults to `chroma_db`, `numpy_index`, or `faiss_index`. `--vector-store none` remains a plain-chat option and is rejected for indexing. Lexical-only ingestion is available through the Python API below.

Shared indexes require their matching catalog rows and embedding state. Keep each catalog/embedding space paired with its vector store. Use a separate catalog database when trying different backends with the same embedding model; otherwise the ledger can skip unchanged chunks whose vectors exist only in the previous store. Existing legacy indexes must be re-indexed into the shared pipeline and kept separate from the new stores.

Start retrieval-grounded chat with an **existing legacy index** (new shared indexes are consumed by the Python retrieval API, pending the chat migration):

```bash
uv run python main.py \
  --model gpt-5-nano \
  --vector-store faiss \
  --store-path path/to/existing-legacy-faiss \
  --top-k 5
```

To bypass retrieval in the flag CLI:

```bash
uv run python main.py --vector-store none
```

### Markdown input

Frontmatter is optional. The shared Markdown adapter retains top-level scalar metadata on catalog documents and dense vector records; nested maps and lists are not retained as filterable attributes. `CatalogChunk` carries text, identity, section, and lifecycle fields separately. Directory ingestion discovers `**/*.md` in stable path order; single `.md` and `.markdown` files are supported. Without frontmatter, document IDs use file stems, so stems must be unique within an ingestion batch. Directory sources are recorded relative to the corpus root.

```markdown
---
doc_id: support-runbook
title: Support Runbook
doc_type: runbook
---

# Support Runbook

## Escalation

Escalate incidents that meet the documented severity threshold.
```

### Dataframe agent experiment

`langchain_main.py` loads CSV files from `data/corpora/nautilus/misc` and exposes constrained dataframe tools to a LangChain agent:

```bash
uv run python langchain_main.py
```

This entry point expects the local CSV corpus to exist. It is separate from the Markdown vector-indexing and chat workflow.

## Architecture

```text
CLI ingestion --> retrieval.indexing --> vectorstore-ai adapter / chunker / pipeline
                                            |                    |
                                            v                    v
                                      Postgres catalog     Chroma / NumPy / FAISS

CLI chat --> KnowledgeBot --> legacy vector store --> retrieved context
                |
                +--> ChatSession --> model-runtime --> OpenAI / Anthropic

Dataframe CLI --> LangChain agent --> constrained pandas tools
```

| Component | Responsibility |
| --- | --- |
| `fieldguide_ai/config/` | Immutable, validated session settings and their default system prompt. |
| `fieldguide_ai/interactive.py` | Rich guided configuration, sourced chat, and live session commands. |
| `fieldguide_ai/knowledge_bot.py` | Optional retrieval through a focused search boundary, prompt augmentation, raw history preservation, and sourced answers. |
| `fieldguide_ai/cli.py` | Flag-based commands, indexing orchestration, and retrieval-capable chat loop. |
| `model-runtime.ChatSession` | System context, normalized `Message` history, atomic conversation turns, `GenerationRecord` telemetry, and guarded sync calls. |
| `fieldguide_ai/providers/` | Immutable provider registry, credential checks, adapter factories, single-route runtime composition, and model discovery. |
| `fieldguide_ai/ingestion/` | Legacy document models and ingestion code, retained until the legacy-stack removal phase. CLI ingestion no longer uses its pipeline. |
| `fieldguide_ai/vectorstore/` | Legacy search/index interfaces, embedding abstraction, and persistence still used by chat. |
| `fieldguide_ai/retrieval/` | Shared Markdown preview/indexing, factories for vectorstore-ai embedders, stores, catalogs, and retrievers, plus source records and diagnostics. Chat integration remains pending. |
| `fieldguide_ai/errors.py` | Application-level configuration, embedding, document-loading, and vector-store exceptions. Model failures use `ModelRuntimeError`. |
| `langchain_pandas/` | Validated dataframe catalog with defensive snapshots and constrained inspection/query tools. |

### Shared retrieval composition API

`fieldguide_ai.retrieval` exports frozen `RetrievalSettings`, `RetrievalMode`, and factories for the shared library. Dense mode enables vector search, lexical mode enables catalog full-text search without constructing an embedder or vector store, and hybrid mode combines both through reciprocal rank fusion. `build_retriever(settings, top_k)` limits the final result count; it also accepts caller-owned `catalog` and `embedder` dependencies.

Set `RetrievalSettings.catalog_dsn` explicitly, for example from `os.environ["POSTGRES_CONNECTIONSTRING"]`. Lexical and hybrid settings require a DSN. Dense settings can omit it when a catalog is injected, but the default `build_catalog(settings)` requires a Postgres DSN in every mode: dense hits also need catalog rows for chunk text. Schema creation is opt-in through `build_catalog(settings, initialize_schema=True)`; ordinary retriever construction does not create tables. `build_embedder(settings)` reads `OPENAI_API_KEY` unless an explicit `api_key` is supplied.

The settings default to dense mode, Chroma at `chroma_db`, collection `documents`, and `text-embedding-3-small`. When choosing NumPy or FAISS, supply a dedicated directory as `store_path`, or `None` for an in-memory store. `build_store(settings, dimension)` loads an existing directory and rejects invalid indexes or incompatible vector dimensions; `persist_store(store, settings)` saves NumPy/FAISS after writes, while Chroma persists automatically. These shared-library directory formats differ from legacy chat's NumPy file and FAISS file prefix. Re-index legacy corpora using `index_corpus` to populate the required catalog rows and shared store.

`index_corpus(path, settings, max_words)` returns the shared `IngestionResult`, including document/chunk counts and per-space embedded/skipped counts. Dense and hybrid ingestion both populate the catalog and selected vector store. Lexical mode populates only the catalog and never constructs an embedder, router, or store:

```python
import os

from dotenv import load_dotenv

from fieldguide_ai.retrieval import RetrievalMode, RetrievalSettings, index_corpus

load_dotenv()
settings = RetrievalSettings(
    mode=RetrievalMode.LEXICAL,
    store_type=None,
    store_path=None,
    embedding_model=None,
    catalog_dsn=os.environ["POSTGRES_CONNECTIONSTRING"],
)
# Initialize once with --init-catalog before this call.
result = index_corpus("data/corpora/nautilus/raw", settings, max_words=900)
print(result.document_count, result.chunk_count)
```

`chunk_corpus(path, max_words)` returns source `Record` objects and `CatalogChunk` objects without creating database or model clients. The CLI uses it for both compact and JSON previews.

`sources_from_result(result, catalog)` performs one document lookup and preserves result order, text, scores, and dense/lexical ranks. Missing document metadata leaves `source` and `title` unset. `RetrievalDiagnostics.from_result(result, mode)` retains the requested mode, degradation state, errors, selected provider, and per-stage timings.

## Observable workflows

### Markdown indexing

1. Read Markdown through `MarkdownSourceAdapter`, including optional scalar frontmatter.
2. Split each record with vectorstore-ai's `MarkdownSectionChunker` and enforce the word limit with overlap.
3. Upsert catalog documents and replace their chunk sets through `IngestionPipeline`, pruning superseded vectors in the configured store.
4. In dense/hybrid mode, route embedding calls for missing or stale chunks to the configured provider. Unchanged chunks with current embedding state are skipped. Lexical mode performs no embedding calls.
5. Save NumPy/FAISS stores after successful ingestion; Chroma persists its writes automatically.
6. Return ingestion counts and render document/chunk totals in the CLI.

The catalog is written before vectors. Catalog and vector persistence are separate operations, so a partial failure can leave them inconsistent; keep the database and store together when backing up or recovering an index. Ingestion replaces chunks for supplied documents but does not remove documents merely because their source files were deleted from a directory.

### Stateful chat

1. Set the configured system prompt on the provider, separate from turn history.
2. Embed the question and retrieve the configured number of nearest chunks when a vector store is enabled.
3. Send accumulated turns plus an augmented copy of the current question and retrieved context to the provider. Only the original question is stored in visible history.
4. Receive a model-runtime `GenerationRecord`; only after generation succeeds, atomically append the original user question and normalized assistant message to history, then display the retrieved document/section sources.
5. Allow the user to inspect, clear, or reconfigure the session. Provider and model changes preserve turn history where possible; clearing keeps the system prompt and generation log.

This describes observable state and tool flow; it does not expose hidden model reasoning.

## Configuration

| Setting | Default | Notes |
| --- | --- | --- |
| LLM provider | `openai` | OpenAI and Anthropic are registered. |
| Chat model | `gpt-5-nano` | The wizard also offers `gpt-5-mini`, `gpt-4o-mini`, and a custom model name. |
| Embedding model | `text-embedding-3-small` | Configurable through `--embedding-model` in the flag-based CLI. |
| Vector store | `chroma` | `numpy`, `faiss`, and retrieval-free `none` are available. |
| Chroma path | `chroma_db` for flag-based indexing | Legacy chat and wizard default to `data/chroma_db`; use a fresh directory for shared ingestion. |
| Chroma collection | `documents` | Configurable in both CLI flows. |
| NumPy path | `numpy_index` for flag-based indexing | Shared index directory; legacy chat/wizard default to the file `numpy_index.npz`. |
| FAISS path | `faiss_index` | Shared ingestion directory; legacy chat interprets it as a `.faiss`/`.json` file prefix. Keep the indexes separate. |
| Retrieval count | `5` | Configurable with `--top-k`; shown in the interactive summary. |
| Maximum chunk size | `900` words | Must exceed the shared chunker's 75-word overlap. |
| `OPENAI_API_KEY` | None | Required before constructing OpenAI chat or embedding clients. |
| `ANTHROPIC_API_KEY` | None | Required before constructing an Anthropic chat client. |
| `POSTGRES_CONNECTIONSTRING` | None | Required by both CLI ingestion paths and `--init-catalog`. Python callers pass it into `RetrievalSettings.catalog_dsn`. |

## Testing and quality checks

The commands below are for manual validation. Codex follows [AGENTS.md](AGENTS.md)
and leaves post-modification formatting, linting, type checking, and test execution
to `codex-orchestrator` unless the user explicitly asks Codex to run them. During
orchestrated plan execution, Codex implements the assigned phase and repairs
reported phase regressions. The orchestrator first validates the starting checkout;
baseline failures stop before implementation and are handled as separate maintenance.
After each turn, it runs `setup_commands` (the locked environment sync), then all
independent checks in `commands`, collecting failures into one repair report. A
prerequisite failure stops without spending a phase repair turn. Codex reports a
blocker when a repair requires unrelated or explicitly excluded work.
Outside an orchestrated run, the user owns check execution by default.

Orchestrated implementation defaults to `gpt-6-astra` with `xhigh` reasoning;
validation repairs resume the same phase session with `gpt-5.6-luna` and `medium`
reasoning. Override these through the shell environment, independently of model
settings in `.codex/config.toml`:

```bash
export CODEX_ORCHESTRATOR_MODEL=gpt-6-astra
export CODEX_ORCHESTRATOR_REASONING_EFFORT=xhigh
export CODEX_ORCHESTRATOR_REPAIR_MODEL=gpt-5.6-luna
export CODEX_ORCHESTRATOR_REPAIR_REASONING_EFFORT=medium
```

The orchestrator reads exported variables, not `.env` files. Resume retains saved
model selections unless an exported variable overrides them. Sandbox and approval
settings continue to come from Codex configuration. The terminal and each turn's
`invocation.json` record the requested model and reasoning; validation reports list
all passed, failed, and skipped commands with log paths. Older resumed runs that
lack a historical baseline explicitly report it as unavailable.

Run the unit test suite:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest discover -s tests
```

Run the locked lint and formatting tools:

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
```

Sync the default development group and run the configured static type checker:

```bash
uv sync --locked
uv run --no-sync mypy
```

Mypy checks `fieldguide_ai/`, `langchain_pandas/`, `tests/`, `main.py`, and `langchain_main.py` with strict checking and explicit override enforcement. `ty check` uses the same source set. These targets match the current repository layout and include the retrieval composition layer. The default development group includes `pandas-stubs` for checking dataframe code against pandas' typed API.

Tests cover integration with model-runtime messages and generation records, adapter discovery and dependency injection, immutable interactive configuration, KnowledgeBot context/history behavior, Markdown parsing and chunking, Chroma/NumPy/FAISS persistence and search, metadata serialization, and dataframe tools. The model-runtime repository owns the detailed atomic-session and sync-bridge contract tests.

Retrieval composition tests cover mode and settings validation, credential/schema wiring, NumPy/FAISS directory persistence, Chroma configuration, lexical retrieval without embedding calls, source hydration, and diagnostics. Ingestion tests use a recording embedder and temporary SQLite catalog to cover preview parity, catalog-only writes, dense/hybrid persistence, skipped unchanged embeddings, replacement/pruning, and failure propagation. CLI tests cover catalog initialization and ingestion configuration. These tests do not need OpenAI credentials or a running Postgres instance. End-to-end tests joining migrated ingestion to chat remain future work.

## Guardrails and data handling

- CLI entry points read credentials from the environment and inject them into provider adapter factories; credentials must not be committed.
- Ingestion persists Markdown text and metadata in the configured Postgres catalog. Dense/hybrid ingestion sends chunk text to OpenAI for embedding and persists text, metadata, and vectors in the selected local store; lexical ingestion makes no embedding calls.
- The dataframe agent is instructed to use its registered tools and avoid answering from general knowledge, but model output should still be treated as untrusted until independently verified.
- Retrieved local chunk text is sent to the configured model provider as prompt context.
- Each in-process `ChatSession` retains visible normalized messages and generation records,
  including raw provider responses, until that session is discarded. Clearing visible history
  does not clear its generation log.
- The primary chat loop has no external action tools or approval workflow.
- There is no production authentication, authorization, audit log, or tenant isolation layer.

## Limitations and tradeoffs

- Dense/hybrid embeddings currently depend on OpenAI, including when Anthropic is selected for chat. Lexical-only ingestion through the Python API does not require OpenAI.
- Chat still uses legacy indexes; wizard ingestion ends after indexing, and shared catalog-backed chat is pending migration.
- Markdown is the only supported ingestion format in the indexing pipeline.
- The NumPy store is intended for small local indexes; it loads records into memory for search.
- The dataframe demo depends on a fixed local CSV directory and is not integrated with the primary CLI.
- Formal quality, latency, cost, groundedness, and human-intervention benchmarks have not been established.
- Production observability and model-output validation are not yet implemented.

## Evaluation strategy

Current automated tests evaluate deterministic correctness: parsing, chunk boundaries, replacement semantics, vector dimensions and ranking, persistence, CLI flow, provider messages, and dataframe tool behavior.

Before production use, add a versioned evaluation set and report at least:

- Retrieval recall and ranking quality
- Grounded answer correctness and citation coverage
- End-to-end latency and model cost
- Tool-selection and argument accuracy
- Failure and human-escalation rates

Do not interpret passing unit tests as evidence of answer quality or production readiness.

## Project structure

```text
.
├── .cursor/rules/          # Project rules, including README maintenance
├── AGENTS.md               # Codex instructions and validation ownership
├── .env.example            # Provider keys and local Postgres configuration
├── docker-compose.yml      # Optional local Postgres for the retrieval migration
├── data/corpora/nautilus/  # Synthetic ITSM Markdown corpus plus CSV demos
│   ├── source_of_truth/    # facts.yaml consistency contract (payments stack, KIs, routing)
│   ├── manifests/          # document_manifest_v0.yaml (history) and document_manifest_v2.yaml
│   ├── raw/                # Ingestible Markdown: incidents, changes, runbooks, SOPs, etc.
│   └── misc/               # CSV files used by the dataframe agent demo
├── fieldguide_ai/
│   ├── ingestion/          # Retained legacy document models and pipeline
│   ├── config/             # Validated interactive session configuration
│   ├── providers/          # Provider metadata, adapter factories, and registry
│   ├── retrieval/          # Shared ingestion, retrieval composition, sources, diagnostics
│   ├── vectorstore/        # Legacy Chroma, NumPy, and FAISS chat stores
│   ├── knowledge_bot.py    # Retrieval-grounded chat orchestration
│   ├── errors.py           # Stable application-level error boundary
│   ├── terminal.py         # Shared terminal history rendering
│   ├── cli.py              # Flag-based CLI and chat loop
│   └── interactive.py      # Rich questionary wizard and chat loop
├── langchain_pandas/       # Dataframe agent catalog and tools
├── notebooks/              # Exploratory scripts and notebooks
│   └── explore.py          # Concurrent OpenAI/Anthropic model-runtime demo
├── tests/                  # unittest suite
├── langchain_main.py       # CSV dataframe agent entry point
├── main.py                 # Flag-based CLI entry point
└── pyproject.toml          # Package metadata and dependencies
```

To compare Fieldguide's `ChatSession` sync bridge with true `asyncio.gather` overlap across OpenAI and Anthropic network calls:

```bash
uv run python notebooks/explore.py
```


The Nautilus ITSM corpus (`data/corpora/nautilus`) is synthetic enterprise support documentation for retrieval and future agentic triage experiments. Payments Reporting is modeled as a simplified inspectable pipeline: Payments Platform feed → Azure SQL staging → Databricks transform → Azure SQL reporting → Power BI, with ordered freshness checkpoints and multiple failure-mode incident clusters. Authoritative constraints live in `source_of_truth/facts.yaml`; the final document inventory is `manifests/document_manifest_v2.yaml`. Runtime ingestion reads `raw/**/*.md` frontmatter and body content only.

## Extension points

- Register another LLM by supplying a factory for an adapter that implements model-runtime's separate `ChatModel` and `ModelCatalog` protocols, then composing a validated `ProviderSpec` into a `ProviderRegistry`.
- Add another vector backend by implementing the `VectorStore` interface and wiring it into CLI construction.
- Add ingestion formats through vectorstore-ai source adapters that produce `Record` objects.
- Add safe dataframe operations as explicit tools rather than enabling arbitrary Python execution.

## Roadmap

1. Add groundedness, retrieval, latency, and cost evaluations.
2. Add provider-neutral embedding and model configuration.
3. Add production logging, retries, and explicit approval boundaries for future actions.

## Contributing and license

Keep implementation, tests, and `README.md` synchronized in the same change.
Contributors should arrange validation before submitting changes. Codex writes
required tests and reviews changes by inspection, while `codex-orchestrator` or
the user executes checks under the policy in [AGENTS.md](AGENTS.md).

No license file is currently included. Add one before distributing or reusing the project outside its current context.
