# Fieldguide AI

Fieldguide AI is an experimental local knowledge assistant that turns Markdown corpora and structured CSV data into searchable, model-assisted workflows.

> **Status:** Active prototype. Markdown ingestion, persistent Chroma, NumPy, and FAISS indexes, retrieval-grounded chat, and a separate dataframe-tool agent are implemented.

## Quickstart

### Prerequisites

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key for embeddings and OpenAI chat; an Anthropic API key for Anthropic chat

Install the project from the repository root:

```bash
uv sync
```

Create a local `.env` file:

```dotenv
OPENAI_API_KEY=your-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

`OPENAI_API_KEY` is required for embeddings and OpenAI chat. Set `ANTHROPIC_API_KEY` when using Anthropic chat. Do not commit `.env` or real credentials.

Start the guided CLI:

```bash
uv run fieldguide
```

The styled wizard prompts for an LLM provider and model, an optional vector store, a system prompt, and optional Markdown ingestion. During chat it retrieves relevant chunks, adds them to the model context, and displays their document and section as sources. Choose `none (plain chat)` to chat without retrieval.

## What it does

- Provides a rich menu-driven CLI with startup and mid-chat configuration.
- Loads Markdown files with optional frontmatter and splits them into section-aware, overlapping chunks.
- Generates OpenAI embeddings and persists vectors in Chroma, a local NumPy `.npz` index, or a FAISS index with a JSON metadata sidecar.
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

The project does not treat every step as agentic. Deterministic work stays explicit and testable; model-driven work is isolated behind provider and tool interfaces. Provider SDK clients, credentials, vector stores, and filesystem loading enter through explicit composition boundaries so core conversation and indexing behavior can be tested without network or disk access.

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

### Flag-based CLI

The original argparse interface remains available through `main.py`.

Inspect all options:

```bash
uv run python main.py --help
```

Preview Markdown chunks without calling an embedding or chat model:

```bash
uv run python main.py --chunk-corpus path/to/corpus --chunk-limit 5
```

Index a corpus in Chroma:

```bash
uv run python main.py \
  --index-corpus path/to/corpus \
  --vector-store chroma \
  --store-path chroma_db
```

Use the lightweight NumPy store instead:

```bash
uv run python main.py \
  --index-corpus path/to/corpus \
  --vector-store numpy \
  --store-path numpy_index.npz
```

Use FAISS for a compact, high-performance local cosine index:

```bash
uv run python main.py \
  --index-corpus path/to/corpus \
  --vector-store faiss \
  --store-path faiss_index
```

Start retrieval-grounded chat directly (the selected persisted store is opened for each question):

```bash
uv run python main.py \
  --model gpt-5-nano \
  --vector-store faiss \
  --store-path faiss_index \
  --top-k 5
```

To bypass retrieval in the flag CLI:

```bash
uv run python main.py --vector-store none
```

### Markdown input

Frontmatter is optional. When present, metadata is retained on generated chunks.

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
User
  |
  +-- rich/questionary wizard ---> provider registry --> OpenAI / Anthropic
  |          |
  |          +--> Markdown ingestion and retrieval
  |                    |
  |                    +--> loader --> section chunker --> embeddings
  |                                                        |
  |                                                        +--> Chroma / NumPy / FAISS
  |                                                                  |
  |          +<---------------- answer + source chunks <--------------+
  |
  +-- dataframe CLI -----------> LangChain agent --> constrained pandas tools
```

| Component | Responsibility |
| --- | --- |
| `fieldguide_ai/interactive.py` | Rich guided configuration, immutable validated session settings, sourced chat, and live session commands. |
| `fieldguide_ai/knowledge_bot.py` | Optional retrieval through a focused search boundary, prompt augmentation, raw history preservation, and sourced answers. |
| `fieldguide_ai/cli.py` | Flag-based commands, indexing orchestration, and retrieval-capable chat loop. |
| `fieldguide_ai/chat/` | Provider-agnostic chat communication objects: immutable `ChatMessage` turns plus normalized `GenerationResult` and token usage models. |
| `fieldguide_ai/providers/` | Stateful conversation abstraction, immutable registry, SDK adapters, explicit backend dependencies, and normalized provider errors. |
| `fieldguide_ai/ingestion/` | Markdown loading, frontmatter parsing, section chunking, and document replacement through a focused index boundary. |
| `fieldguide_ai/vectorstore/` | Focused search/index interfaces, composition factory, embedding abstraction, and Chroma/NumPy/FAISS persistence. |
| `fieldguide_ai/errors.py` | Application-level configuration, provider, embedding, document-loading, and vector-store exceptions. |
| `langchain_pandas/` | Validated dataframe catalog with defensive snapshots and constrained inspection/query tools. |

## Observable workflows

### Markdown indexing

1. Discover Markdown files at the requested path.
2. Parse optional frontmatter and document content.
3. Split content by Markdown headings and enforce the configured word limit with overlap.
4. Generate embeddings for each chunk.
5. Replace existing chunks for each document in the selected vector store.
6. Report indexed document and chunk counts.

Embedding is completed before persisted document records are replaced, reducing the chance that an embedding failure removes a usable existing index.

### Stateful chat

1. Set the configured system prompt on the provider, separate from turn history.
2. Embed the question and retrieve the configured number of nearest chunks when a vector store is enabled.
3. Send accumulated turns plus an augmented copy of the current question and retrieved context to the provider. Only the original question is stored in visible history.
4. Normalize the provider payload into a `GenerationResult`; only after generation succeeds, atomically append the original user question and answer to history, then display the retrieved document/section sources.
5. Allow the user to inspect, clear, or reconfigure the session. Provider and model changes preserve turn history where possible; clearing keeps the system prompt and generation log.

This describes observable state and tool flow; it does not expose hidden model reasoning.

## Configuration

| Setting | Default | Notes |
| --- | --- | --- |
| LLM provider | `openai` | OpenAI and Anthropic are registered. |
| Chat model | `gpt-5-nano` | The wizard also offers `gpt-5-mini`, `gpt-4o-mini`, and a custom model name. |
| Embedding model | `text-embedding-3-small` | Configurable through `--embedding-model` in the flag-based CLI. |
| Vector store | `chroma` | `numpy`, `faiss`, and retrieval-free `none` are available. |
| Chroma path | `chroma_db` | Local persistent directory. |
| Chroma collection | `documents` | Configurable in both CLI flows. |
| NumPy path | `numpy_index.npz` | Compressed local persistence file. |
| FAISS path | `faiss_index` | Prefix for `.faiss` index and `.json` metadata files. |
| Retrieval count | `5` | Configurable with `--top-k`; shown in the interactive summary. |
| Maximum chunk size | `900` words | Large sections use overlapping splits. |
| `OPENAI_API_KEY` | None | Required before constructing OpenAI chat or embedding clients. |
| `ANTHROPIC_API_KEY` | None | Required before constructing an Anthropic chat client. |

## Testing and quality checks

Run the unit test suite:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest discover -s tests
```

Run lint and formatting checks without adding development dependencies to the project:

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_TOOL_DIR=/tmp/uv-tools uvx ruff check .
UV_CACHE_DIR=/tmp/uv-cache UV_TOOL_DIR=/tmp/uv-tools uvx ruff format --check .
```

Install the development extras and run the static type checker:

```bash
uv sync --extra dev
uv run pyright
```

Tests cover validated and defensive value objects, atomic message history, provider error translation and dependency injection, immutable interactive configuration, KnowledgeBot context/history behavior, Markdown parsing and chunking, Chroma/NumPy/FAISS persistence and search, metadata serialization, and dataframe tools.

## Guardrails and data handling

- CLI entry points read credentials from the environment and inject them into provider backends; credentials must not be committed.
- Ingestion reads local Markdown content and persists embeddings and source text locally in the selected store.
- The dataframe agent is instructed to use its registered tools and avoid answering from general knowledge, but model output should still be treated as untrusted until independently verified.
- Retrieved local chunk text is sent to the configured model provider as prompt context.
- The primary chat loop has no external action tools or approval workflow.
- There is no production authentication, authorization, audit log, or tenant isolation layer.

## Limitations and tradeoffs

- Embeddings currently depend on OpenAI, including when Anthropic is selected for chat.
- Markdown is the only supported ingestion format in the indexing pipeline.
- The NumPy store is intended for small local indexes; it loads records into memory for search.
- The dataframe demo depends on a fixed local CSV directory and is not integrated with the primary CLI.
- Formal quality, latency, cost, groundedness, and human-intervention benchmarks have not been established.
- Retry policies, production observability, and model-output validation are not yet implemented.

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
├── fieldguide_ai/
│   ├── ingestion/          # Markdown loading and chunking pipeline
│   ├── chat/               # ChatMessage, GenerationResult, and token usage
│   ├── providers/          # LLM abstractions, adapters, and registry
│   ├── vectorstore/        # Chroma, NumPy, and FAISS vector stores
│   ├── knowledge_bot.py    # Retrieval-grounded chat orchestration
│   ├── errors.py           # Stable application-level error boundary
│   ├── terminal.py         # Shared terminal history rendering
│   ├── cli.py              # Flag-based CLI and chat loop
│   └── interactive.py      # Rich questionary wizard and chat loop
├── langchain_pandas/       # Dataframe agent catalog and tools
├── tests/                  # unittest suite
├── langchain_main.py       # CSV dataframe agent entry point
├── main.py                 # Flag-based CLI entry point
└── pyproject.toml          # Package metadata and dependencies
```

## Extension points

- Register another LLM by implementing `ProviderBackend` for SDK-client construction and model discovery, implementing `LLMProvider` for a configured chat session, and composing a validated `ProviderSpec` into a `ProviderRegistry`.
- Add another vector backend by implementing the `VectorStore` interface and wiring it into CLI construction.
- Add ingestion formats behind loaders that produce the existing document model.
- Add safe dataframe operations as explicit tools rather than enabling arbitrary Python execution.

## Roadmap

1. Add groundedness, retrieval, latency, and cost evaluations.
2. Add provider-neutral embedding and model configuration.
3. Add production logging, retries, and explicit approval boundaries for future actions.

## Contributing and license

Keep implementation, tests, and `README.md` synchronized in the same change. Run the test and lint commands above before submitting changes.

No license file is currently included. Add one before distributing or reusing the project outside its current context.
