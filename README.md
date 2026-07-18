# Fieldguide AI

Fieldguide AI is an experimental local knowledge assistant that turns Markdown corpora and structured CSV data into searchable, model-assisted workflows.

> **Status:** Active prototype. Markdown ingestion, local vector persistence, OpenAI chat, and a separate dataframe-tool agent are implemented. Retrieval from the vector store is not yet connected to the main chat response path.

## Quickstart

### Prerequisites

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key for chat or embedding operations

Install the project from the repository root:

```bash
uv sync
```

Create a local `.env` file:

```dotenv
OPENAI_API_KEY=your-api-key
```

Do not commit `.env` or real credentials.

Start the guided CLI:

```bash
uv run fieldguide
```

The wizard prompts for an LLM provider and model, vector-store settings, a system prompt, and whether to ingest a Markdown corpus before starting a stateful chat session.

## What it does

- Provides a menu-driven CLI for model, vector-store, prompt, and ingestion configuration.
- Loads Markdown files with optional frontmatter and splits them into section-aware, overlapping chunks.
- Generates OpenAI embeddings and persists vectors in Chroma or a local NumPy `.npz` index.
- Maintains stateful OpenAI chat history with commands for inspection and reset.
- Uses an extensible provider registry; OpenAI is the currently registered LLM provider.
- Includes a separate LangChain dataframe agent with explicit tools for inspecting, searching, filtering, and aggregating CSV datasets.

## Problem and target user

Operational knowledge often lives across runbooks, policies, product documentation, FAQs, and tabular exports. Finding the relevant source and turning it into a defensible answer becomes slow and inconsistent as the corpus grows.

- **Primary user:** A forward-deployed engineer, support operator, or operations specialist working with customer and company knowledge.
- **Job to be done:** Ingest local knowledge, inspect or search it consistently, and use a model-assisted interface to answer follow-up questions.
- **Secondary user:** A developer extending provider, vector-store, ingestion, or dataframe-tool behavior.

## Why model-assisted behavior

Loading, chunking, validation, persistence, and dataframe operations are deterministic and remain ordinary Python pipelines. A model is useful where the user asks open-ended questions, conversational context must be maintained, or the appropriate dataframe tool and arguments must be selected dynamically.

The project does not treat every step as agentic. Deterministic work stays explicit and testable; model-driven work is isolated behind provider and tool interfaces.

## Usage

### Interactive wizard

```bash
uv run fieldguide
```

Available chat commands:

| Command | Effect |
| --- | --- |
| `:history` | Print the current system, user, and assistant messages. |
| `:clear` | Clear conversation history and restore the configured system prompt. |
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

Start stateful chat directly:

```bash
uv run python main.py --model gpt-5-nano
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
  +-- questionary wizard --------> provider registry --> OpenAI Responses API
  |          |
  |          +--> optional Markdown ingestion
  |                    |
  |                    +--> loader --> section chunker --> embeddings
  |                                                        |
  |                                                        +--> Chroma or NumPy
  |
  +-- dataframe CLI -----------> LangChain agent --> constrained pandas tools
```

| Component | Responsibility |
| --- | --- |
| `fieldguide_ai/interactive.py` | Guided configuration, optional ingestion, and chat startup. |
| `fieldguide_ai/cli.py` | Flag-based commands, indexing orchestration, and stateful chat loop. |
| `fieldguide_ai/providers/` | Provider interface, OpenAI adapter, and provider registry. |
| `fieldguide_ai/ingestion/` | Markdown loading, frontmatter parsing, section chunking, and document replacement. |
| `fieldguide_ai/vectorstore/` | Embedding abstraction plus Chroma and NumPy persistence/search. |
| `langchain_pandas/` | Dataframe catalog and constrained inspection/query tools. |

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

1. Add the configured system prompt to provider history.
2. Append each user message.
3. Send the accumulated conversation to the provider.
4. Append and display the assistant response.
5. Allow the user to inspect or clear the history.

This describes observable state and tool flow; it does not expose hidden model reasoning.

## Configuration

| Setting | Default | Notes |
| --- | --- | --- |
| LLM provider | `openai` | The provider registry currently contains OpenAI only. |
| Chat model | `gpt-5-nano` | The wizard also offers `gpt-5-mini`, `gpt-4o-mini`, and a custom model name. |
| Embedding model | `text-embedding-3-small` | Configurable through `--embedding-model` in the flag-based CLI. |
| Vector store | `chroma` | `numpy` is available for smaller local indexes. |
| Chroma path | `chroma_db` | Local persistent directory. |
| Chroma collection | `documents` | Configurable in both CLI flows. |
| NumPy path | `numpy_index.npz` | Compressed local persistence file. |
| Maximum chunk size | `900` words | Large sections use overlapping splits. |
| `OPENAI_API_KEY` | None | Required before constructing OpenAI chat or embedding clients. |

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

Tests cover message history, provider behavior, interactive configuration, Markdown parsing and chunking, indexing replacement, vector-store persistence/search, metadata serialization, and dataframe tools.

## Guardrails and data handling

- Credentials are read from the environment and must not be committed.
- Ingestion reads local Markdown content and persists embeddings and source text locally in the selected store.
- The dataframe agent is instructed to use its registered tools and avoid answering from general knowledge, but model output should still be treated as untrusted until independently verified.
- The primary chat loop can call the configured model but has no external action tools or approval workflow.
- There is no production authentication, authorization, audit log, or tenant isolation layer.

## Limitations and tradeoffs

- The primary chat loop does **not** yet retrieve vector-store results or add them to model context. Indexing and chat are adjacent capabilities, not a complete RAG loop.
- LLM and embedding integrations currently depend on OpenAI.
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
│   ├── providers/          # LLM abstractions and registry
│   ├── vectorstore/        # Chroma and NumPy vector stores
│   ├── cli.py              # Flag-based CLI and chat loop
│   └── interactive.py      # questionary wizard
├── langchain_pandas/       # Dataframe agent catalog and tools
├── tests/                  # unittest suite
├── langchain_main.py       # CSV dataframe agent entry point
├── main.py                 # Flag-based CLI entry point
└── pyproject.toml          # Package metadata and dependencies
```

## Extension points

- Register another LLM by adding a `ProviderSpec` in `fieldguide_ai/providers/registry.py` and implementing `LLMProvider`.
- Add another vector backend by implementing the `VectorStore` interface and wiring it into CLI construction.
- Add ingestion formats behind loaders that produce the existing document model.
- Add safe dataframe operations as explicit tools rather than enabling arbitrary Python execution.

## Roadmap

1. Connect vector retrieval to the main chat path and include source attribution.
2. Add groundedness, retrieval, latency, and cost evaluations.
3. Add provider-neutral embedding and model configuration.
4. Add production logging, retries, and explicit approval boundaries for future actions.

## Contributing and license

Keep implementation, tests, and `README.md` synchronized in the same change. Run the test and lint commands above before submitting changes.

No license file is currently included. Add one before distributing or reusing the project outside its current context.
