"""Command-line interface for Fieldguide AI."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TextIO

from dotenv import load_dotenv

from fieldguide_ai.demo import build_demo_messages, build_system_prompt
from fieldguide_ai.errors import ConfigurationError
from fieldguide_ai.ingestion import (
    DocumentIndexingPipeline,
    IndexingResult,
    MarkdownSectionChunker,
    load_markdown_documents,
)
from fieldguide_ai.knowledge_bot import KnowledgeBot
from fieldguide_ai.providers import (
    LLMProvider,
    OpenAIProvider,
    ProviderRegistry,
    registry_from_environment,
)
from fieldguide_ai.providers import (
    build_provider as build_registered_provider,
)
from fieldguide_ai.providers.registry import OPENAI_DEFAULT_MODEL
from fieldguide_ai.terminal import write_history
from fieldguide_ai.vectorstore import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    VectorStore,
)
from fieldguide_ai.vectorstore import (
    build_vector_store as build_configured_vector_store,
)

DEFAULT_MODEL = OPENAI_DEFAULT_MODEL
EXIT_COMMANDS = {":exit", ":q", ":quit", "exit", "quit"}


def build_provider(
    model: str,
    registry: ProviderRegistry | None = None,
) -> OpenAIProvider:
    """Build the registered OpenAI provider for a model."""
    provider = build_registered_provider("openai", model, registry)
    if not isinstance(provider, OpenAIProvider):
        raise TypeError("the openai registry entry did not create an OpenAIProvider")
    return provider


def build_vector_store(
    provider_name: str,
    embedding_provider: EmbeddingProvider,
    path: str | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> VectorStore:
    """Build the requested vector-store implementation."""
    return build_configured_vector_store(
        provider_name=provider_name,
        embedding_provider=embedding_provider,
        path=path,
        collection_name=collection_name,
    )


def run_demo(provider: LLMProvider, output_stream: TextIO = sys.stdout) -> None:
    """Run the stateless demonstration prompt."""
    provider.set_system_prompt(build_system_prompt())
    result = provider.generate(build_demo_messages())
    output_stream.write(f"{result.text}\n")


def run_chat_loop(
    provider: LLMProvider,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
    system_prompt: str | None = None,
    vector_store: VectorStore | None = None,
    top_k: int = 5,
) -> None:
    """Run an interactive, stateful chat session with optional retrieval."""
    provider.set_system_prompt(
        build_system_prompt() if system_prompt is None else system_prompt
    )
    output_stream.write(
        "Stateful chat started. Type :quit to exit, :history to inspect state.\n"
    )
    bot = KnowledgeBot(provider, vector_store)

    while True:
        output_stream.write("\nYou> ")
        output_stream.flush()

        user_input = input_stream.readline()
        if user_input == "":
            output_stream.write("\n")
            return

        user_input = user_input.strip()
        if not user_input:
            continue

        command = user_input.lower()
        if command in EXIT_COMMANDS:
            output_stream.write("Goodbye.\n")
            return

        if command == ":history":
            print_history(provider, output_stream)
            continue

        if command == ":clear":
            provider.clear_history()
            output_stream.write("History cleared.\n")
            continue

        response = bot.ask(user_input, top_k=top_k)
        output_stream.write(f"\nAssistant> {response.answer}\n")
        if response.sources:
            output_stream.write("Sources:\n")
            for source in response.sources:
                path = source.metadata.get("source_path", "unknown")
                section = source.metadata.get("section_path") or source.metadata.get(
                    "section_title", "unknown"
                )
                output_stream.write(f"- {path} — {section}\n")


def print_history(provider: LLMProvider, output_stream: TextIO = sys.stdout) -> None:
    """Write the provider's conversation history to a stream."""
    write_history(provider, output_stream)


def preview_chunks(
    corpus_path: str | Path,
    max_words: int,
    limit: int,
    output_stream: TextIO = sys.stdout,
    details: bool = False,
) -> None:
    """Load a corpus and write a preview of its chunks."""
    documents = load_markdown_documents(corpus_path)
    chunker = MarkdownSectionChunker(max_words=max_words)
    chunks = chunker.chunk_documents(documents)

    output_stream.write(
        f"Loaded {len(documents)} documents and created {len(chunks)} chunks.\n"
    )
    for chunk in chunks[:limit]:
        section = " > ".join(chunk.section_path)
        doc_type = chunk.metadata.get("doc_type", "unknown")
        if details:
            output_stream.write(f"\n{json.dumps(chunk.to_record(), indent=2)}\n")
        else:
            output_stream.write(
                f"\n{chunk.chunk_id} [{doc_type}] {section}\n"
                f"source: {chunk.source_path}\n"
                f"words: {len(chunk.content.split())}\n"
                f"{_preview_text(chunk.content)}\n"
            )


def index_corpus(
    corpus_path: str | Path,
    vector_store: VectorStore,
    max_words: int,
    output_stream: TextIO = sys.stdout,
) -> IndexingResult:
    """Load and index a Markdown corpus in a vector store."""
    pipeline = DocumentIndexingPipeline(
        vector_store=vector_store,
        chunker=MarkdownSectionChunker(max_words=max_words),
    )
    result = pipeline.index_path(corpus_path)
    output_stream.write(
        f"Indexed {result.document_count} documents and {result.chunk_count} chunks.\n"
    )
    return result


def _parse_bool(value: str) -> bool:
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


def _parse_positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a positive whole number, got {value!r}"
        ) from None
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"expected a positive whole number, got {value!r}"
        )
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fieldguide AI command-line interface."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the original stateless demo prompt instead of interactive chat.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use. Defaults to {DEFAULT_MODEL}.",
    )
    corpus_action = parser.add_mutually_exclusive_group()
    corpus_action.add_argument(
        "--chunk-corpus",
        metavar="PATH",
        help="Preview markdown chunks for a corpus path without calling an LLM.",
    )
    corpus_action.add_argument(
        "--index-corpus",
        metavar="PATH",
        help="Load, chunk, embed, and index a Markdown corpus.",
    )
    parser.add_argument(
        "--chunk-max-words",
        type=int,
        default=900,
        help="Maximum words per chunk when previewing or indexing. Defaults to 900.",
    )
    parser.add_argument(
        "--chunk-limit",
        type=int,
        default=10,
        help="Number of preview chunks to print. Defaults to 10.",
    )
    parser.add_argument(
        "--chunk-details",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        metavar="BOOL",
        help="Print full JSON for each chunk. Use alone or pass true/false.",
    )
    parser.add_argument(
        "--vector-store",
        choices=("chroma", "numpy", "faiss", "none"),
        default="chroma",
        help="Vector store used for indexing and chat. Use none for plain chat.",
    )
    parser.add_argument(
        "--store-path",
        help="Storage path for Chroma, NumPy (.npz), or FAISS (file prefix).",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help=f"Chroma collection name. Defaults to {DEFAULT_COLLECTION_NAME}.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"OpenAI embedding model. Defaults to {DEFAULT_EMBEDDING_MODEL}.",
    )
    parser.add_argument(
        "--top-k",
        type=_parse_positive_integer,
        default=5,
        help="Number of chunks retrieved for each chat question. Defaults to 5.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the Fieldguide command-line interface."""
    load_dotenv()
    args = parse_args(argv)
    provider_registry = registry_from_environment()

    if args.chunk_corpus:
        preview_chunks(
            corpus_path=args.chunk_corpus,
            max_words=args.chunk_max_words,
            limit=args.chunk_limit,
            details=args.chunk_details,
        )
        return

    if args.index_corpus:
        if args.vector_store == "none":
            parser_error = "--vector-store none cannot be used with --index-corpus"
            raise ConfigurationError(parser_error)
        embedding_provider = OpenAIEmbeddingProvider(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=args.embedding_model,
        )
        vector_store = build_vector_store(
            provider_name=args.vector_store,
            embedding_provider=embedding_provider,
            path=args.store_path,
            collection_name=args.collection_name,
        )
        index_corpus(
            corpus_path=args.index_corpus,
            vector_store=vector_store,
            max_words=args.chunk_max_words,
        )
        return

    provider = build_provider(model=args.model, registry=provider_registry)

    if args.demo:
        run_demo(provider)
        return

    vector_store = None
    if args.vector_store != "none":
        embedding_provider = OpenAIEmbeddingProvider(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=args.embedding_model,
        )
        vector_store = build_vector_store(
            provider_name=args.vector_store,
            embedding_provider=embedding_provider,
            path=args.store_path,
            collection_name=args.collection_name,
        )
    run_chat_loop(provider, vector_store=vector_store, top_k=args.top_k)


def _preview_text(text: str, max_chars: int = 240) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3]}..."
