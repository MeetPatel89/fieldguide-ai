import argparse
import json
import os
import sys
from pathlib import Path
from typing import TextIO

from dotenv import load_dotenv

from fieldguide_ai.demo import build_demo_messages, build_system_prompt
from fieldguide_ai.ingestion import (
    DocumentIndexingPipeline,
    IndexingResult,
    MarkdownSectionChunker,
    load_markdown_documents,
)
from fieldguide_ai.providers import (
    LLMProvider,
    OpenAIProvider,
    build_provider as build_registered_provider,
    get_provider,
)
from fieldguide_ai.vectorstore import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    ChromaVectorStore,
    EmbeddingProvider,
    NumpyVectorStore,
    OpenAIEmbeddingProvider,
    VectorStore,
)

DEFAULT_MODEL = get_provider("openai").default_model
DEFAULT_CHROMA_PATH = "chroma_db"
DEFAULT_NUMPY_PATH = "numpy_index.npz"
EXIT_COMMANDS = {":exit", ":q", ":quit", "exit", "quit"}


def build_provider(model: str) -> OpenAIProvider:
    provider = build_registered_provider("openai", model)
    if not isinstance(provider, OpenAIProvider):
        raise TypeError("the openai registry entry did not create an OpenAIProvider")
    return provider


def build_vector_store(
    provider_name: str,
    embedding_provider: EmbeddingProvider,
    path: str | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> VectorStore:
    if provider_name == "chroma":
        return ChromaVectorStore(
            path=path or DEFAULT_CHROMA_PATH,
            collection_name=collection_name,
            embedding_provider=embedding_provider,
        )
    if provider_name == "numpy":
        return NumpyVectorStore(
            path=path or DEFAULT_NUMPY_PATH,
            embedding_provider=embedding_provider,
        )
    raise ValueError(f"unsupported vector store provider: {provider_name}")


def run_demo(provider: LLMProvider, output_stream: TextIO = sys.stdout) -> None:
    provider.system_prompt = build_system_prompt()
    result = provider.generate(build_demo_messages())
    output_stream.write(f"{result.text}\n")


def run_chat_loop(
    provider: LLMProvider,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
    system_prompt: str | None = None,
) -> None:
    provider.system_prompt = (
        build_system_prompt() if system_prompt is None else system_prompt
    )
    output_stream.write(
        "Stateful chat started. Type :quit to exit, :history to inspect state.\n"
    )

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

        response_text = provider.chat(user_input)
        output_stream.write(f"\nAssistant> {response_text}\n")


def print_history(provider: LLMProvider, output_stream: TextIO = sys.stdout) -> None:
    index = 1
    if provider.system_prompt is not None:
        output_stream.write(f"{index}. system: {provider.system_prompt}\n")
        index += 1
    for message in provider.get_history():
        output_stream.write(f"{index}. {message.role}: {message.content}\n")
        index += 1


def preview_chunks(
    corpus_path: str | Path,
    max_words: int,
    limit: int,
    output_stream: TextIO = sys.stdout,
    details: bool = False,
) -> None:
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
        choices=("chroma", "numpy"),
        default="chroma",
        help="Vector store used by --index-corpus. Defaults to chroma.",
    )
    parser.add_argument(
        "--store-path",
        help="Storage directory for Chroma or .npz path for NumPy.",
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)

    if args.chunk_corpus:
        preview_chunks(
            corpus_path=args.chunk_corpus,
            max_words=args.chunk_max_words,
            limit=args.chunk_limit,
            details=args.chunk_details,
        )
        return

    if args.index_corpus:
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

    provider = build_provider(model=args.model)

    if args.demo:
        run_demo(provider)
        return

    run_chat_loop(provider)


def _preview_text(text: str, max_chars: int = 240) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3]}..."
