import argparse
import json
import os
import sys
from pathlib import Path
from typing import TextIO

from dotenv import load_dotenv

from fieldguide_ai.demo import build_demo_messages, build_system_message
from fieldguide_ai.ingestion import MarkdownSectionChunker, load_markdown_documents
from fieldguide_ai.providers import LLMProvider, OpenAIProvider

DEFAULT_MODEL = "gpt-5-nano"
EXIT_COMMANDS = {":exit", ":q", ":quit", "exit", "quit"}


def build_provider(model: str) -> OpenAIProvider:
    return OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=model,
    )


def run_demo(provider: LLMProvider, output_stream: TextIO = sys.stdout) -> None:
    response_text = provider.generate(build_demo_messages())
    output_stream.write(f"{response_text}\n")


def run_chat_loop(
    provider: LLMProvider,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
) -> None:
    provider.add_message(build_system_message())
    output_stream.write("Stateful chat started. Type :quit to exit, :history to inspect state.\n")

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
            provider.add_message(build_system_message())
            output_stream.write("History cleared.\n")
            continue

        response_text = provider.chat(user_input)
        output_stream.write(f"\nAssistant> {response_text}\n")


def print_history(provider: LLMProvider, output_stream: TextIO = sys.stdout) -> None:
    for index, message in enumerate(provider.get_history(), start=1):
        output_stream.write(f"{index}. {message.role}: {message.content}\n")


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

    output_stream.write(f"Loaded {len(documents)} documents and created {len(chunks)} chunks.\n")
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


def _parse_bool(value: str) -> bool:
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fieldguide AI command-line interface.")
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
    parser.add_argument(
        "--chunk-corpus",
        metavar="PATH",
        help="Preview markdown chunks for a corpus path without calling an LLM.",
    )
    parser.add_argument(
        "--chunk-max-words",
        type=int,
        default=900,
        help="Maximum words per preview chunk. Defaults to 900.",
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
