import argparse
import os
import sys
from typing import TextIO

from dotenv import load_dotenv

from fieldguide_ai.demo import build_demo_messages, build_system_message
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)
    provider = build_provider(model=args.model)

    if args.demo:
        run_demo(provider)
        return

    run_chat_loop(provider)
