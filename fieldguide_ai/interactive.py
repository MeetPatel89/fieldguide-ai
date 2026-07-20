"""Rich interactive chat wizard for Fieldguide AI."""

import os
import sys
from dataclasses import dataclass
from typing import TextIO

import questionary
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from fieldguide_ai.cli import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_NUMPY_PATH,
    build_vector_store,
    index_corpus,
    print_history,
)
from fieldguide_ai.demo import build_system_prompt
from fieldguide_ai.knowledge_bot import KnowledgeAnswer, KnowledgeBot
from fieldguide_ai.messages import ChatMessage
from fieldguide_ai.providers import PROVIDERS, LLMProvider, ProviderSpec
from fieldguide_ai.vectorstore import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_FAISS_PATH,
    OpenAIEmbeddingProvider,
    VectorSearchResult,
    VectorStore,
)

OTHER_MODEL = "Other (type a model name)"
NONE_STORE = "none (plain chat)"
DEFAULT_MAX_WORDS = 900
CHAT_COMMANDS = (
    ":help",
    ":config",
    ":provider",
    ":model",
    ":store",
    ":system",
    ":history",
    ":clear",
    ":quit",
)


@dataclass
class SessionConfig:
    """Mutable configuration for an interactive chat session."""

    provider_name: str = "openai"
    model: str = "gpt-5-nano"
    store_type: str | None = "chroma"
    store_path: str | None = DEFAULT_CHROMA_PATH
    collection_name: str = DEFAULT_COLLECTION_NAME
    system_prompt: str = ""
    top_k: int = 5


class WizardCancelledError(Exception):
    """Raised internally when a wizard operation is cancelled."""


def _ask(question: questionary.Question) -> object:
    """Ask a question and return the answer."""
    answer = question.ask()
    if answer is None:
        raise WizardCancelledError("Wizard operation cancelled.")
    return answer


def _ask_text(question: questionary.Question) -> str:
    """Ask a question whose answer must be text."""
    return str(_ask(question))


def _provider_choices() -> tuple[list[str], dict[str, ProviderSpec]]:
    providers_by_label = {provider.label: provider for provider in PROVIDERS.values()}
    return list(providers_by_label), providers_by_label


def _is_positive_integer(value: str) -> bool | str:
    try:
        if int(value) > 0:
            return True
    except ValueError:
        pass
    return "Enter a positive whole number."


def _available_models(
    provider_spec: ProviderSpec, output_stream: TextIO
) -> tuple[str, ...]:
    try:
        return provider_spec.available_models()
    except Exception as error:
        print(
            f"Could not load current {provider_spec.label} models: {error}. "
            "Using configured models.",
            file=output_stream,
        )
        return provider_spec.models


def _prompt_model(
    provider_spec: ProviderSpec,
    output_stream: TextIO,
    current_model: str | None = None,
) -> str:
    available_models = _available_models(provider_spec, output_stream)
    default_model = (
        current_model
        if current_model in available_models
        else provider_spec.default_model
        if provider_spec.default_model in available_models
        else available_models[0]
    )
    model = _ask_text(
        questionary.select(
            "Select a model:",
            choices=[*available_models, OTHER_MODEL],
            default=default_model,
        )
    )
    if model == OTHER_MODEL:
        return _ask_text(questionary.text("Model name:"))
    return model


def _prompt_provider_and_model(
    output_stream: TextIO,
) -> tuple[ProviderSpec, str]:
    provider_choices, providers_by_label = _provider_choices()
    provider_label = _ask_text(
        questionary.select("Select an LLM provider:", choices=provider_choices)
    )
    provider_spec = providers_by_label[provider_label]
    return provider_spec, _prompt_model(provider_spec, output_stream)


def _default_store_path(store_type: str) -> str:
    return {
        "chroma": DEFAULT_CHROMA_PATH,
        "numpy": DEFAULT_NUMPY_PATH,
        "faiss": DEFAULT_FAISS_PATH,
    }[store_type]


def _prompt_store(
    current: SessionConfig | None = None,
) -> tuple[str | None, str | None, str]:
    current_choice = (
        current.store_type if current and current.store_type else NONE_STORE
    )
    choice = _ask_text(
        questionary.select(
            "Select a vector store:",
            choices=["chroma", "numpy", "faiss", NONE_STORE],
            default=current_choice,
        )
    )
    if choice == NONE_STORE:
        return None, None, DEFAULT_COLLECTION_NAME

    default_path = (
        current.store_path
        if current and current.store_type == choice and current.store_path
        else _default_store_path(choice)
    )
    store_path = _ask_text(questionary.text("Vector store path:", default=default_path))
    collection_name = DEFAULT_COLLECTION_NAME
    if choice == "chroma":
        collection_name = _ask_text(
            questionary.text(
                "Chroma collection name:",
                default=(
                    current.collection_name if current else DEFAULT_COLLECTION_NAME
                ),
            )
        )
    return choice, store_path, collection_name


def _build_store(config: SessionConfig) -> VectorStore | None:
    if config.store_type is None:
        return None
    embedding_provider = OpenAIEmbeddingProvider(api_key=os.getenv("OPENAI_API_KEY"))
    return build_vector_store(
        provider_name=config.store_type,
        embedding_provider=embedding_provider,
        path=config.store_path,
        collection_name=config.collection_name,
    )


def _build_provider(
    provider_spec: ProviderSpec,
    config: SessionConfig,
    history: list[ChatMessage] | None = None,
) -> LLMProvider:
    provider = provider_spec.build_provider(config.model)
    provider.system_prompt = config.system_prompt
    for message in history or []:
        provider.add_message(message)
    return provider


def _console(output_stream: TextIO) -> Console:
    return Console(file=output_stream, highlight=False)


def _show_banner(console: Console) -> None:
    console.print(
        Panel.fit(
            "[bold cyan]Fieldguide AI[/bold cyan]\n"
            "[dim]Retrieval-grounded Knowledge Chatbot for "
            "Nautilus Financial Services Incident Reports[/dim]",
            border_style="cyan",
        )
    )


def _config_table(config: SessionConfig) -> Table:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("Provider", config.provider_name)
    table.add_row("Model", config.model)
    table.add_row("Vector store", config.store_type or "none (plain chat)")
    if config.store_type is not None:
        table.add_row("Store path", config.store_path or "")
    if config.store_type == "chroma":
        table.add_row("Collection", config.collection_name)
    table.add_row("Top-k", str(config.top_k))
    table.add_row("System prompt", config.system_prompt)
    return table


def _show_config(console: Console, config: SessionConfig) -> None:
    console.print(Panel(_config_table(config), title="Session configuration"))


def _show_help(console: Console) -> None:
    table = Table(title="Chat commands", show_header=False)
    table.add_column(style="bold cyan")
    table.add_column()
    descriptions = {
        ":help": "Show this command list",
        ":config": "Show current session settings",
        ":provider": "Change provider and model",
        ":model": "Change model",
        ":store": "Change vector store or switch to plain chat",
        ":system": "Change the system prompt",
        ":history": "Show conversation history",
        ":clear": "Clear conversation history",
        ":quit": "Exit chat (:q and :exit also work)",
    }
    for command in CHAT_COMMANDS:
        table.add_row(command, descriptions[command])
    console.print(table)


def _show_sources(console: Console, sources: list[VectorSearchResult]) -> None:
    if not sources:
        return
    lines = Text()
    for index, source in enumerate(sources, start=1):
        path = str(source.metadata.get("source_path", "unknown"))
        section = str(
            source.metadata.get("section_path")
            or source.metadata.get("section_title", "unknown")
        )
        if index > 1:
            lines.append("\n")
        lines.append(f"{index}. ", style="bold cyan")
        lines.append(path)
        lines.append(f" — {section}", style="dim")
    console.print(Panel(lines, title="Sources", border_style="cyan"))


def run_chat_loop(
    provider: LLMProvider,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
    system_prompt: str | None = None,
    config: SessionConfig | None = None,
    vector_store: VectorStore | None = None,
) -> None:
    """Run the rich chat loop and support live session reconfiguration."""
    if config is None:
        config = SessionConfig(
            system_prompt=system_prompt
            or provider.system_prompt
            or build_system_prompt(),
            store_type=None,
            store_path=None,
        )
    provider.system_prompt = config.system_prompt
    bot = KnowledgeBot(provider, vector_store)
    console = _console(output_stream)
    console.print(
        "[green]Chat ready.[/green] Type [bold]:help[/bold] for commands or "
        "[bold]:quit[/bold] to exit."
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
        if command in {":quit", ":q", ":exit", "quit", "exit"}:
            console.print("[cyan]Goodbye.[/cyan]")
            return
        if command == ":help":
            _show_help(console)
            continue
        if command == ":config":
            _show_config(console, config)
            continue
        if command == ":history":
            print_history(provider, output_stream)
            continue
        if command == ":clear":
            provider.clear_history()
            console.print("[green]History cleared.[/green]")
            continue

        try:
            if command == ":provider":
                provider_spec, model = _prompt_provider_and_model(output_stream)
                updated = SessionConfig(**vars(config))
                updated.provider_name = provider_spec.name
                updated.model = model
                provider = _build_provider(
                    provider_spec, updated, history=provider.get_history()
                )
                config.provider_name = updated.provider_name
                config.model = updated.model
                bot = KnowledgeBot(provider, bot.vector_store)
                _show_config(console, config)
                continue
            if command == ":model":
                provider_spec = PROVIDERS[config.provider_name]
                model = _prompt_model(provider_spec, output_stream, config.model)
                updated = SessionConfig(**vars(config))
                updated.model = model
                provider = _build_provider(
                    provider_spec, updated, history=provider.get_history()
                )
                config.model = model
                bot = KnowledgeBot(provider, bot.vector_store)
                _show_config(console, config)
                continue
            if command == ":store":
                store_type, store_path, collection_name = _prompt_store(config)
                updated = SessionConfig(**vars(config))
                updated.store_type = store_type
                updated.store_path = store_path
                updated.collection_name = collection_name
                vector_store = _build_store(updated)
                config.store_type = store_type
                config.store_path = store_path
                config.collection_name = collection_name
                bot = KnowledgeBot(provider, vector_store)
                _show_config(console, config)
                continue
            if command == ":system":
                prompt = _ask_text(
                    questionary.text("System prompt:", default=config.system_prompt)
                )
                config.system_prompt = prompt
                provider.system_prompt = prompt
                _show_config(console, config)
                continue
        except WizardCancelledError:
            console.print("[yellow]Configuration change cancelled.[/yellow]")
            continue

        if command.startswith(":"):
            console.print(
                f"[yellow]Unknown command {command!r}.[/yellow] Type :help for help."
            )
            continue

        with console.status("[cyan]Searching and generating…[/cyan]"):
            response: KnowledgeAnswer = bot.ask(user_input, top_k=config.top_k)
        console.print("\n[bold cyan]Assistant[/bold cyan]")
        console.print(Markdown(response.answer))
        _show_sources(console, response.sources)


def run_wizard(
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
) -> None:
    """Configure and start an interactive retrieval-grounded chat session."""
    console = _console(output_stream)
    _show_banner(console)
    provider_spec, model = _prompt_provider_and_model(output_stream)
    store_type, store_path, collection_name = _prompt_store()
    system_prompt = _ask_text(
        questionary.text("System prompt:", default=build_system_prompt())
    )
    config = SessionConfig(
        provider_name=provider_spec.name,
        model=model,
        store_type=store_type,
        store_path=store_path,
        collection_name=collection_name,
        system_prompt=system_prompt,
    )
    vector_store = _build_store(config)

    if vector_store is not None and bool(
        _ask(questionary.confirm("Run ingestion pipeline?", default=False))
    ):
        corpus_path = _ask_text(questionary.text("Corpus path:"))
        max_words = int(
            _ask_text(
                questionary.text(
                    "Maximum words per chunk:",
                    default=str(DEFAULT_MAX_WORDS),
                    validate=_is_positive_integer,
                )
            )
        )
        index_corpus(
            corpus_path=corpus_path,
            vector_store=vector_store,
            max_words=max_words,
            output_stream=output_stream,
        )

    provider = _build_provider(provider_spec, config)
    _show_config(console, config)
    run_chat_loop(
        provider,
        input_stream=input_stream,
        output_stream=output_stream,
        config=config,
        vector_store=vector_store,
    )


def main() -> None:
    """Run the interactive chat wizard entry point."""
    load_dotenv()
    try:
        run_wizard()
    except (EOFError, KeyboardInterrupt, WizardCancelledError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
