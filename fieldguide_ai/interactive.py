"""Rich interactive chat wizard for Fieldguide AI."""

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TextIO

import questionary
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from fieldguide_ai.chat import ChatMessage
from fieldguide_ai.errors import ConfigurationError, FieldguideError
from fieldguide_ai.ingestion import (
    DocumentIndexingPipeline,
    IndexingResult,
    MarkdownSectionChunker,
)
from fieldguide_ai.knowledge_bot import KnowledgeAnswer, KnowledgeBot
from fieldguide_ai.providers import (
    LLMProvider,
    ProviderRegistry,
    ProviderSpec,
    registry_from_environment,
)
from fieldguide_ai.terminal import write_history
from fieldguide_ai.vectorstore import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_FAISS_PATH,
    DEFAULT_NUMPY_PATH,
    OpenAIEmbeddingProvider,
    VectorSearchResult,
    VectorStore,
    build_vector_store,
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

DEFAULT_SYSTEM_PROMPT = (
    "You are a domain expert in incident reports for Nautilus Financial Services. "
    "You are capable of answering user queries about issues/incidents for Nautilus "
    "Financial Services without making up stuff or referring to external knowledge "
    "not strictly within company corpus."
)


@dataclass(frozen=True)
class SessionConfig:
    """Validated value object describing an interactive chat session."""

    provider_name: str = "openai"
    model: str = "gpt-5-nano"
    store_type: str | None = "chroma"
    store_path: str | None = DEFAULT_CHROMA_PATH
    collection_name: str = DEFAULT_COLLECTION_NAME
    system_prompt: str = ""
    top_k: int = 5

    def __post_init__(self) -> None:
        """Prevent partially configured sessions."""
        if not self.provider_name.strip():
            raise ConfigurationError("provider name must not be blank")
        if not self.model.strip():
            raise ConfigurationError("model must not be blank")
        if self.store_type not in {None, "chroma", "numpy", "faiss"}:
            raise ConfigurationError(f"unsupported vector store: {self.store_type}")
        if self.store_type is None and self.store_path is not None:
            raise ConfigurationError("plain chat cannot have a vector-store path")
        if self.store_type is not None and not (self.store_path or "").strip():
            raise ConfigurationError("a configured vector store requires a path")
        if not self.collection_name.strip():
            raise ConfigurationError("collection name must not be blank")
        if self.top_k <= 0:
            raise ConfigurationError("top_k must be greater than zero")

    def with_provider(self, provider_name: str, model: str) -> "SessionConfig":
        """Return this configuration with a different provider and model."""
        return replace(self, provider_name=provider_name, model=model)

    def with_model(self, model: str) -> "SessionConfig":
        """Return this configuration with a different model."""
        return replace(self, model=model)

    def with_store(
        self,
        store_type: str | None,
        store_path: str | None,
        collection_name: str,
    ) -> "SessionConfig":
        """Return this configuration with different retrieval storage."""
        return replace(
            self,
            store_type=store_type,
            store_path=store_path,
            collection_name=collection_name,
        )

    def with_system_prompt(self, system_prompt: str) -> "SessionConfig":
        """Return this configuration with a different system prompt."""
        return replace(self, system_prompt=system_prompt)


def _ask(question: questionary.Question) -> object | None:
    """Ask a question, returning ``None`` when the user cancels."""
    return question.ask()


def _ask_text(question: questionary.Question) -> str | None:
    """Ask a text question, returning ``None`` when the user cancels."""
    answer = _ask(question)
    return None if answer is None else str(answer)


def _provider_choices(
    registry: ProviderRegistry,
) -> tuple[list[str], dict[str, ProviderSpec]]:
    providers_by_label = {provider.label: provider for provider in registry.all()}
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
    except FieldguideError as error:
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
) -> str | None:
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
    if model is None:
        return None
    if model == OTHER_MODEL:
        return _ask_text(questionary.text("Model name:"))
    return model


def _prompt_provider_and_model(
    output_stream: TextIO,
    registry: ProviderRegistry,
) -> tuple[ProviderSpec, str] | None:
    provider_choices, providers_by_label = _provider_choices(registry)
    provider_label = _ask_text(
        questionary.select("Select an LLM provider:", choices=provider_choices)
    )
    if provider_label is None:
        return None
    provider_spec = providers_by_label[provider_label]
    model = _prompt_model(provider_spec, output_stream)
    if model is None:
        return None
    return provider_spec, model


def _default_store_path(store_type: str) -> str:
    return {
        "chroma": DEFAULT_CHROMA_PATH,
        "numpy": DEFAULT_NUMPY_PATH,
        "faiss": DEFAULT_FAISS_PATH,
    }[store_type]


def _prompt_store(
    current: SessionConfig | None = None,
) -> tuple[str | None, str | None, str] | None:
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
    if choice is None:
        return None
    if choice == NONE_STORE:
        return None, None, DEFAULT_COLLECTION_NAME

    default_path = (
        current.store_path
        if current and current.store_type == choice and current.store_path
        else _default_store_path(choice)
    )
    store_path = _ask_text(questionary.text("Vector store path:", default=default_path))
    if store_path is None:
        return None
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
        if collection_name is None:
            return None
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


def index_corpus(
    corpus_path: str,
    vector_store: VectorStore,
    max_words: int,
    output_stream: TextIO = sys.stdout,
) -> IndexingResult:
    """Index a Markdown corpus and render the resulting counts."""
    result = DocumentIndexingPipeline(
        vector_store=vector_store,
        chunker=MarkdownSectionChunker(max_words=max_words),
    ).index_path(corpus_path)
    output_stream.write(
        f"Indexed {result.document_count} documents and {result.chunk_count} chunks.\n"
    )
    return result


def _build_provider(
    provider_spec: ProviderSpec,
    config: SessionConfig,
    history: list[ChatMessage] | None = None,
) -> LLMProvider:
    return provider_spec.build_provider(
        config.model,
        message_history=history,
        system_prompt=config.system_prompt,
    )


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


def _show_cancelled(console: Console) -> None:
    """Render a non-fatal mid-session cancellation."""
    console.print("[yellow]Configuration change cancelled.[/yellow]")


def _show_sources(console: Console, sources: Sequence[VectorSearchResult]) -> None:
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
    registry: ProviderRegistry | None = None,
) -> SessionConfig:
    """Run the rich chat loop and support live session reconfiguration."""
    provider_registry = registry or registry_from_environment()
    if config is None:
        config = SessionConfig(
            system_prompt=system_prompt
            or provider.system_prompt
            or DEFAULT_SYSTEM_PROMPT,
            store_type=None,
            store_path=None,
        )
    provider.set_system_prompt(config.system_prompt)
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
            return config
        user_input = user_input.strip()
        if not user_input:
            continue

        command = user_input.lower()
        if command in {":quit", ":q", ":exit", "quit", "exit"}:
            console.print("[cyan]Goodbye.[/cyan]")
            return config
        if command == ":help":
            _show_help(console)
            continue
        if command == ":config":
            _show_config(console, config)
            continue
        if command == ":history":
            write_history(provider, output_stream)
            continue
        if command == ":clear":
            provider.clear_history()
            console.print("[green]History cleared.[/green]")
            continue

        if command == ":provider":
            selection = _prompt_provider_and_model(
                output_stream,
                provider_registry,
            )
            if selection is None:
                _show_cancelled(console)
                continue
            provider_spec, model = selection
            updated = config.with_provider(provider_spec.name, model)
            provider = _build_provider(
                provider_spec, updated, history=provider.get_history()
            )
            config = updated
            bot = KnowledgeBot(provider, vector_store)
            _show_config(console, config)
            continue
        if command == ":model":
            provider_spec = provider_registry.get(config.provider_name)
            model = _prompt_model(provider_spec, output_stream, config.model)
            if model is None:
                _show_cancelled(console)
                continue
            updated = config.with_model(model)
            provider = _build_provider(
                provider_spec, updated, history=provider.get_history()
            )
            config = updated
            bot = KnowledgeBot(provider, vector_store)
            _show_config(console, config)
            continue
        if command == ":store":
            selection = _prompt_store(config)
            if selection is None:
                _show_cancelled(console)
                continue
            store_type, store_path, collection_name = selection
            updated = config.with_store(
                store_type,
                store_path,
                collection_name,
            )
            vector_store = _build_store(updated)
            config = updated
            bot = KnowledgeBot(provider, vector_store)
            _show_config(console, config)
            continue
        if command == ":system":
            prompt = _ask_text(
                questionary.text("System prompt:", default=config.system_prompt)
            )
            if prompt is None:
                _show_cancelled(console)
                continue
            config = config.with_system_prompt(prompt)
            provider.set_system_prompt(prompt)
            _show_config(console, config)
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
    registry: ProviderRegistry | None = None,
) -> bool:
    """Configure and start an interactive retrieval-grounded chat session."""
    provider_registry = registry or registry_from_environment()
    console = _console(output_stream)
    _show_banner(console)
    provider_selection = _prompt_provider_and_model(
        output_stream,
        provider_registry,
    )
    if provider_selection is None:
        return False
    provider_spec, model = provider_selection
    store_selection = _prompt_store()
    if store_selection is None:
        return False
    store_type, store_path, collection_name = store_selection
    system_prompt = _ask_text(
        questionary.text("System prompt:", default=DEFAULT_SYSTEM_PROMPT)
    )
    if system_prompt is None:
        return False
    config = SessionConfig(
        provider_name=provider_spec.name,
        model=model,
        store_type=store_type,
        store_path=store_path,
        collection_name=collection_name,
        system_prompt=system_prompt,
    )
    vector_store = _build_store(config)

    should_ingest = False
    if vector_store is not None:
        ingestion_answer = _ask(
            questionary.confirm("Run ingestion pipeline?", default=False)
        )
        if ingestion_answer is None:
            return False
        should_ingest = bool(ingestion_answer)

    if should_ingest:
        corpus_path = _ask_text(questionary.text("Corpus path:"))
        if corpus_path is None:
            return False
        max_words_answer = _ask_text(
            questionary.text(
                "Maximum words per chunk:",
                default=str(DEFAULT_MAX_WORDS),
                validate=_is_positive_integer,
            )
        )
        if max_words_answer is None:
            return False
        max_words = int(max_words_answer)
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
        registry=provider_registry,
    )
    return True


def main() -> None:
    """Run the interactive chat wizard entry point."""
    load_dotenv()
    try:
        completed = run_wizard(registry=registry_from_environment())
        if completed:
            return
        print("Error: Wizard operation cancelled.", file=sys.stderr)
        sys.exit(1)
    except (EOFError, KeyboardInterrupt) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
