"""Interactive chat wizard for Fieldguide AI."""

import os
import sys
from typing import TextIO

import questionary
from dotenv import load_dotenv

from fieldguide_ai.cli import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_NUMPY_PATH,
    build_vector_store,
    index_corpus,
    run_chat_loop,
)
from fieldguide_ai.demo import build_system_prompt
from fieldguide_ai.providers import PROVIDERS, ProviderSpec
from fieldguide_ai.vectorstore import DEFAULT_COLLECTION_NAME, OpenAIEmbeddingProvider

OTHER_MODEL = "Other (type a model name)"
DEFAULT_MAX_WORDS = 900


class WizardCancelledError(Exception):
    """Raised internally when a wizard operation is cancelled."""


def _ask(question: questionary.Question) -> str:
    """Ask a question and return the answer."""
    answer = question.ask()
    if answer is None:
        raise WizardCancelledError("Wizard operation cancelled.")
    return answer


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


def run_wizard(
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
) -> None:
    """Run the interactive chat wizard."""
    provider_choices, providers_by_label = _provider_choices()
    provider_label = _ask(
        questionary.select("Select an LLM provider:", choices=provider_choices)
    )
    provider_spec = providers_by_label[provider_label]

    try:
        available_models = provider_spec.available_models()
    except Exception as error:
        print(
            f"Could not load current {provider_spec.label} models: {error}. "
            "Using configured models.",
            file=output_stream,
        )
        available_models = provider_spec.models

    default_model = (
        provider_spec.default_model
        if provider_spec.default_model in available_models
        else available_models[0]
    )
    model = _ask(
        questionary.select(
            "Select a model:",
            choices=[*available_models, OTHER_MODEL],
            default=default_model,
        )
    )
    if model == OTHER_MODEL:
        model = _ask(questionary.text("Model name:"))

    vector_store_name = _ask(
        questionary.select(
            "Select a vector store:",
            choices=["chroma", "numpy"],
            default="chroma",
        )
    )
    default_store_path = (
        DEFAULT_CHROMA_PATH if vector_store_name == "chroma" else DEFAULT_NUMPY_PATH
    )
    store_path = _ask(
        questionary.text("Vector store path:", default=default_store_path)
    )

    collection_name = DEFAULT_COLLECTION_NAME
    if vector_store_name == "chroma":
        collection_name = _ask(
            questionary.text(
                "Chroma collection name:",
                default=DEFAULT_COLLECTION_NAME,
            )
        )

    system_prompt = _ask(
        questionary.text(
            "System prompt:",
            default=build_system_prompt(),
        )
    )
    should_ingest = _ask(questionary.confirm("Run ingestion pipeline?", default=False))

    if should_ingest:
        corpus_path = _ask(questionary.text("Corpus path:"))
        max_words = int(
            _ask(
                questionary.text(
                    "Maximum words per chunk:",
                    default=str(DEFAULT_MAX_WORDS),
                    validate=_is_positive_integer,
                )
            )
        )
        embedding_provider = OpenAIEmbeddingProvider(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        vector_store = build_vector_store(
            provider_name=vector_store_name,
            embedding_provider=embedding_provider,
            path=store_path,
            collection_name=collection_name,
        )
        index_corpus(
            corpus_path=corpus_path,
            vector_store=vector_store,
            max_words=max_words,
            output_stream=output_stream,
        )

    provider = provider_spec.factory(model)
    run_chat_loop(
        provider,
        input_stream=input_stream,
        output_stream=output_stream,
        system_prompt=system_prompt,
    )


def main() -> None:
    """Run main entry point for the interactive chat wizard."""
    load_dotenv()
    try:
        run_wizard()
    except (EOFError, KeyboardInterrupt, WizardCancelledError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
