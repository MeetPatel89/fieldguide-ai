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
from fieldguide_ai.demo import build_system_message
from fieldguide_ai.providers import PROVIDERS, ProviderSpec
from fieldguide_ai.vectorstore import DEFAULT_COLLECTION_NAME, OpenAIEmbeddingProvider

OTHER_MODEL = "Other (type a model name)"
DEFAULT_MAX_WORDS = 900


class WizardCancelled(Exception):
    """Raised internally when an interactive prompt is cancelled."""


def _ask(question):
    answer = question.ask()
    if answer is None:
        raise WizardCancelled
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
    provider_choices, providers_by_label = _provider_choices()
    provider_label = _ask(
        questionary.select("Select an LLM provider:", choices=provider_choices)
    )
    provider_spec = providers_by_label[provider_label]

    model = _ask(
        questionary.select(
            "Select a model:",
            choices=[*provider_spec.models, OTHER_MODEL],
            default=provider_spec.default_model,
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
            default=build_system_message().content,
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
    load_dotenv()
    try:
        run_wizard()
    except EOFError, KeyboardInterrupt, WizardCancelled:
        return


if __name__ == "__main__":
    main()
