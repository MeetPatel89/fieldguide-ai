import io
import os
import unittest
from unittest.mock import Mock, patch

from fieldguide_ai import interactive
from fieldguide_ai.generation import GenerationResult
from fieldguide_ai.messages import ChatMessage
from fieldguide_ai.providers import OpenAIProvider, ProviderSpec, get_provider
from fieldguide_ai.providers.base import LLMProvider


class Answer:
    def __init__(self, value) -> None:
        self.value = value

    def ask(self):
        return self.value


class FakeProvider(LLMProvider):
    def list_models(self) -> list[str]:
        return ["gpt-5-nano", "gpt-5-mini"]

    def generate(self, messages: list[ChatMessage]) -> GenerationResult:
        return self._record_generation(
            GenerationResult(
                text="response",
                provider="fake",
                model="fake-model",
            )
        )


class ProviderRegistryTest(unittest.TestCase):
    def test_openai_provider_is_registered(self) -> None:
        provider = get_provider("openai")

        self.assertEqual(provider.label, "OpenAI")
        self.assertEqual(provider.default_model, "gpt-5-nano")
        self.assertIn(provider.default_model, provider.models)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("fieldguide_ai.providers.openai_provider.OpenAI")
    def test_factory_builds_openai_provider(self, openai_client_type: Mock) -> None:
        provider = get_provider("openai").factory("gpt-5-mini")

        self.assertIsInstance(provider, OpenAIProvider)
        self.assertEqual(provider.model, "gpt-5-mini")
        openai_client_type.assert_called_once_with(api_key="test-key")

    def test_unknown_provider_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported LLM provider: unknown"):
            get_provider("unknown")


class InteractiveWizardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = FakeProvider()
        self.factory = Mock(return_value=self.provider)
        self.model_loader = Mock(return_value=["gpt-5-mini", "gpt-5-nano"])
        self.provider_spec = ProviderSpec(
            name="openai",
            label="OpenAI",
            models=("gpt-5-nano", "gpt-5-mini"),
            default_model="gpt-5-nano",
            factory=self.factory,
            model_loader=self.model_loader,
        )

    def test_wizard_maps_numpy_configuration_without_ingestion(self) -> None:
        input_stream = io.StringIO(":quit\n")
        output_stream = io.StringIO()

        with (
            patch.dict(
                interactive.PROVIDERS, {"openai": self.provider_spec}, clear=True
            ),
            patch.object(
                interactive.questionary,
                "select",
                side_effect=[Answer("OpenAI"), Answer("gpt-5-mini"), Answer("numpy")],
            ) as select,
            patch.object(
                interactive.questionary,
                "text",
                side_effect=[
                    Answer("custom-index.npz"),
                    Answer("Custom system prompt"),
                ],
            ),
            patch.object(
                interactive.questionary, "confirm", return_value=Answer(False)
            ),
            patch.object(interactive, "OpenAIEmbeddingProvider") as embedding_type,
            patch.object(interactive, "build_vector_store") as build_vector_store,
            patch.object(interactive, "index_corpus") as index_corpus,
            patch.object(interactive, "run_chat_loop") as run_chat_loop,
        ):
            interactive.run_wizard(
                input_stream=input_stream,
                output_stream=output_stream,
            )

        embedding_type.assert_not_called()
        build_vector_store.assert_not_called()
        index_corpus.assert_not_called()
        self.model_loader.assert_called_once_with()
        self.assertEqual(
            select.call_args_list[1].kwargs,
            {
                "choices": [
                    "gpt-5-mini",
                    "gpt-5-nano",
                    interactive.OTHER_MODEL,
                ],
                "default": "gpt-5-nano",
            },
        )
        self.factory.assert_called_once_with("gpt-5-mini")
        run_chat_loop.assert_called_once_with(
            self.provider,
            input_stream=input_stream,
            output_stream=output_stream,
            system_prompt="Custom system prompt",
        )

    def test_wizard_accepts_custom_model_and_runs_ingestion(self) -> None:
        embedding_provider = Mock()
        vector_store = Mock()
        output_stream = io.StringIO()

        with (
            patch.dict(
                interactive.PROVIDERS, {"openai": self.provider_spec}, clear=True
            ),
            patch.object(
                interactive.questionary,
                "select",
                side_effect=[
                    Answer("OpenAI"),
                    Answer(interactive.OTHER_MODEL),
                    Answer("chroma"),
                ],
            ),
            patch.object(
                interactive.questionary,
                "text",
                side_effect=[
                    Answer("custom-model"),
                    Answer("custom-chroma"),
                    Answer("custom-collection"),
                    Answer("Use indexed knowledge."),
                    Answer("docs"),
                    Answer("450"),
                ],
            ),
            patch.object(interactive.questionary, "confirm", return_value=Answer(True)),
            patch.object(
                interactive,
                "OpenAIEmbeddingProvider",
                return_value=embedding_provider,
            ),
            patch.object(
                interactive,
                "build_vector_store",
                return_value=vector_store,
            ) as build_vector_store,
            patch.object(interactive, "index_corpus") as index_corpus,
            patch.object(interactive, "run_chat_loop") as run_chat_loop,
        ):
            interactive.run_wizard(output_stream=output_stream)

        build_vector_store.assert_called_once_with(
            provider_name="chroma",
            embedding_provider=embedding_provider,
            path="custom-chroma",
            collection_name="custom-collection",
        )
        index_corpus.assert_called_once_with(
            corpus_path="docs",
            vector_store=vector_store,
            max_words=450,
            output_stream=output_stream,
        )
        self.factory.assert_called_once_with("custom-model")
        self.assertEqual(
            run_chat_loop.call_args.kwargs["system_prompt"],
            "Use indexed knowledge.",
        )

    @patch.object(
        interactive,
        "run_wizard",
        side_effect=interactive.WizardCancelledError("Wizard operation cancelled."),
    )
    @patch.object(interactive, "load_dotenv")
    def test_main_exits_cleanly_when_cancelled(
        self,
        load_dotenv: Mock,
        run_wizard: Mock,
    ) -> None:
        with (
            patch.object(interactive.sys, "stderr"),
            self.assertRaisesRegex(SystemExit, "1"),
        ):
            interactive.main()

        load_dotenv.assert_called_once_with()
        run_wizard.assert_called_once_with()
