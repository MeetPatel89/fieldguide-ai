import io
import os
import unittest
from collections.abc import Sequence
from unittest.mock import Mock, patch

from fieldguide_ai import interactive
from fieldguide_ai.chat import ChatMessage, GenerationResult
from fieldguide_ai.providers import (
    OpenAIProvider,
    ProviderRegistry,
    ProviderSpec,
    get_provider,
)
from fieldguide_ai.providers.base import LLMProvider
from fieldguide_ai.vectorstore import VectorSearchResult


class Answer:
    def __init__(self, value: object) -> None:
        self.value = value

    def ask(self) -> object:
        return self.value


class FakeProvider(LLMProvider):
    def generate(self, messages: Sequence[ChatMessage]) -> GenerationResult:
        return self._record_generation(
            GenerationResult(text="response", provider="fake", model="fake-model")
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
        provider = get_provider("openai").build_provider("gpt-5-mini")

        self.assertIsInstance(provider, OpenAIProvider)
        self.assertEqual(provider.model, "gpt-5-mini")
        openai_client_type.assert_called_once_with(api_key="test-key")

    def test_unknown_provider_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported LLM provider: unknown"):
            get_provider("unknown")


class InteractiveWizardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = FakeProvider()
        self.backend = Mock()
        self.backend.build_provider.return_value = self.provider
        self.backend.list_models.return_value = ["gpt-5-mini", "gpt-5-nano"]
        self.provider_spec = ProviderSpec(
            name="openai",
            label="OpenAI",
            models=("gpt-5-nano", "gpt-5-mini"),
            default_model="gpt-5-nano",
            backend=self.backend,
        )
        self.registry = ProviderRegistry([self.provider_spec])

    def test_session_reconfiguration_returns_a_new_validated_value(self) -> None:
        config = interactive.SessionConfig()

        updated = config.with_store(None, None, "documents").with_model("new-model")

        self.assertEqual(config.model, "gpt-5-nano")
        self.assertEqual(config.store_type, "chroma")
        self.assertEqual(updated.model, "new-model")
        self.assertIsNone(updated.store_type)

    def test_session_rejects_partial_store_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a path"):
            interactive.SessionConfig(store_type="faiss", store_path=None)

    def test_wizard_builds_configured_store_for_retrieval(self) -> None:
        output_stream = io.StringIO()
        embedding_provider = Mock()
        vector_store = Mock()

        with (
            patch.object(
                interactive.questionary,
                "select",
                side_effect=[Answer("OpenAI"), Answer("gpt-5-mini"), Answer("numpy")],
            ),
            patch.object(
                interactive.questionary,
                "text",
                side_effect=[Answer("custom.npz"), Answer("Use indexed knowledge.")],
            ),
            patch.object(
                interactive.questionary, "confirm", return_value=Answer(False)
            ),
            patch.object(
                interactive, "OpenAIEmbeddingProvider", return_value=embedding_provider
            ),
            patch.object(
                interactive, "build_vector_store", return_value=vector_store
            ) as build_vector_store,
            patch.object(interactive, "run_chat_loop") as run_chat_loop,
        ):
            interactive.run_wizard(
                output_stream=output_stream,
                registry=self.registry,
            )

        build_vector_store.assert_called_once_with(
            provider_name="numpy",
            embedding_provider=embedding_provider,
            path="custom.npz",
            collection_name="documents",
        )
        config = run_chat_loop.call_args.kwargs["config"]
        self.assertEqual(config.model, "gpt-5-mini")
        self.assertEqual(config.store_type, "numpy")
        self.assertEqual(config.system_prompt, "Use indexed knowledge.")
        self.assertIs(run_chat_loop.call_args.kwargs["vector_store"], vector_store)

    def test_wizard_supports_plain_chat_without_embeddings(self) -> None:
        output_stream = io.StringIO()

        with (
            patch.object(
                interactive.questionary,
                "select",
                side_effect=[
                    Answer("OpenAI"),
                    Answer("gpt-5-nano"),
                    Answer(interactive.NONE_STORE),
                ],
            ),
            patch.object(
                interactive.questionary,
                "text",
                return_value=Answer("Plain chat prompt"),
            ),
            patch.object(interactive.questionary, "confirm") as confirm,
            patch.object(interactive, "OpenAIEmbeddingProvider") as embedding_type,
            patch.object(interactive, "run_chat_loop") as run_chat_loop,
        ):
            interactive.run_wizard(
                output_stream=output_stream,
                registry=self.registry,
            )

        embedding_type.assert_not_called()
        confirm.assert_not_called()
        config = run_chat_loop.call_args.kwargs["config"]
        self.assertIsNone(config.store_type)
        self.assertIsNone(run_chat_loop.call_args.kwargs["vector_store"])

    def test_wizard_runs_ingestion_for_faiss(self) -> None:
        output_stream = io.StringIO()
        vector_store = Mock()

        with (
            patch.object(
                interactive.questionary,
                "select",
                side_effect=[
                    Answer("OpenAI"),
                    Answer(interactive.OTHER_MODEL),
                    Answer("faiss"),
                ],
            ),
            patch.object(
                interactive.questionary,
                "text",
                side_effect=[
                    Answer("custom-model"),
                    Answer("custom-faiss"),
                    Answer("Ground answers."),
                    Answer("docs"),
                    Answer("450"),
                ],
            ),
            patch.object(interactive.questionary, "confirm", return_value=Answer(True)),
            patch.object(interactive, "OpenAIEmbeddingProvider", return_value=Mock()),
            patch.object(interactive, "build_vector_store", return_value=vector_store),
            patch.object(interactive, "index_corpus") as index_corpus,
            patch.object(interactive, "run_chat_loop"),
        ):
            interactive.run_wizard(
                output_stream=output_stream,
                registry=self.registry,
            )

        index_corpus.assert_called_once_with(
            corpus_path="docs",
            vector_store=vector_store,
            max_words=450,
            output_stream=output_stream,
        )
        self.backend.build_provider.assert_called_once_with(
            "custom-model",
            message_history=None,
            system_prompt="Ground answers.",
        )

    def test_rich_chat_displays_sources_and_raw_history(self) -> None:
        source = VectorSearchResult(
            chunk_id="DOC::0",
            content="Indexed context",
            metadata={"source_path": "docs/guide.md", "section_path": "Guide > Help"},
            distance=0.0,
        )
        vector_store = Mock()
        vector_store.query.return_value = [source]
        config = interactive.SessionConfig(
            system_prompt="Use sources.", store_type="faiss", store_path="index"
        )
        output_stream = io.StringIO()

        interactive.run_chat_loop(
            self.provider,
            input_stream=io.StringIO("What is indexed?\n:history\n:config\n:quit\n"),
            output_stream=output_stream,
            config=config,
            vector_store=vector_store,
            registry=self.registry,
        )

        output = output_stream.getvalue()
        self.assertIn("response", output)
        self.assertIn("docs/guide.md", output)
        self.assertIn("What is indexed?", output)
        self.assertIn("Session configuration", output)
        self.assertEqual(self.provider.get_history()[0].content, "What is indexed?")

    def test_chat_commands_rebuild_components_and_preserve_history(self) -> None:
        self.provider = FakeProvider(
            message_history=[ChatMessage(role="user", content="Earlier question")]
        )
        rebuilt_for_model = FakeProvider(
            message_history=[ChatMessage(role="user", content="Earlier question")]
        )
        rebuilt_for_provider = FakeProvider()
        self.backend.build_provider.side_effect = [rebuilt_for_model]
        anthropic_backend = Mock()
        anthropic_backend.build_provider.return_value = rebuilt_for_provider
        anthropic_backend.list_models.return_value = ["claude-test"]
        anthropic_spec = ProviderSpec(
            name="anthropic",
            label="Anthropic",
            models=("claude-test",),
            default_model="claude-test",
            backend=anthropic_backend,
        )
        config = interactive.SessionConfig(system_prompt="Original prompt")
        output_stream = io.StringIO()

        registry = ProviderRegistry([self.provider_spec, anthropic_spec])

        with (
            patch.object(
                interactive.questionary,
                "select",
                side_effect=[
                    Answer("gpt-5-mini"),
                    Answer(interactive.NONE_STORE),
                    Answer("Anthropic"),
                    Answer("claude-test"),
                ],
            ),
            patch.object(
                interactive.questionary,
                "text",
                return_value=Answer("Updated prompt"),
            ),
        ):
            updated_config = interactive.run_chat_loop(
                self.provider,
                input_stream=io.StringIO(":model\n:store\n:system\n:provider\n:quit\n"),
                output_stream=output_stream,
                config=config,
                vector_store=Mock(),
                registry=registry,
            )

        self.assertEqual(updated_config.provider_name, "anthropic")
        self.assertEqual(updated_config.model, "claude-test")
        self.assertIsNone(updated_config.store_type)
        self.assertEqual(updated_config.system_prompt, "Updated prompt")
        anthropic_backend.build_provider.assert_called_once_with(
            "claude-test",
            message_history=[ChatMessage(role="user", content="Earlier question")],
            system_prompt="Updated prompt",
        )

    def test_mid_session_cancellation_is_an_explicit_non_fatal_result(self) -> None:
        output_stream = io.StringIO()

        with patch.object(
            interactive.questionary,
            "select",
            return_value=Answer(None),
        ):
            config = interactive.run_chat_loop(
                self.provider,
                input_stream=io.StringIO(":model\n:quit\n"),
                output_stream=output_stream,
                config=interactive.SessionConfig(
                    store_type=None,
                    store_path=None,
                ),
                registry=self.registry,
            )

        self.assertEqual(config.model, "gpt-5-nano")
        self.assertIn("Configuration change cancelled.", output_stream.getvalue())

    @patch.object(
        interactive,
        "run_wizard",
        return_value=False,
    )
    @patch.object(interactive, "registry_from_environment")
    @patch.object(interactive, "load_dotenv")
    def test_main_exits_cleanly_when_cancelled(
        self,
        load_dotenv: Mock,
        registry_from_environment: Mock,
        run_wizard: Mock,
    ) -> None:
        with (
            patch.object(interactive.sys, "stderr"),
            self.assertRaisesRegex(SystemExit, "1"),
        ):
            interactive.main()

        load_dotenv.assert_called_once_with()
        registry_from_environment.assert_called_once_with()
        run_wizard.assert_called_once_with(
            registry=registry_from_environment.return_value
        )
