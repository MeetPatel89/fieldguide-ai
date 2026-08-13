"""Tests for the LLM provider registry and runtime composition."""

import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, Mock, patch

from model_runtime import ChatSession, ProviderUnavailableError

from fieldguide_ai.providers import (
    ProviderRegistry,
    ProviderSpec,
    build_provider,
    create_provider_registry,
    get_provider,
)
from tests.session_fakes import (
    FakeChatModel,
    RecordingAdapterFactory,
    selected_model,
)


def fake_spec(
    adapter: FakeChatModel | None = None,
    *,
    name: str = "fake",
    label: str = "Fake",
    api_key: str | None = "test-key",
) -> tuple[ProviderSpec, RecordingAdapterFactory]:
    """Build a provider spec and its recording adapter factory."""
    selected_adapter = adapter or FakeChatModel(discovered_models=("fake-model",))
    factory = RecordingAdapterFactory(selected_adapter)
    return (
        ProviderSpec(
            name=name,
            label=label,
            models=("fake-model", "fallback-model"),
            default_model="fake-model",
            adapter_factory=factory,
            api_key=api_key,
            credential_name="FAKE_API_KEY",
        ),
        factory,
    )


class ProviderRegistryTest(unittest.TestCase):
    """Verify provider metadata, discovery, and session construction."""

    def test_anthropic_default_is_a_registered_api_model_id(self) -> None:
        spec = get_provider("anthropic")

        self.assertEqual(spec.default_model, "claude-haiku-4-5-20251001")
        self.assertIn(spec.default_model, spec.models)

    def test_available_models_prefers_adapter_discovery(self) -> None:
        spec, factory = fake_spec(
            FakeChatModel(discovered_models=("model-b", "model-a", "model-b"))
        )

        models = spec.available_models()

        self.assertEqual(models, ("model-b", "model-a"))
        self.assertEqual(factory.api_keys, ["test-key"])

    def test_available_models_uses_fallback_when_discovery_is_empty(self) -> None:
        spec, _ = fake_spec(FakeChatModel(discovered_models=()))

        self.assertEqual(spec.available_models(), spec.models)

    def test_discovery_preserves_normalized_runtime_errors(self) -> None:
        discovery_error = ProviderUnavailableError(
            "discovery failed",
            retryable=False,
            provider="fake",
        )
        spec, _ = fake_spec(FakeChatModel(discovery_error=discovery_error))

        with self.assertRaises(ProviderUnavailableError) as raised:
            spec.available_models()

        self.assertIs(raised.exception, discovery_error)

    def test_registry_rejects_duplicate_names(self) -> None:
        first, _ = fake_spec()
        second = replace(first, label="Second")

        with self.assertRaisesRegex(ValueError, "duplicate provider registration"):
            ProviderRegistry([first, second])

    def test_provider_spec_rejects_an_unselectable_default(self) -> None:
        with self.assertRaisesRegex(ValueError, "is not a fallback model"):
            ProviderSpec(
                name="invalid",
                label="Invalid",
                models=("available",),
                default_model="missing",
                adapter_factory=RecordingAdapterFactory(FakeChatModel()),
                api_key="test-key",
            )

    def test_provider_spec_requires_api_key_before_adapter_construction(self) -> None:
        spec, factory = fake_spec(api_key=None)

        with self.assertRaisesRegex(ValueError, "FAKE_API_KEY is not set"):
            spec.build_provider()

        self.assertEqual(factory.api_keys, [])

    @patch("fieldguide_ai.providers.registry.OpenAIAdapter")
    def test_openai_discovery_uses_async_runtime_adapter(
        self,
        adapter_type: Mock,
    ) -> None:
        adapter_type.return_value.list_models = AsyncMock(
            return_value=["model-a", "model-b"]
        )
        registry = create_provider_registry(
            openai_api_key="test-key",
            anthropic_api_key=None,
        )

        models = registry.get("openai").available_models()

        self.assertEqual(models, ("model-a", "model-b"))
        adapter_type.assert_called_once_with(api_key="test-key")

    def test_factory_builds_chat_session_for_selected_model(self) -> None:
        spec, factory = fake_spec()
        registry = ProviderRegistry([spec])

        provider = build_provider("fake", "fallback-model", registry)

        self.assertIsInstance(provider, ChatSession)
        self.assertEqual(provider.route, "fake")
        self.assertEqual(selected_model(provider), "fallback-model")
        self.assertEqual(factory.api_keys, ["test-key"])

    def test_unknown_provider_raises_configuration_error(self) -> None:
        spec, _ = fake_spec()

        with self.assertRaisesRegex(ValueError, "unsupported LLM provider: unknown"):
            get_provider("unknown", ProviderRegistry([spec]))


if __name__ == "__main__":
    unittest.main()
