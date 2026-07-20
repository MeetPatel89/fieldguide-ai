"""Tests for the LLM provider registry."""

import os
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from fieldguide_ai.providers import (
    AnthropicProvider,
    OpenAIProvider,
    build_provider,
    get_provider,
)


class ProviderRegistryTest(unittest.TestCase):
    """Verify provider metadata and construction."""

    def test_anthropic_default_is_a_registered_api_model_id(self) -> None:
        """The Anthropic default should be selectable and API-compatible."""
        spec = get_provider("anthropic")

        self.assertEqual(spec.default_model, "claude-haiku-4-5-20251001")
        self.assertIn(spec.default_model, spec.models)

    def test_available_models_prefers_provider_discovery(self) -> None:
        """Model discovery should replace the configured fallback choices."""
        discovered_models = ["model-b", "model-a"]
        backend = Mock()
        backend.list_models.return_value = discovered_models
        spec = replace(
            get_provider("anthropic"),
            backend=backend,
        )

        models = spec.available_models()

        self.assertEqual(models, ("model-b", "model-a"))

    def test_available_models_uses_fallback_when_discovery_is_empty(self) -> None:
        """An empty discovery response should retain configured choices."""
        backend = Mock()
        backend.list_models.return_value = []
        spec = replace(get_provider("anthropic"), backend=backend)

        models = spec.available_models()

        self.assertEqual(models, spec.models)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("fieldguide_ai.providers.openai_provider.OpenAI")
    @patch("fieldguide_ai.providers.openai_provider.OpenAIProvider")
    def test_openai_discovery_does_not_construct_a_chat_provider(
        self,
        provider_type: Mock,
        openai_client_type: Mock,
    ) -> None:
        """Discovery should use the SDK client without selecting a chat model."""
        openai_client_type.return_value.models.list.return_value.data = [
            Mock(id="model-b"),
            Mock(id="model-a"),
        ]

        models = get_provider("openai").available_models()

        self.assertEqual(models, ("model-a", "model-b"))
        openai_client_type.assert_called_once_with(api_key="test-key")
        provider_type.assert_not_called()

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("fieldguide_ai.providers.anthropic_provider.Anthropic")
    @patch("fieldguide_ai.providers.anthropic_provider.AnthropicProvider")
    def test_anthropic_discovery_does_not_construct_a_chat_provider(
        self,
        provider_type: Mock,
        anthropic_client_type: Mock,
    ) -> None:
        """Discovery should use the SDK client without selecting a chat model."""
        anthropic_client_type.return_value.models.list.return_value.data = [
            Mock(id="model-b"),
            Mock(id="model-a"),
        ]

        models = get_provider("anthropic").available_models()

        self.assertEqual(models, ("model-a", "model-b"))
        anthropic_client_type.assert_called_once_with(api_key="test-key")
        provider_type.assert_not_called()

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("fieldguide_ai.providers.anthropic_provider.Anthropic")
    def test_anthropic_factory_forwards_selected_model(
        self,
        anthropic_client_type: Mock,
    ) -> None:
        """The Anthropic factory should retain the selected model ID."""
        provider = build_provider("anthropic", "claude-sonnet-5")

        self.assertIsInstance(provider, AnthropicProvider)
        self.assertEqual(provider.model, "claude-sonnet-5")
        anthropic_client_type.assert_called_once_with(api_key="test-key")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("fieldguide_ai.providers.openai_provider.OpenAI")
    def test_openai_factory_forwards_selected_model(
        self,
        openai_client_type: Mock,
    ) -> None:
        """The OpenAI factory should retain the selected model ID."""
        provider = build_provider("openai", "gpt-5-mini")

        self.assertIsInstance(provider, OpenAIProvider)
        self.assertEqual(provider.model, "gpt-5-mini")
        openai_client_type.assert_called_once_with(api_key="test-key")


if __name__ == "__main__":
    unittest.main()
