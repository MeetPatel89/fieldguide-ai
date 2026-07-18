"""Tests for the LLM provider registry."""

import os
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from fieldguide_ai.providers import AnthropicProvider, build_provider, get_provider


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
        spec = replace(
            get_provider("anthropic"),
            model_loader=Mock(return_value=discovered_models),
        )

        models = spec.available_models()

        self.assertEqual(models, ("model-b", "model-a"))

    def test_available_models_uses_fallback_when_discovery_is_empty(self) -> None:
        """An empty discovery response should retain configured choices."""
        spec = replace(
            get_provider("anthropic"),
            model_loader=Mock(return_value=[]),
        )

        models = spec.available_models()

        self.assertEqual(models, spec.models)

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


if __name__ == "__main__":
    unittest.main()
