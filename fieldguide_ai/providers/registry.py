"""Validated registry of LLM provider specifications."""

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from fieldguide_ai.chat import ChatMessage
from fieldguide_ai.errors import ConfigurationError, ProviderNotFoundError
from fieldguide_ai.providers.anthropic_provider import AnthropicBackend
from fieldguide_ai.providers.base import LLMProvider, ProviderBackend
from fieldguide_ai.providers.openai_provider import OpenAIBackend

OPENAI_DEFAULT_MODEL = "gpt-5-nano"
ANTHROPIC_DEFAULT_MODEL = "claude-haiku-4-5-20251001"


@dataclass(frozen=True)
class ProviderSpec:
    """Validated presentation metadata and backend for one LLM provider."""

    name: str
    label: str
    models: tuple[str, ...]
    default_model: str
    backend: ProviderBackend

    def __post_init__(self) -> None:
        """Reject incomplete provider registrations."""
        name = self.name.strip()
        label = self.label.strip()
        models = tuple(model.strip() for model in self.models)
        default_model = self.default_model.strip()
        if not name:
            raise ConfigurationError("provider name must not be blank")
        if not label:
            raise ConfigurationError("provider label must not be blank")
        if not models:
            raise ConfigurationError(f"provider {name!r} needs fallback models")
        if len(set(models)) != len(models):
            raise ConfigurationError(f"provider {name!r} has duplicate fallback models")
        if any(not model for model in models):
            raise ConfigurationError(f"provider {name!r} has a blank fallback model")
        if default_model not in models:
            raise ConfigurationError(
                f"default model {default_model!r} is not a fallback model for "
                f"provider {name!r}"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "default_model", default_model)

    def available_models(self) -> tuple[str, ...]:
        """Return provider-discovered models, or configured fallback models."""
        discovered_models = tuple(
            dict.fromkeys(
                model.strip() for model in self.backend.list_models() if model.strip()
            )
        )
        return discovered_models or self.models

    def build_provider(
        self,
        model: str | None = None,
        *,
        message_history: Sequence[ChatMessage] | None = None,
        system_prompt: str | None = None,
    ) -> LLMProvider:
        """Build a configured chat session through this provider's backend."""
        selected_model = model or self.default_model
        if not selected_model.strip():
            raise ConfigurationError("model must not be blank")
        return self.backend.build_provider(
            selected_model,
            message_history=message_history,
            system_prompt=system_prompt,
        )


class ProviderRegistry:
    """Immutable, validated collection of available LLM providers."""

    def __init__(self, providers: Iterable[ProviderSpec]) -> None:
        providers_by_name: dict[str, ProviderSpec] = {}
        labels: set[str] = set()
        for provider in providers:
            if provider.name in providers_by_name:
                raise ConfigurationError(
                    f"duplicate provider registration: {provider.name}"
                )
            if provider.label in labels:
                raise ConfigurationError(f"duplicate provider label: {provider.label}")
            providers_by_name[provider.name] = provider
            labels.add(provider.label)
        if not providers_by_name:
            raise ConfigurationError("at least one provider must be registered")
        self._providers: Mapping[str, ProviderSpec] = MappingProxyType(
            providers_by_name
        )

    def get(self, name: str) -> ProviderSpec:
        """Return a provider specification by registry name."""
        try:
            return self._providers[name]
        except KeyError:
            raise ProviderNotFoundError(f"unsupported LLM provider: {name}") from None

    def all(self) -> tuple[ProviderSpec, ...]:
        """Return provider specifications in registration order."""
        return tuple(self._providers.values())


def create_provider_registry(
    *,
    openai_api_key: str | None,
    anthropic_api_key: str | None,
) -> ProviderRegistry:
    """Compose the built-in provider registry from explicit credentials."""
    return ProviderRegistry(
        [
            ProviderSpec(
                name="openai",
                label="OpenAI",
                models=("gpt-5-nano", "gpt-5-mini", "gpt-4o-mini"),
                default_model=OPENAI_DEFAULT_MODEL,
                backend=OpenAIBackend(api_key=openai_api_key),
            ),
            ProviderSpec(
                name="anthropic",
                label="Anthropic",
                models=("claude-haiku-4-5-20251001", "claude-sonnet-5"),
                default_model=ANTHROPIC_DEFAULT_MODEL,
                backend=AnthropicBackend(api_key=anthropic_api_key),
            ),
        ]
    )


def registry_from_environment() -> ProviderRegistry:
    """Build the default registry at an application composition boundary."""
    return create_provider_registry(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )


def get_provider(
    name: str,
    registry: ProviderRegistry | None = None,
) -> ProviderSpec:
    """Look up a provider in an explicit registry or a fresh default registry."""
    return (registry or registry_from_environment()).get(name)


def build_provider(
    name: str,
    model: str | None = None,
    registry: ProviderRegistry | None = None,
) -> LLMProvider:
    """Construct a provider from an explicit or freshly composed registry."""
    return get_provider(name, registry).build_provider(model)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    spec = ProviderSpec(
        name="openai",
        label="OpenAI",
        models=("gpt-5-nano", "gpt-5-mini", "gpt-4o-mini"),
        default_model=OPENAI_DEFAULT_MODEL,
        backend=OpenAIBackend(api_key=os.getenv("OPENAI_API_KEY")),
    )
    spec.available_models()
