"""Validated registry and model-runtime composition for LLM providers."""

import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Protocol

from model_runtime import (
    AnthropicAdapter,
    ChatModel,
    ChatSession,
    Message,
    ModelCatalog,
    ModelRouter,
    ModelRuntime,
    OpenAIAdapter,
    run_sync,
)

from fieldguide_ai.errors import ConfigurationError, ProviderNotFoundError

OPENAI_DEFAULT_MODEL = "gpt-5-nano"
ANTHROPIC_DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class _RegistryAdapter(ChatModel, ModelCatalog, Protocol):
    """Adapter behavior needed at Fieldguide's composition boundary."""


AdapterFactory = Callable[[str], _RegistryAdapter]


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """Validated presentation, credential, and adapter metadata for a provider."""

    name: str
    label: str
    models: tuple[str, ...]
    default_model: str
    adapter_factory: AdapterFactory
    api_key: str | None = None
    credential_name: str | None = None

    def __post_init__(self) -> None:
        """Reject incomplete provider registrations."""
        name = self.name.strip()
        label = self.label.strip()
        models = tuple(model.strip() for model in self.models)
        default_model = self.default_model.strip()
        credential_name = (self.credential_name or f"{name.upper()}_API_KEY").strip()
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
        if not credential_name:
            raise ConfigurationError("credential name must not be blank")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "default_model", default_model)
        object.__setattr__(self, "credential_name", credential_name)

    def available_models(self) -> tuple[str, ...]:
        """Return provider-discovered models, or configured fallback models."""
        adapter = self._build_adapter()
        discovered = run_sync(
            adapter.list_models,
            operation_name=f"{self.name} model discovery",
        )
        discovered_models = tuple(
            dict.fromkeys(model.strip() for model in discovered if model.strip())
        )
        return discovered_models or self.models

    def build_provider(
        self,
        model: str | None = None,
        *,
        history: Sequence[Message] | None = None,
        system_prompt: str | None = None,
    ) -> ChatSession:
        """Build a conversation session around one model-runtime route."""
        selected_model = model or self.default_model
        if not selected_model.strip():
            raise ConfigurationError("model must not be blank")
        adapter = self._build_adapter()
        router = ModelRouter({self.name: (adapter, selected_model)})
        return ChatSession(
            ModelRuntime(router),
            self.name,
            history=history if history is not None else (),
            system_prompt=system_prompt,
        )

    def _build_adapter(self) -> _RegistryAdapter:
        api_key = self._require_api_key()
        try:
            return self.adapter_factory(api_key)
        except Exception as error:
            raise ConfigurationError(
                f"{self.name} client initialization failed"
            ) from error

    def _require_api_key(self) -> str:
        if not self.api_key:
            raise ConfigurationError(
                f"{self.credential_name} is not set. Add it to your .env file."
            )
        return self.api_key


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


def _create_openai_adapter(api_key: str) -> _RegistryAdapter:
    return OpenAIAdapter(api_key=api_key)


def _create_anthropic_adapter(api_key: str) -> _RegistryAdapter:
    return AnthropicAdapter(api_key=api_key)


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
                adapter_factory=_create_openai_adapter,
                api_key=openai_api_key,
                credential_name="OPENAI_API_KEY",
            ),
            ProviderSpec(
                name="anthropic",
                label="Anthropic",
                models=("claude-haiku-4-5-20251001", "claude-sonnet-5"),
                default_model=ANTHROPIC_DEFAULT_MODEL,
                adapter_factory=_create_anthropic_adapter,
                api_key=anthropic_api_key,
                credential_name="ANTHROPIC_API_KEY",
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
) -> ChatSession:
    """Construct a chat session from an explicit or freshly composed registry."""
    return get_provider(name, registry).build_provider(model)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    registry_from_environment().get("openai").available_models()
