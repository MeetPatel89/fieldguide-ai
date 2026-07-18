"""Registry of LLM providers for Fieldguide AI."""

import os
from collections.abc import Callable
from dataclasses import dataclass

from fieldguide_ai.providers.anthropic_provider import AnthropicProvider
from fieldguide_ai.providers.base import LLMProvider
from fieldguide_ai.providers.openai_provider import OpenAIProvider

OPENAI_DEFAULT_MODEL = "gpt-5-nano"
ANTHROPIC_DEFAULT_MODEL = "claude-haiku-4-5-20251001"


@dataclass(frozen=True)
class ProviderSpec:
    """Configuration needed to present and construct an LLM provider."""

    name: str
    label: str
    models: tuple[str, ...]
    default_model: str
    factory: Callable[[str], LLMProvider]
    model_loader: Callable[[], list[str]] | None = None

    def available_models(self) -> tuple[str, ...]:
        """Return provider-discovered models, or configured fallback models."""
        if self.model_loader is None:
            return self.models

        discovered_models = tuple(self.model_loader())
        return discovered_models or self.models


def _build_openai(model: str) -> LLMProvider:
    return OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=model,
    )


def _build_anthropic(model: str) -> LLMProvider:
    return AnthropicProvider(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model=model,
    )


def _list_openai_models() -> list[str]:
    return _build_openai(OPENAI_DEFAULT_MODEL).list_models()


def _list_anthropic_models() -> list[str]:
    return _build_anthropic(ANTHROPIC_DEFAULT_MODEL).list_models()


PROVIDERS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        name="openai",
        label="OpenAI",
        models=("gpt-5-nano", "gpt-5-mini", "gpt-4o-mini"),
        default_model=OPENAI_DEFAULT_MODEL,
        factory=_build_openai,
        model_loader=_list_openai_models,
    ),
    "anthropic": ProviderSpec(
        name="anthropic",
        label="Anthropic",
        models=("claude-haiku-4-5-20251001", "claude-sonnet-5"),
        default_model=ANTHROPIC_DEFAULT_MODEL,
        factory=_build_anthropic,
        model_loader=_list_anthropic_models,
    ),
}


def get_provider(name: str) -> ProviderSpec:
    """Look up a provider specification by its registry name."""
    try:
        return PROVIDERS[name]
    except KeyError:
        raise ValueError(f"unsupported LLM provider: {name}") from None


def build_provider(name: str, model: str | None = None) -> LLMProvider:
    """Construct a registered provider, using its default model when omitted."""
    provider = get_provider(name)
    return provider.factory(model or provider.default_model)
