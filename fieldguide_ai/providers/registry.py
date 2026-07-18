import os
from collections.abc import Callable
from dataclasses import dataclass

from fieldguide_ai.providers.base import LLMProvider
from fieldguide_ai.providers.openai_provider import OpenAIProvider


@dataclass(frozen=True)
class ProviderSpec:
    """Configuration needed to present and construct an LLM provider."""

    name: str
    label: str
    models: tuple[str, ...]
    default_model: str
    factory: Callable[[str], LLMProvider]


def _build_openai(model: str) -> OpenAIProvider:
    return OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=model,
    )


PROVIDERS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        name="openai",
        label="OpenAI",
        models=("gpt-5-nano", "gpt-5-mini", "gpt-4o-mini"),
        default_model="gpt-5-nano",
        factory=_build_openai,
    )
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
