"""Registry of LLM providers for Fieldguide AI."""

from dataclasses import dataclass

from fieldguide_ai.providers.anthropic_provider import AnthropicBackend
from fieldguide_ai.providers.base import LLMProvider, ProviderBackend
from fieldguide_ai.providers.openai_provider import OpenAIBackend

OPENAI_DEFAULT_MODEL = "gpt-5-nano"
ANTHROPIC_DEFAULT_MODEL = "claude-haiku-4-5-20251001"


@dataclass(frozen=True)
class ProviderSpec:
    """Configuration needed to present and construct an LLM provider."""

    name: str
    label: str
    models: tuple[str, ...]
    default_model: str
    backend: ProviderBackend

    def available_models(self) -> tuple[str, ...]:
        """Return provider-discovered models, or configured fallback models."""
        discovered_models = tuple(self.backend.list_models())
        return discovered_models or self.models

    def build_provider(self, model: str | None = None) -> LLMProvider:
        """Build a chat provider, using the configured default when omitted."""
        return self.backend.build_provider(model or self.default_model)


PROVIDERS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        name="openai",
        label="OpenAI",
        models=("gpt-5-nano", "gpt-5-mini", "gpt-4o-mini"),
        default_model=OPENAI_DEFAULT_MODEL,
        backend=OpenAIBackend(),
    ),
    "anthropic": ProviderSpec(
        name="anthropic",
        label="Anthropic",
        models=("claude-haiku-4-5-20251001", "claude-sonnet-5"),
        default_model=ANTHROPIC_DEFAULT_MODEL,
        backend=AnthropicBackend(),
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
    return get_provider(name).build_provider(model)
