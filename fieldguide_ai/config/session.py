"""Validated configuration for an interactive chat session."""

from dataclasses import dataclass, replace

from fieldguide_ai.errors import ConfigurationError
from fieldguide_ai.vectorstore import DEFAULT_CHROMA_PATH, DEFAULT_COLLECTION_NAME

DEFAULT_SYSTEM_PROMPT = (
    "You are a domain expert in incident reports for Nautilus Financial Services. "
    "You are capable of answering user queries about issues/incidents for Nautilus "
    "Financial Services without making up stuff or referring to external knowledge "
    "not strictly within company corpus."
)


@dataclass(frozen=True)
class SessionConfig:
    """Validated value object describing an interactive chat session."""

    provider_name: str = "openai"
    model: str = "gpt-5-nano"
    store_type: str | None = "chroma"
    store_path: str | None = DEFAULT_CHROMA_PATH
    collection_name: str = DEFAULT_COLLECTION_NAME
    system_prompt: str = ""
    top_k: int = 5

    def __post_init__(self) -> None:
        """Prevent partially configured sessions."""
        if not self.provider_name.strip():
            raise ConfigurationError("provider name must not be blank")
        if not self.model.strip():
            raise ConfigurationError("model must not be blank")
        if self.store_type not in {None, "chroma", "numpy", "faiss"}:
            raise ConfigurationError(f"unsupported vector store: {self.store_type}")
        if self.store_type is None and self.store_path is not None:
            raise ConfigurationError("plain chat cannot have a vector-store path")
        if self.store_type is not None and not (self.store_path or "").strip():
            raise ConfigurationError("a configured vector store requires a path")
        if not self.collection_name.strip():
            raise ConfigurationError("collection name must not be blank")
        if self.top_k <= 0:
            raise ConfigurationError("top_k must be greater than zero")

    def with_provider(self, provider_name: str, model: str) -> "SessionConfig":
        """Return this configuration with a different provider and model."""
        return replace(self, provider_name=provider_name, model=model)

    def with_model(self, model: str) -> "SessionConfig":
        """Return this configuration with a different model."""
        return replace(self, model=model)

    def with_store(
        self,
        store_type: str | None,
        store_path: str | None,
        collection_name: str,
    ) -> "SessionConfig":
        """Return this configuration with different retrieval storage."""
        return replace(
            self,
            store_type=store_type,
            store_path=store_path,
            collection_name=collection_name,
        )

    def with_system_prompt(self, system_prompt: str) -> "SessionConfig":
        """Return this configuration with a different system prompt."""
        return replace(self, system_prompt=system_prompt)
