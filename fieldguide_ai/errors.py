"""Application-level exceptions exposed by Fieldguide AI boundaries."""


class FieldguideError(Exception):
    """Base class for errors intentionally exposed by Fieldguide AI."""


class ConfigurationError(FieldguideError, ValueError):
    """Raised when application configuration is missing or invalid."""


class ProviderNotFoundError(ConfigurationError):
    """Raised when an LLM provider name is not registered."""


class ProviderOperationError(FieldguideError, RuntimeError):
    """Raised when an LLM provider operation fails."""


class ProviderDiscoveryError(ProviderOperationError):
    """Raised when the available models cannot be retrieved."""


class ProviderInitializationError(ProviderOperationError):
    """Raised when an LLM provider client cannot be initialized."""


class ProviderGenerationError(ProviderOperationError):
    """Raised when an LLM response cannot be generated."""


class EmbeddingError(FieldguideError, RuntimeError):
    """Raised when an embedding provider cannot embed a text batch."""


class DocumentLoadError(FieldguideError, RuntimeError):
    """Raised when Markdown documents cannot be read from storage."""


class InvalidVectorStoreError(FieldguideError, ValueError):
    """Raised when a persisted vector store is incomplete or malformed."""


class VectorStoreOperationError(FieldguideError, RuntimeError):
    """Raised when vector-store infrastructure cannot complete an operation."""
